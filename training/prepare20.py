"""Prepare video datasets without random window splits or synthetic fallback.

python -m training.prepare20 manifest-rldd --root DATASET --output manifest.csv
python -m training.prepare20 build --manifest manifest.csv --model face_landmarker.task
"""
import argparse
import csv
import hashlib
import json
import importlib.metadata
import re
from pathlib import Path
import numpy as np
from core.features20 import SCHEMA, FEATURE_NAMES, make_features

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".m4v", ".mkv"}
FIELDS = ["path", "subject", "split", "label", "start_s", "end_s"]


def rldd_manifest(root, output, include_low=False):
    """Preserve original folds: 1-3 train, 4 validation, 5 test.

    Handles split clips such as subject 32/10_1.mp4 and 32/10_2.mp4.
    No labels are inferred from facial geometry.
    """
    rows, ignored = [], []
    labels = {"0": "ALERT", "10": "DROWSY", "5": "LOW_VIGILANCE"}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        fold = re.search(r"Fold([1-5])(?:_|/)", p.as_posix(), re.IGNORECASE)
        label = re.fullmatch(r"(0|5|10)(?:_\d+)?", p.stem)
        if not fold or not label or not p.parent.name.isdigit():
            ignored.append(str(p))
            continue
        if label[1] == "5" and not include_low:
            continue
        split = {"1": "train", "2": "train", "3": "train", "4": "val", "5": "test"}[fold[1]]
        rows.append(dict(path=str(p.resolve()), subject="rldd:" + p.parent.name.zfill(2),
                         split=split, label=labels[label[1]], start_s="0", end_s=""))
    if ignored:
        raise ValueError(f"Unrecognized video paths, inspect before labeling: {ignored[:5]}")
    if not rows:
        raise ValueError("No original RLDD videos found; image-only mirrors are unsupported")
    validate_rows(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest: {len(rows)} labeled videos -> {output}")


def validate_rows(rows):
    if not rows:
        raise ValueError("Empty manifest")
    subject_splits, path_splits, intervals = {}, {}, {}
    split_labels = {s: set() for s in ("train", "val", "test")}
    for r in rows:
        if any(not r.get(k) for k in ("path", "subject", "split", "label")):
            raise ValueError("Every row needs path, subject, split and label")
        split = r["split"]
        if split not in split_labels:
            raise ValueError(f"Invalid split: {split}")
        path = str(Path(r["path"]).resolve())
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        for key, mapping in [(r["subject"], subject_splits), (path, path_splits)]:
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Split leakage: {key}")
            mapping[key] = split
        start, end = float(r.get("start_s") or 0), float(r.get("end_s") or "inf")
        if not np.isfinite(start) or start < 0 or np.isnan(end) or end <= start:
            raise ValueError("Invalid segment times")
        for previous_start, previous_end in intervals.setdefault(path, []):
            if start < previous_end and previous_start < end:
                raise ValueError(f"Overlapping/duplicate labeled segments: {path}")
        intervals[path].append((start, end))
        split_labels[split].add(r["label"])
    if len(split_labels["train"]) < 2 or any(v != split_labels["train"] for v in split_labels.values()):
        raise ValueError(f"Each split must contain the same >=2 labels: {split_labels}")
    return sorted(split_labels["train"])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_signature(path, model, sample_fps):
    sample_fps = int(sample_fps) if float(sample_fps).is_integer() else float(sample_fps)
    from training.video_timestamps20 import decoded_timestamps, TIMESTAMP_VERSION, timestamp_exception
    model = Path(model)
    stat = Path(path).stat()
    signature = dict(path=str(Path(path).resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, sample_fps=sample_fps, pose_convention="camera-v2",
                     timestamp_version=TIMESTAMP_VERSION,
                     model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                     extractor_sha256=hashlib.sha256(
                         (Path(__file__).parents[1] / "core/face_analyzer.py").read_bytes()).hexdigest())
    exception = timestamp_exception(path)
    if exception:
        signature["timestamp_exception"] = exception
    return signature


def cache_filename(path, model, sample_fps):
    return hashlib.sha256(json.dumps(cache_signature(path, model, sample_fps), sort_keys=True).encode()).hexdigest() + ".npz"


def extract_video(path, model, sample_fps, cache_dir, cache_sources=(), cache_only=False):
    import cv2
    from core.face_analyzer import FaceAnalyzer
    from training.video_timestamps20 import decoded_timestamps, timestamp_exception
    exception = timestamp_exception(path)
    signature = cache_signature(path, model, sample_fps)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key}.npz"
    candidates = [cached] + [Path(root) / cached.name for root in cache_sources]
    if not any(p.exists() for p in candidates):
        # A remount may change mtime. Only portable caches with a matching full
        # video checksum may ignore that field; legacy files need explicit migration.
        comparable = {k:v for k,v in signature.items() if k != 'mtime_ns'}
        source_digest = None
        for root in [cache_dir] + list(map(Path, cache_sources)):
            for p in root.glob('*.npz'):
                with np.load(p, allow_pickle=False) as d:
                    if 'source_signature' not in d or 'source_sha256' not in d:
                        continue
                    old = json.loads(str(d['source_signature'].item()))
                    if {k:v for k,v in old.items() if k != 'mtime_ns'} != comparable:
                        continue
                    if source_digest is None:
                        source_digest = file_sha256(path)
                    if str(d['source_sha256'].item()) == source_digest:
                        candidates.append(p)
    for candidate in candidates:
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as data:
                raw, times, valid = data["raw"], data["times"], data["valid"]
                if raw.shape != (len(times), 9) or valid.shape != times.shape or len(times) < 2 or not np.all(np.diff(times) > 0):
                    raise ValueError(f"Invalid cache: {candidate}")
                return raw, times, valid
    if cache_only:
        raise FileNotFoundError(f"Missing matching cache for {path} at {sample_fps} FPS ({cached.name})")
    source_times = decoded_timestamps(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    effective_fps = (len(source_times)-1) / source_times[-1]
    if effective_fps + 1e-6 < sample_fps:
        cap.release()
        raise ValueError(f"Decoded source FPS {effective_fps} below target {sample_fps}: {path}")
    analyzer = FaceAnalyzer(model_path=str(model), use_image_mode=True, pose_convention="camera-v2")
    raw, times, valid = [], [], []
    frame_idx, next_time = 0, 0.0
    try:
        while cap.grab():
            if frame_idx >= len(source_times):
                raise ValueError(f"OpenCV decoded more frames than FFprobe: {path}")
            timestamp = source_times[frame_idx]
            frame_idx += 1
            if frame_idx > 1 and timestamp == source_times[frame_idx - 2]:
                continue  # only the explicitly verified duplicate can reach this point
            if timestamp + 1e-8 < next_time:
                continue
            ok, frame = cap.retrieve()
            # Skip elapsed sampling slots rather than duplicating frames after a gap.
            next_time = (np.floor((timestamp + 1e-8) * sample_fps) + 1) / sample_fps
            if not ok:
                raise ValueError(f"Frame decode failed in {path}")
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, round(h * 640 / w)))
            features = analyzer.analyze(frame)
            raw.append(analyzer.to_feature_vector(features)[:9] if features is not None else np.zeros(9))
            times.append(timestamp)
            valid.append(features is not None)
    finally:
        analyzer.close()
        cap.release()
    if frame_idx != len(source_times):
        raise ValueError(f"Decoder count mismatch: OpenCV={frame_idx}, FFprobe={len(source_times)}: {path}")
    if not raw:
        raise ValueError(f"Empty video: {path}")
    arrays = np.asarray(raw, np.float32), np.asarray(times, np.float64), np.asarray(valid, bool)
    temporary = cached.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, raw=arrays[0], times=arrays[1], valid=arrays[2],
                            timestamp_exception=np.asarray(exception or "none"),
                            source_signature=np.asarray(json.dumps(signature, sort_keys=True)),
                            source_sha256=np.asarray(file_sha256(path)))
    temporary.replace(cached)
    return arrays


def build(manifest, model, output, cache_dir, sample_fps=15, seq_len=60, stride=15, cache_sources=(), cache_only=False):
    if sample_fps <= 0 or seq_len < 2 or stride < 1:
        raise ValueError("Invalid sampling configuration")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; choose a new versioned dataset directory")
    with Path(manifest).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # Relative paths are resolved against the manifest location, not shell cwd.
    for r in rows:
        if not Path(r["path"]).is_absolute():
            r["path"] = str((Path(manifest).resolve().parent / r["path"]).resolve())
    classes = validate_rows(rows)
    from training.preflight20 import require_preflight
    require_preflight(manifest, output.parent / (output.name + "_preflight.json"), sample_fps)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for split in ("train", "val", "test"):
        X, y, subjects, videos, starts = [], [], [], [], []
        for i, row in enumerate(r for r in rows if r["split"] == split):
            raw, times, valid = extract_video(row["path"], model, sample_fps, cache_dir, cache_sources, cache_only)
            selected = (times >= float(row.get("start_s") or 0)) & (times < float(row.get("end_s") or "inf"))
            raw, times, valid = raw[selected], times[selected], valid[selected]
            accepted = rejected = 0
            for start in range(0, len(raw) - seq_len + 1, stride):
                sl = slice(start, start + seq_len)
                features = make_features(raw[sl], times[sl], valid[sl])
                if features is None:
                    rejected += 1
                    continue
                X.append(features); y.append(classes.index(row["label"]))
                subjects.append(row["subject"]); videos.append(row["path"]); starts.append(times[start])
                accepted += 1
            report.append(dict(**row, frames=len(raw), valid_fraction=float(valid.mean()) if len(valid) else 0,
                               accepted=accepted, rejected=rejected))
            print(f"{split} video {i+1}: {accepted} windows, {rejected} rejected", flush=True)
        if set(y) != set(range(len(classes))):
            raise ValueError(f"{split}: classes missing after quality filtering; inspect source videos")
        np.savez_compressed(output / f"{split}.npz", X=np.asarray(X, np.float32), y=np.asarray(y, np.int64),
                            subjects=np.asarray(subjects), videos=np.asarray(videos), starts=np.asarray(starts))
    preprocessing = dict(schema=SCHEMA, sample_fps=sample_fps, seq_len=seq_len, stride=stride,
                         min_valid=0.8, max_missing_seconds=0.2, feature_names=FEATURE_NAMES,
                         detector_mode="IMAGE", pose_convention="camera-v2", resize_width=640,
                         timestamp_source="ffprobe_best_effort_timestamp_time",
                         timestamp_version="ffprobe-best-effort-v1")
    code_root = Path(__file__).parents[1]
    metadata = dict(classes=classes, preprocessing=preprocessing, videos=report,
                    manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                    face_model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                    code_sha256={name: hashlib.sha256((code_root/name).read_bytes()).hexdigest()
                                 for name in ("core/features20.py", "core/face_analyzer.py", "training/prepare20.py", "training/video_timestamps20.py")},
                    extraction_versions={name: importlib.metadata.version(name) for name in ("mediapipe", "numpy", "scipy")})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"Dataset saved: {output}")


def cache_videos(manifest, model, cache_dir, sample_fps=15, shard_index=0, num_shards=1,
                 cache_sources=()):
    """Resume compatible caches; collect all runtime failures without silently dropping videos."""
    if num_shards < 1 or not 0 <= shard_index < num_shards or sample_fps <= 0:
        raise ValueError("Invalid shard or sample rate")
    from training.preflight20 import read_manifest, require_preflight
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    require_preflight(manifest, cache_dir / f"preflight_{shard_index}.json", sample_fps)
    rows = read_manifest(manifest)
    paths = sorted({r["path"] for r in rows})[shard_index::num_shards]
    records = []
    report_path = cache_dir / f"progress_{shard_index}.json"
    for i, path in enumerate(paths):
        print(f"Extract {i+1}/{len(paths)}: {path}", flush=True)
        try:
            raw, _, valid = extract_video(path, model, sample_fps, cache_dir, cache_sources)
            records.append(dict(path=path, status="ok", frames=len(raw), valid_fraction=float(valid.mean())))
        except Exception as exc:
            records.append(dict(path=path, status="error", error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED: {path}: {exc}", flush=True)
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
        temporary.replace(report_path)
    failures = sum(r["status"] == "error" for r in records)
    if failures:
        raise RuntimeError(f"{failures} videos failed; successful caches retained. Inspect {report_path}; do not train incomplete data.")
    print(f"Shard {shard_index}: {len(paths)} videos ready", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest-rldd")
    p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--include-low", action="store_true")
    p = sub.add_parser("preflight")
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p.add_argument("--sample-fps", type=float)
    p = sub.add_parser("build")
    p.add_argument("--manifest", required=True); p.add_argument("--model", required=True)
    p.add_argument("--cache-sources", nargs="*", default=[])
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--output", default="data/processed20")
    p.add_argument("--cache-dir", default="data/cache20")
    p.add_argument("--sample-fps", type=float, default=15)
    p.add_argument("--seq-len", type=int, default=60); p.add_argument("--stride", type=int, default=15)
    p = sub.add_parser("cache")
    p.add_argument("--cache-sources", nargs="*", default=[])
    p.add_argument("--manifest", required=True); p.add_argument("--model", required=True)
    p.add_argument("--cache-dir", required=True); p.add_argument("--sample-fps", type=float, default=15)
    p.add_argument("--shard-index", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "manifest-rldd":
        rldd_manifest(**args)
    elif command == "preflight":
        from training.preflight20 import preflight
        report = preflight(**args)
        if report["failed"]:
            raise SystemExit(2)
    elif command == "cache":
        cache_videos(**args)
    else:
        build(**args)


if __name__ == "__main__":
    main()
