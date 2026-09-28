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


def extract_video(path, model, sample_fps, cache_dir):
    # Imported lazily so manifest checks work without MediaPipe/OpenCV.
    import cv2
    from core.face_analyzer import FaceAnalyzer
    model = Path(model)
    stat = Path(path).stat()
    signature = dict(path=str(Path(path).resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, sample_fps=sample_fps, pose_convention="camera-v2",
                     model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                     extractor_sha256=hashlib.sha256(
                         (Path(__file__).parents[1] / "core/face_analyzer.py").read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key}.npz"
    if cached.exists():
        with np.load(cached, allow_pickle=False) as data:
            return data["raw"], data["times"], data["valid"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < sample_fps:
        cap.release()
        raise ValueError(f"Source FPS {fps} below target {sample_fps}: {path}; do not duplicate frames")
    analyzer = FaceAnalyzer(model_path=str(model), use_image_mode=True, pose_convention="camera-v2")
    raw, times, valid = [], [], []
    frame_idx, next_time = 0, 0.0
    try:
        while cap.grab():
            timestamp = frame_idx / fps
            frame_idx += 1
            if timestamp + 1e-8 < next_time:
                continue
            ok, frame = cap.retrieve()
            next_time += 1.0 / sample_fps
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
    if not raw:
        raise ValueError(f"Empty video: {path}")
    arrays = np.asarray(raw, np.float32), np.asarray(times, np.float64), np.asarray(valid, bool)
    np.savez_compressed(cached, raw=arrays[0], times=arrays[1], valid=arrays[2])
    return arrays


def build(manifest, model, output, cache_dir, sample_fps=15, seq_len=60, stride=15):
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
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for split in ("train", "val", "test"):
        X, y, subjects, videos, starts = [], [], [], [], []
        for i, row in enumerate(r for r in rows if r["split"] == split):
            raw, times, valid = extract_video(row["path"], model, sample_fps, cache_dir)
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
                         timestamp_source="frame_index/source_fps")
    code_root = Path(__file__).parents[1]
    metadata = dict(classes=classes, preprocessing=preprocessing, videos=report,
                    manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                    face_model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                    code_sha256={name: hashlib.sha256((code_root/name).read_bytes()).hexdigest()
                                 for name in ("core/features20.py", "core/face_analyzer.py", "training/prepare20.py")},
                    extraction_versions={name: importlib.metadata.version(name) for name in ("mediapipe", "numpy", "scipy")})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"Dataset saved: {output}")


def cache_videos(manifest, model, cache_dir, sample_fps=15, shard_index=0, num_shards=1):
    """Optional extraction-only stage; resume using the same cache directory."""
    if num_shards < 1 or not 0 <= shard_index < num_shards or sample_fps <= 0:
        raise ValueError("Invalid shard or sample rate")
    with Path(manifest).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = sorted({str((Path(manifest).resolve().parent / r["path"]).resolve()) for r in rows})
    selected = paths[shard_index::num_shards]
    for i, path in enumerate(selected):
        raw, _, valid = extract_video(path, model, sample_fps, cache_dir)
        print(f"Cached {i+1}/{len(selected)}: {Path(path).name}, {len(raw)} frames, face={valid.mean():.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest-rldd")
    p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--include-low", action="store_true")
    p = sub.add_parser("build")
    p.add_argument("--manifest", required=True); p.add_argument("--model", required=True)
    p.add_argument("--output", default="data/processed20")
    p.add_argument("--cache-dir", default="data/cache20")
    p.add_argument("--sample-fps", type=float, default=15)
    p.add_argument("--seq-len", type=int, default=60); p.add_argument("--stride", type=int, default=15)
    p = sub.add_parser("cache")
    p.add_argument("--manifest", required=True); p.add_argument("--model", required=True)
    p.add_argument("--cache-dir", required=True); p.add_argument("--sample-fps", type=float, default=15)
    p.add_argument("--shard-index", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "manifest-rldd":
        rldd_manifest(**args)
    elif command == "cache":
        cache_videos(**args)
    else:
        build(**args)


if __name__ == "__main__":
    main()
