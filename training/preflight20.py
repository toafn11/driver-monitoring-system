"""Cheap video inventory before MediaPipe extraction (not a full decode audit)."""
import csv
import json
import math
from pathlib import Path


def sequential_probe(path, expected_frames):
    """Reopen and decode without seeking; do not silently accept an early EOF."""
    import cv2
    cap = cv2.VideoCapture(path)
    decoded = 0
    try:
        if not cap.isOpened():
            return dict(decoded_frames=0, passed=False, reason="Cannot reopen video")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            decoded += 1
            if decoded % 5000 == 0:
                print(f"Sequential audit: {Path(path).name}: {decoded} frames", flush=True)
    finally:
        cap.release()
    # Only allow one frame of container count rounding, not a percentage of a video.
    passed = decoded >= max(2, math.ceil(expected_frames) - 1)
    return dict(decoded_frames=decoded, expected_frames=expected_frames, passed=passed,
                reason="Sequential decode reached advertised length" if passed else
                "Sequential decode ended early; investigate file/decoder/frame-count metadata")


def read_manifest(manifest):
    from training.prepare20 import validate_rows
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["path"] = str((manifest.parent / row["path"]).resolve())
    validate_rows(rows)
    return rows


def preflight(manifest, output, sample_fps=None):
    import cv2
    if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
        raise ValueError("Invalid sample_fps")
    rows = read_manifest(manifest)
    records = []
    for path in sorted({r["path"] for r in rows}):
        cap = cv2.VideoCapture(path)
        record = dict(path=path, issues=[], warnings=[], probes=[])
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            record.update(fps=fps if math.isfinite(fps) else None,
                          frames=count if math.isfinite(count) else None)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 2:
                record["issues"].append("Invalid video metadata/open failure")
            else:
                record["duration_s"] = count / fps
                if sample_fps is not None and fps < sample_fps:
                    record["issues"].append(f"FPS {fps:.6f} below target {sample_fps}")
                failed_probes = []
                for index in sorted({0, int(count // 2), max(0, int(count) - 2)}):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    record["probes"].append(dict(frame=index, decoded=bool(ok and frame is not None),
                                                  timestamp_s=timestamp if math.isfinite(timestamp) else None))
                    if not ok or frame is None:
                        failed_probes.append(index)
                if failed_probes:
                    cap.release()
                    print(f"Seek probe failed: {path}; checking sequential decode", flush=True)
                    result = sequential_probe(path, count)
                    record["sequential_audit"] = result
                    if result["passed"]:
                        record["warnings"].append(f"Seek probes failed at {failed_probes}; sequential decode passed")
                    else:
                        from training.video_timestamps20 import decoded_timestamps
                        times = decoded_timestamps(path)
                        record["ffprobe_audit"] = dict(decoded_frames=len(times),
                                                      last_timestamp_s=float(times[-1]),
                                                      strictly_increasing=True)
                        if len(times) == result["decoded_frames"]:
                            record["warnings"].append(
                                f"Metadata count {count} differs from decoded count {len(times)}; "
                                "OpenCV and FFprobe agree; FFprobe reports no decode errors and timestamps increase")
                        else:
                            record["issues"].append(
                                f"Decoder count disagreement: OpenCV={result['decoded_frames']}, FFprobe={len(times)}, metadata={count}")
        except Exception as exc:
            record["issues"].append(f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()
        records.append(record)
    rates = [r["fps"] for r in records if r.get("fps") and r["fps"] > 0]
    report = dict(videos=records, total=len(records),
                  failed=sum(bool(r["issues"]) for r in records), target_fps=sample_fps,
                  suggested_fps=min(10, math.floor(min(rates))) if rates and min(rates) >= 1 else None,
                  limitation="Quick seek probes, then sequential checks for failures. Count shortages require agreement with error-free FFprobe decoding and increasing timestamps. Decoder-concealed damage is not excluded. Extraction independently validates every frame timestamp using FFprobe.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Preflight: {report['total']} videos, {report['failed']} failed; suggested FPS={report['suggested_fps']}; {output}", flush=True)
    return report


def require_preflight(manifest, output, sample_fps):
    report = preflight(manifest, output, sample_fps)
    if report["failed"]:
        raise ValueError(f"Preflight failed for {report['failed']} videos. Inspect {output} before extraction.")
    return report
