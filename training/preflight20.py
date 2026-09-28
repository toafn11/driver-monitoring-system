"""Cheap video inventory before MediaPipe extraction (not a full decode audit)."""
import csv
import json
import math
from pathlib import Path


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
        record = dict(path=path, issues=[], probes=[])
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
                for index in sorted({0, int(count // 2), max(0, int(count) - 2)}):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    record["probes"].append(dict(frame=index, decoded=bool(ok and frame is not None),
                                                  timestamp_s=timestamp if math.isfinite(timestamp) else None))
                    if not ok or frame is None:
                        record["issues"].append(f"Decode probe failed at frame {index}")
        except Exception as exc:
            record["issues"].append(f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()
        records.append(record)
    rates = [r["fps"] for r in records if r.get("fps") and r["fps"] > 0]
    report = dict(videos=records, total=len(records),
                  failed=sum(bool(r["issues"]) for r in records), target_fps=sample_fps,
                  suggested_fps=min(10, math.floor(min(rates))) if rates and min(rates) >= 1 else None,
                  limitation="First/middle/end decode probes only. Timestamps are diagnostic; extraction assumes CFR frame_index/FPS. VFR is not certified by this audit.")
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
