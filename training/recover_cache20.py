"""Explicit legacy-cache migration: progress + full PTS + sampled geometry checks.

Never infer video identity from filenames, frame count or archive ordering alone.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from training.prepare20 import cache_filename, cache_signature, extract_video, file_sha256
from training.preflight20 import read_manifest
from training.video_timestamps20 import decoded_timestamps


def sampled_indices(times, fps):
    selected, next_time = [], 0.0
    for i, timestamp in enumerate(times):
        if i and timestamp == times[i-1]:
            continue
        if timestamp + 1e-8 < next_time:
            continue
        selected.append(i)
        next_time = (np.floor((timestamp+1e-8)*fps)+1)/fps
    return np.asarray(selected, dtype=np.int64)


def verify_geometry(path, model, raw, valid, indices):
    import cv2
    from core.face_analyzer import FaceAnalyzer
    usable = np.flatnonzero(valid)
    if len(usable) < 8:
        raise ValueError(f"Not enough valid geometry for legacy identity verification: {path}")
    anchors = usable[np.linspace(0, len(usable)-1, min(32, len(usable))).astype(int)]
    targets = {int(indices[i]): int(i) for i in anchors}
    cap = cv2.VideoCapture(str(path))
    analyzer = FaceAnalyzer(model_path=str(model), use_image_mode=True, pose_convention="camera-v2")
    frame_index, checked = 0, 0
    try:
        while frame_index <= max(targets) and cap.grab():
            if frame_index in targets:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"Cannot decode verification frame {frame_index}: {path}")
                h,w=frame.shape[:2]
                if w>640: frame=cv2.resize(frame,(640,round(h*640/w)))
                found=analyzer.analyze(frame)
                if found is None or not np.allclose(analyzer.to_feature_vector(found)[:9],
                                                    raw[targets[frame_index]], rtol=0, atol=2e-4):
                    raise ValueError(f"Cached geometry does not match source at frame {frame_index}: {path}")
                checked += 1
            frame_index += 1
    finally:
        cap.release(); analyzer.close()
    if checked != len(targets):
        raise ValueError(f"Incomplete geometry verification: {path}")
    return checked


def recover_shard0(manifest, model, cache_dir, cache_sources, sample_fps=10):
    """Only recover the 30 successful videos from the user's failed shard 0."""
    paths=sorted({r['path'] for r in read_manifest(manifest)})[0::4]
    missing='/Fold1_part2/Fold1_part2/11/0.mp4'
    expected=[p for p in paths if not p.endswith(missing)]
    if len(paths)!=31 or len(expected)!=30 or sample_fps!=10:
        raise ValueError('This recovery is restricted to the original 31-video shard 0 at 10 FPS')
    sources=list(map(Path,cache_sources)); cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    progress=[p/'progress_0.json' for p in sources if (p/'progress_0.json').is_file()]
    records=[r for p in progress for r in json.loads(p.read_text()) if r.get('status')=='ok']
    if not records:
        raise ValueError('Attach original progress_0.json alongside legacy NPZ caches')
    candidates=[]
    for source in sources:
        for p in sorted(source.glob('*.npz')):
            with np.load(p,allow_pickle=False) as d:
                raw,times,valid=d['raw'],d['times'],d['valid']
                if raw.shape!=(len(times),9) or valid.shape!=times.shape or not np.isfinite(raw).all() or not np.isfinite(times).all() or not np.all(np.diff(times)>0):
                    raise ValueError(f'Invalid cache arrays: {p}')
                candidates.append((p,len(times),float(valid.mean()),times.copy()))
    report=[]
    for number,path in enumerate(expected,1):
        print(f'Legacy recovery {number}/30: {path}',flush=True)
        try:
            extract_video(path,model,sample_fps,cache_dir,cache_sources=sources,cache_only=True)
            report.append(dict(path=path,status='existing-key'))
            continue
        except FileNotFoundError:
            pass
        matches=[r for r in records if r['path']==path]
        if not matches or len({(r['frames'],r['valid_fraction']) for r in matches})!=1:
            raise ValueError(f'Missing or conflicting legacy progress: {path}')
        r=matches[0]
        possible=[c for c in candidates if c[1]==r['frames'] and abs(c[2]-r['valid_fraction'])<1e-12]
        pts=decoded_timestamps(path); indices=sampled_indices(pts,sample_fps); selected=pts[indices]
        possible=[c for c in possible if c[3].shape==selected.shape and np.allclose(c[3],selected,rtol=0,atol=1e-9)]
        # Duplicate copies are okay only if their bytes are identical.
        distinct={hashlib.sha256(c[0].read_bytes()).hexdigest():c[0] for c in possible}
        if len(distinct)!=1:
            raise ValueError(f'Cannot uniquely identify legacy cache for {path}: {len(distinct)} candidates')
        checksum,old=next(iter(distinct.items()))
        with np.load(old,allow_pickle=False) as d:
            raw,times,valid=d['raw'],d['times'],d['valid']
            anchors=verify_geometry(path,model,raw,valid,indices)
            signature=cache_signature(path,model,sample_fps)
            target=cache_dir/cache_filename(path,model,sample_fps)
            temporary=target.with_suffix('.tmp')
            provenance=dict(legacy_file=str(old),legacy_sha256=checksum,geometry_anchors=anchors,
                            timestamp_comparison='all sampled timestamps; atol=1e-9',geometry_atol=2e-4)
            with temporary.open('wb') as f:
                np.savez_compressed(f,raw=raw,times=times,valid=valid,
                                    source_signature=np.asarray(json.dumps(signature,sort_keys=True)),
                                    source_sha256=np.asarray(file_sha256(path)),
                                    recovery_provenance=np.asarray(json.dumps(provenance,sort_keys=True)))
            temporary.replace(target)
        report.append(dict(path=path,status='verified-migration',target=str(target),**provenance))
        (cache_dir/'recovery_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'Restored {number}/30 using full timestamp match and {anchors} geometry checks',flush=True)
    (cache_dir/'recovery_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('LEGACY RECOVERY COMPLETE: all 30 existing video caches are usable',flush=True)
