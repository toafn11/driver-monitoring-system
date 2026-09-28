import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import csv
import cv2
import numpy as np
from core.face_analyzer import FaceFeatures
from training.prepare20 import build, extract_video
from training.train20 import load_data


class DetectorFixture:
    """Known geometry for testing decode/cache/split plumbing only."""
    calls = 0
    def __init__(self, **kwargs): pass
    def analyze(self, frame):
        type(self).calls += 1
        return FaceFeatures(ear_left=0.3, ear_right=0.3, ear_avg=0.3, mar=0.1)
    def to_feature_vector(self, f):
        return np.array([f.ear_left, f.ear_right, f.ear_avg, f.mar] + [0.] * 8, np.float32)
    def close(self): pass


def video(path, frames=120):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 64))
    if not writer.isOpened(): raise RuntimeError('MJPG encoder unavailable')
    for _ in range(frames): writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()


class PreparationTest(unittest.TestCase):
    def test_decode_sampling_cache_and_complete_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root/'dummy.task'; model.write_bytes(b'fixture')
            rows = []
            for split in ('train', 'val', 'test'):
                for label in ('ALERT', 'DROWSY'):
                    path = root/f'{split}_{label}.avi'; video(path)
                    rows.append(dict(path=str(path), subject=split, split=split, label=label,
                                     start_s=0, end_s=''))
            manifest = root/'manifest.csv'
            with manifest.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            with patch('core.face_analyzer.FaceAnalyzer', DetectorFixture):
                DetectorFixture.calls = 0
                raw, times, valid = extract_video(rows[0]['path'], model, 15, root/'cache')
                self.assertEqual(len(raw), 60)
                np.testing.assert_allclose(np.diff(times), 1/15)
                self.assertTrue(valid.all())
                calls = DetectorFixture.calls
                extract_video(rows[0]['path'], model, 15, root/'cache')
                self.assertEqual(DetectorFixture.calls, calls)
                extract_video(rows[0]['path'], model, 15, root/'new_cache',
                              cache_sources=[root/'cache'], cache_only=True)
                self.assertEqual(DetectorFixture.calls, calls)
                with self.assertRaises(FileNotFoundError):
                    extract_video(rows[0]['path'], model, 10, root/'new_cache',
                                  cache_sources=[root/'cache'], cache_only=True)
                build(manifest, model, root/'data', root/'cache')
            splits, metadata = load_data(root/'data')
            self.assertEqual(splits['test']['X'].shape, (2, 60, 20))
            self.assertEqual(metadata['preprocessing']['pose_convention'], 'camera-v2')


if __name__ == '__main__': unittest.main()
