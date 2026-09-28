import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
from training.preflight20 import preflight, sequential_probe
from training.prepare20 import build, cache_videos


class PreflightTest(unittest.TestCase):
    def test_metadata_shortage_requires_independent_count_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else (100 if prop == cv2.CAP_PROP_FRAME_COUNT else 0)
            cap.read.return_value = (False, None)
            with patch('training.preflight20.read_manifest', return_value=[dict(path='fixture.mov')]), \
                 patch('cv2.VideoCapture', return_value=cap), \
                 patch('training.preflight20.sequential_probe', return_value=dict(decoded_frames=90, passed=False)), \
                 patch('training.video_timestamps20.decoded_timestamps', return_value=np.arange(90)/30):
                report = preflight('manifest', Path(d)/'audit.json', 10)
                self.assertEqual(report['failed'], 0)
                self.assertTrue(report['videos'][0]['warnings'])
            with patch('training.preflight20.read_manifest', return_value=[dict(path='fixture.mov')]), \
                 patch('cv2.VideoCapture', return_value=cap), \
                 patch('training.preflight20.sequential_probe', return_value=dict(decoded_frames=90, passed=False)), \
                 patch('training.video_timestamps20.decoded_timestamps', return_value=np.arange(91)/30):
                self.assertEqual(preflight('manifest', Path(d)/'bad.json', 10)['failed'], 1)

    def test_failed_seek_recovers_only_with_complete_sequential_decode(self):
        real_capture = cv2.VideoCapture
        class BrokenSeek:
            def __init__(self, path):
                self.cap = real_capture(path)
                self.sought = False
            def isOpened(self): return self.cap.isOpened()
            def get(self, prop): return self.cap.get(prop)
            def set(self, prop, value):
                self.sought = True
                return False
            def read(self):
                if self.sought: return False, None
                return self.cap.read()
            def release(self): self.cap.release()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest = self.fixture(root)
            with patch('cv2.VideoCapture', BrokenSeek):
                report = preflight(manifest, root/'audit.json', 10)
                self.assertEqual(report['failed'], 0)
                self.assertTrue(all(r['warnings'] for r in report['videos']))
                self.assertTrue(all(r['sequential_audit']['decoded_frames']==48 for r in report['videos']))
                # Recovering decode must not override a separate FPS failure.
                self.assertEqual(preflight(manifest, root/'audit15.json', 15)['failed'], 6)

    def test_early_eof_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            result = sequential_probe(root/'train_ALERT.avi', 100)
            self.assertFalse(result['passed'])
            self.assertEqual(result['decoded_frames'], 48)

    def fixture(self, root):
        rows = []
        for split in ('train', 'val', 'test'):
            for label in ('ALERT', 'DROWSY'):
                path = root / f'{split}_{label}.avi'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 12, (64,64))
                self.assertTrue(writer.isOpened())
                for _ in range(48): writer.write(np.zeros((64,64,3), np.uint8))
                writer.release()
                rows.append(dict(path=path.name, subject=split, split=split, label=label))
        manifest = root/'manifest.csv'
        with manifest.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        return manifest

    def test_low_fps_fails_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=self.fixture(root)
            report=preflight(manifest, root/'audit.json',15)
            self.assertEqual(report['failed'],6)
            self.assertEqual(report['suggested_fps'],10)
            self.assertEqual(preflight(manifest,root/'audit10.json',10)['failed'],0)
            with patch('training.prepare20.extract_video') as extract:
                with self.assertRaisesRegex(ValueError,'Preflight failed'):
                    build(manifest,root/'model.task',root/'data',root/'cache')
                extract.assert_not_called()

    def test_runtime_failure_keeps_progress_and_continues(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=self.fixture(root)
            good=(np.zeros((40,9)),np.arange(40)/10,np.ones(40,dtype=bool))
            with patch('training.prepare20.extract_video',side_effect=[ValueError('bad decode')]+[good]*5) as extract:
                with self.assertRaisesRegex(RuntimeError,'1 videos failed'):
                    cache_videos(manifest,root/'model.task',root/'cache',sample_fps=10)
                self.assertEqual(extract.call_count,6)
            records=json.loads((root/'cache/progress_0.json').read_text())
            self.assertEqual(sum(r['status']=='ok' for r in records),5)


if __name__ == '__main__': unittest.main()
