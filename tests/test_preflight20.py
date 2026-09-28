import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from training.preflight20 import preflight
from training.prepare20 import build, cache_videos


class PreflightTest(unittest.TestCase):
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
