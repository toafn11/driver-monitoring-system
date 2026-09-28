import unittest
import tempfile
from pathlib import Path
import numpy as np
from core.features20 import FeatureWindow, make_features
from training.prepare20 import rldd_manifest, validate_rows


def example(n=60):
    raw = np.zeros((n, 9), np.float32)
    raw[:, :3] = 0.3; raw[:, 3] = 0.1
    return raw, np.arange(n) / 15, np.ones(n, dtype=bool)


class FeaturesTest(unittest.TestCase):
    def test_missing_is_not_eye_closure(self):
        raw, times, valid = example()
        valid[20:22] = False; raw[20:22] = 0
        x = make_features(raw, times, valid)
        self.assertIsNotNone(x)
        self.assertTrue((x[20:22] == 0).all())
        self.assertTrue((x[:, 9:12] == 0).all())
        self.assertTrue((x[:, 16] == 0).all())

    def test_long_missing_gap_is_rejected(self):
        raw, times, valid = example()
        valid[20:24] = False
        self.assertIsNone(make_features(raw, times, valid))

    def test_actual_seconds_and_no_fake_first_blink(self):
        raw, times, valid = example()
        raw[:, :3] = 0.1
        x = make_features(raw, times, valid)
        self.assertAlmostEqual(float(x[-1, 10]), 4 / 5, places=5)
        self.assertEqual(x[-1, 16], 0)
        self.assertEqual(x[-1, 9], 1)

    def test_velocity_uses_seconds(self):
        for fps in (15, 30):
            raw, _, valid = example()
            times = np.arange(60) / fps; raw[:, 4] = times * 0.1
            x = make_features(raw, times, valid)
            np.testing.assert_allclose(x[1:, 12], 0.1, atol=1e-6)

    def test_live_and_offline_match(self):
        raw, times, valid = example()
        raw[20:25, :3] = 0.1
        window = FeatureWindow(60)
        for r, t in zip(raw, times):
            live = window.push(r, t)
        np.testing.assert_array_equal(live, make_features(raw, times, valid))

    def test_bad_timestamps(self):
        raw, times, valid = example(); times[10] = times[9]
        with self.assertRaises(ValueError): make_features(raw, times, valid)


class ManifestTest(unittest.TestCase):
    def test_same_person_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'v.mp4'; p.touch()
            rows = [dict(path=str(p), subject='same', split=s, label='ALERT') for s in ('train', 'test')]
            with self.assertRaisesRegex(ValueError, 'leakage'): validate_rows(rows)

    def test_original_rldd_fold_and_split_clips(self):
        import csv
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for fold, subject in ((1, '01'), (4, '37'), (5, '49')):
                folder = root / f'Fold{fold}_part1' / subject; folder.mkdir(parents=True)
                for name in ('0.mov', '10_1.mp4', '10_2.mp4', '5.MOV'):
                    (folder/name).touch()
            manifest = root / 'manifest.csv'; rldd_manifest(root, manifest)
            with manifest.open() as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            self.assertEqual({r['label'] for r in rows}, {'ALERT', 'DROWSY'})
            self.assertEqual({r['subject'] for r in rows if r['split'] == 'test'}, {'rldd:49'})

    def test_duplicate_intervals_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'v.mp4'; p.touch()
            row = dict(path=str(p), subject='s', split='train', label='ALERT')
            with self.assertRaisesRegex(ValueError, 'Overlapping'): validate_rows([row, row])


if __name__ == '__main__': unittest.main()
