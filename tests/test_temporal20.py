import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from core.features20 import SCHEMA, FEATURE_NAMES, make_features
from models.temporal20 import Temporal20, save_checkpoint, load_checkpoint
from training.train20 import train, evaluate, fit_normalizer, load_data


class ModelTest(unittest.TestCase):
    def test_checkpoint_round_trip_and_backward(self):
        torch.set_num_threads(1)
        for cell in ('gru', 'lstm'):
            for dim in (12, 20):
                model = Temporal20(['ALERT', 'DROWSY'], cell=cell, feature_dim=dim)
                x = torch.rand(3, 60, 20); x[..., 19] = 1; x[:, 20:22, 19] = 0
                logits, attn = model(x)
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
                self.assertEqual(tuple(logits.shape), (3, 2))
                self.assertTrue(torch.all(attn[:, 20:22] == 0))
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / 'model.pt'; model.eval()
                    save_checkpoint(path, model, dict(schema=SCHEMA))
                    loaded, _ = load_checkpoint(path)
                    torch.testing.assert_close(model(x)[0], loaded(x)[0])
                    self.assertTrue(all(p[1] == 'UNCERTAIN' for p in loaded.predict(x, 1.1)))

    def test_normalization_ignores_missing(self):
        model = Temporal20(['ALERT', 'DROWSY'])
        x = np.ones((2, 60, 20), np.float32) * 0.3; x[..., 19] = 1
        x[:, :2] = 0
        fit_normalizer(model, x)
        np.testing.assert_allclose(model.mean.numpy()[:19], 0.3, atol=1e-6)
        self.assertEqual(model.mean[19], 0)

    def test_tiny_training_and_evaluation(self):
        # Integration smoke test, NOT scientific performance evidence.
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = root / 'data'; data.mkdir()
            meta = dict(classes=['ALERT', 'DROWSY'], manifest_sha256='test-fixture',
                        preprocessing=dict(schema=SCHEMA, seq_len=60, sample_fps=15,
                                           feature_names=FEATURE_NAMES))
            (data/'metadata.json').write_text(json.dumps(meta))
            for split in ('train', 'val', 'test'):
                sequences = []
                for y in [0, 1] * 4:
                    raw = np.zeros((60, 9), np.float32); raw[:, :3] = 0.3 if y == 0 else 0.1
                    raw[:, 3] = 0.1
                    sequences.append(make_features(raw, np.arange(60)/15, np.ones(60, bool)))
                np.savez(data/f'{split}.npz', X=np.asarray(sequences), y=np.asarray([0, 1]*4),
                         subjects=np.asarray([split]*8), videos=np.asarray([split+'.mp4']*8), starts=np.arange(8))
            run = root/'run'
            train(data, run, epochs=2, hidden=8, batch_size=4, device='cpu')
            evaluate(data, run/'best.pt', root/'evaluation', device='cpu')
            self.assertTrue((root/'evaluation/metrics.json').exists())
            self.assertTrue((root/'evaluation/predictions.npz').exists())
            # Tampering with split identity must stop training, not silently leak.
            with np.load(data/'test.npz') as z: arrays = {k:z[k] for k in z.files}
            arrays['subjects'] = np.full(8, 'train'); np.savez(data/'test.npz', **arrays)
            with self.assertRaisesRegex(ValueError, 'overlap'): load_data(data)


if __name__ == '__main__': unittest.main()
