"""Validation-only baseline/regularization comparison; never loads test.npz."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models.temporal20 import Temporal20, save_checkpoint, load_checkpoint
from training.train20 import fit_normalizer, metrics, predict_batches


def run_experiments(data, output, seeds=(42, 43, 44), epochs=40, patience=8):
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError('Use a new experiment directory')
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU before running these experiments')
    metadata = json.loads((data / 'metadata.json').read_text())
    splits = {}
    for split in ('train', 'val'):
        with np.load(data / f'{split}.npz', allow_pickle=False) as a:
            splits[split] = {k: a[k] for k in ('X', 'y', 'subjects', 'videos')}
        d = splits[split]
        assert d['X'].shape == (len(d['y']), 40, 20)
        assert np.isfinite(d['X']).all()
        assert set(d['y'].tolist()) == {0, 1}
    tr, va = splits['train'], splits['val']
    for key in ('subjects', 'videos'):
        assert not set(tr[key]) & set(va[key]), f'Overlapping {key}'
    fingerprint = hashlib.sha256((data / 'metadata.json').read_bytes()).hexdigest()
    configs = {
        'baseline': dict(hidden=64, dropout=0.2, lr=1e-3, weight_decay=1e-3),
        'regularized': dict(hidden=32, dropout=0.4, lr=3e-4, weight_decay=1e-2),
    }
    output.mkdir(parents=True)
    results = []
    for name, config in configs.items():
        for dim in (12, 20):
            for seed in seeds:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                folder = output / f'{name}_gru{dim}_seed{seed}'
                folder.mkdir()
                model = Temporal20(metadata['classes'], hidden=config['hidden'],
                                   dropout=config['dropout'], feature_dim=dim)
                fit_normalizer(model, tr['X']); model.to('cuda')
                counts = np.bincount(tr['y'])
                sampler = WeightedRandomSampler(
                    torch.as_tensor(1 / counts[tr['y']], dtype=torch.double),
                    len(tr['y']), replacement=True,
                    generator=torch.Generator().manual_seed(seed))
                loader = DataLoader(TensorDataset(torch.from_numpy(tr['X']),
                    torch.from_numpy(tr['y'])), batch_size=128, sampler=sampler,
                    num_workers=0, pin_memory=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', patience=3, factor=0.5)
                loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
                history, best, stale = [], -1., 0
                (folder / 'run.json').write_text(json.dumps(dict(
                    config=config, seed=seed, feature_dim=dim, epochs=epochs,
                    patience=patience, data_metadata_sha256=fingerprint), indent=2))
                for epoch in range(1, epochs + 1):
                    model.train(); total = 0.
                    for x, y in loader:
                        x, y = x.cuda(), y.cuda()
                        optimizer.zero_grad(set_to_none=True)
                        logits, _ = model(x); loss = loss_fn(logits, y)
                        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step(); total += loss.item() * len(y)
                    probs = predict_batches(model, va['X'], 'cuda', 128)
                    report = metrics(va['y'], probs.argmax(1), metadata['classes'])
                    score = report['macro_f1']; scheduler.step(score)
                    history.append(dict(epoch=epoch, train_loss=total/len(tr['y']), validation=report))
                    (folder / 'history.json').write_text(json.dumps(history, indent=2))
                    print(folder.name, epoch, round(score, 4), flush=True)
                    if score > best + 1e-6:
                        best, stale = score, 0
                        save_checkpoint(folder / 'best.pt', model, metadata['preprocessing'],
                            epoch=epoch, seed=seed, validation=report, data_metadata_sha256=fingerprint)
                    else:
                        stale += 1
                        if stale >= patience: break
                model, saved = load_checkpoint(folder / 'best.pt', 'cuda')
                probs = predict_batches(model, va['X'], 'cuda', 128)
                pred = probs.argmax(1)
                report = metrics(va['y'], pred, metadata['classes'])
                report['by_subject'] = {str(s): metrics(va['y'][va['subjects']==s],
                    pred[va['subjects']==s], metadata['classes']) for s in sorted(set(va['subjects']))}
                (folder / 'validation.json').write_text(json.dumps(report, indent=2))
                np.savez_compressed(folder / 'validation_predictions.npz', probabilities=probs,
                                    y=va['y'], subjects=va['subjects'], videos=va['videos'])
                results.append(dict(config=name, feature_dim=dim, seed=seed,
                    best_epoch=saved['epoch'], macro_f1=report['macro_f1'],
                    drowsy_recall=report['per_class']['DROWSY']['recall']))
                (output / 'results.json').write_text(json.dumps(results, indent=2))
    summary = []
    for name in configs:
        for dim in (12, 20):
            rows = [r for r in results if r['config']==name and r['feature_dim']==dim]
            summary.append(dict(config=name, feature_dim=dim,
                mean_macro_f1=float(np.mean([r['macro_f1'] for r in rows])),
                std_macro_f1=float(np.std([r['macro_f1'] for r in rows])),
                mean_drowsy_recall=float(np.mean([r['drowsy_recall'] for r in rows]))))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'complete.json').write_text(json.dumps(dict(runs=len(results), test_evaluated=False)))
    print(json.dumps(summary, indent=2))
