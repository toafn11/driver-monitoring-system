"""Reproducible training for geometric20-v2; test is a separate command."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from core.features20 import SCHEMA
from models.temporal20 import Temporal20, load_checkpoint, save_checkpoint


def load_data(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["preprocessing"]["schema"] != SCHEMA:
        raise ValueError("Wrong feature schema; rebuild with training.prepare20")
    splits = {}
    for split in ("train", "val", "test"):
        with np.load(directory / f"{split}.npz", allow_pickle=False) as z:
            splits[split] = {k: z[k] for k in ("X", "y", "subjects", "videos", "starts")}
        d = splits[split]
        if d["X"].shape != (len(d["y"]), metadata["preprocessing"]["seq_len"], 20):
            raise ValueError(f"Invalid {split} shape")
        if not np.isfinite(d["X"]).all() or not ((d["X"][..., 19] > 0.5).any(1)).all():
            raise ValueError(f"Non-finite or empty-face windows: {split}")
        if set(d["y"].tolist()) != set(range(len(metadata["classes"]))):
            raise ValueError(f"Missing/invalid classes in {split}")
        if any(len(d[k]) != len(d["y"]) for k in ("subjects", "videos", "starts")):
            raise ValueError(f"Metadata lengths differ in {split}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for k in ("subjects", "videos"):
            if set(splits[a][k]) & set(splits[b][k]):
                raise ValueError(f"{k} overlap: {a}/{b}")
    return splits, metadata


def metrics(y, predicted, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    tp = np.diag(matrix)
    recall = tp / np.maximum(matrix.sum(1), 1)
    precision = tp / np.maximum(matrix.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(accuracy=float(tp.sum() / max(matrix.sum(), 1)), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), confusion_matrix=matrix.tolist(),
                per_class={name: dict(precision=float(precision[i]), recall=float(recall[i]),
                                      f1=float(f1[i]), support=int(matrix[i].sum()))
                           for i, name in enumerate(classes)})


def fit_normalizer(model, X):
    # Streaming moments avoid a second full dataset-sized copy.
    dim = model.config["feature_dim"]
    total = np.zeros(dim, np.float64); squares = total.copy(); count = 0
    for start in range(0, len(X), 256):
        batch = X[start:start+256]
        valid = batch[..., 19] > 0.5
        values = batch[..., :dim][valid].astype(np.float64)
        total += values.sum(0); squares += (values ** 2).sum(0); count += len(values)
    mean = total / count
    scale = np.maximum(np.sqrt(np.maximum(squares / count - mean ** 2, 0)), 0.01)
    if dim == 20:
        mean[19], scale[19] = 0, 1  # keep the detection mask interpretable
    model.mean.copy_(torch.tensor(mean, dtype=torch.float32))
    model.scale.copy_(torch.tensor(scale, dtype=torch.float32))


def predict_batches(model, X, device, batch_size):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            logits, _ = model(torch.from_numpy(X[start:start+batch_size]).to(device))
            probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def train(data, output, cell="gru", feature_dim=20, hidden=64, epochs=40,
          patience=8, batch_size=128, seed=42, device=None):
    if min(epochs, patience, batch_size, hidden) < 1:
        raise ValueError("Training limits must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory for each experiment")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits, metadata = load_data(data)
    # Test metadata is checked for leakage, but no test predictions select models.
    tr, va = splits["train"], splits["val"]
    model = Temporal20(metadata["classes"], hidden=hidden, cell=cell, feature_dim=feature_dim)
    fit_normalizer(model, tr["X"])
    model.to(device)
    counts = np.bincount(tr["y"], minlength=len(metadata["classes"]))
    generator = torch.Generator().manual_seed(seed)
    # One balancing method only; do not also apply inverse-frequency loss.
    sampler = WeightedRandomSampler(torch.as_tensor(1 / counts[tr["y"]], dtype=torch.double),
                                    len(tr["y"]), replacement=True, generator=generator)
    loader = DataLoader(TensorDataset(torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"])),
                        batch_size=batch_size, sampler=sampler, num_workers=0,
                        pin_memory=str(device).startswith("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    output.mkdir(parents=True, exist_ok=True)
    run = dict(seed=seed, device=str(device), model=model.config, epochs=epochs,
               batch_size=batch_size, patience=patience, parameters=sum(p.numel() for p in model.parameters()),
               torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
               data_metadata_sha256=hashlib.sha256((Path(data)/"metadata.json").read_bytes()).hexdigest(),
               manifest_sha256=metadata["manifest_sha256"], preprocessing=metadata["preprocessing"],
               subjects={s: sorted(set(d["subjects"].tolist())) for s, d in splits.items()})
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    best, stale, history = -1.0, 0, []
    print(f"{cell.upper()} {feature_dim} inputs, {run['parameters']:,} parameters on {device}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(X)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        probs = predict_batches(model, va["X"], device, batch_size)
        validation = metrics(va["y"], probs.argmax(1), metadata["classes"])
        score = validation["macro_f1"]
        scheduler.step(score)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(tr["y"]), validation=validation))
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}: loss={history[-1]['train_loss']:.4f}, val macro-F1={score:.4f}", flush=True)
        if score > best + 1e-6:
            best, stale = score, 0
            save_checkpoint(output / "best.pt", model, metadata["preprocessing"],
                            epoch=epoch, validation=validation, seed=seed,
                            data_metadata_sha256=run["data_metadata_sha256"])
        else:
            stale += 1
            if stale >= patience:
                break
    print(f"Best validation macro-F1={best:.4f}; test has not been scored")


def evaluate(data, checkpoint, output, device=None, batch_size=128):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new evaluation directory")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits, metadata = load_data(data)
    model, saved = load_checkpoint(checkpoint, device)
    fingerprint = hashlib.sha256((Path(data)/"metadata.json").read_bytes()).hexdigest()
    if saved["data_metadata_sha256"] != fingerprint:
        raise ValueError("Dataset differs from training; external evaluation requires a separate protocol")
    if saved["preprocessing"] != metadata["preprocessing"] or model.config["classes"] != metadata["classes"]:
        raise ValueError("Checkpoint preprocessing/labels mismatch")
    test = splits["test"]
    probs = predict_batches(model, test["X"], device, batch_size)
    predicted = probs.argmax(1)
    report = metrics(test["y"], predicted, metadata["classes"])
    report["by_subject"] = {s: metrics(test["y"][test["subjects"] == s],
                                      predicted[test["subjects"] == s], metadata["classes"])
                            for s in sorted(set(test["subjects"].tolist()))}
    # Model-only latency, not webcam FPS. CUDA timing requires synchronization.
    x = torch.from_numpy(test["X"][:1]).to(device)
    elapsed = []
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        for _ in range(100):
            if str(device).startswith("cuda"): torch.cuda.synchronize()
            start = time.perf_counter(); model(x)
            if str(device).startswith("cuda"): torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
    report["model_only_latency_ms"] = dict(median=float(np.median(elapsed)), p95=float(np.percentile(elapsed, 95)))
    report["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", y=test["y"], probabilities=probs,
                        subjects=test["subjects"], videos=test["videos"], starts=test["starts"])
    print(json.dumps({k: v for k, v in report.items() if k != "by_subject"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train"); ev = sub.add_parser("evaluate")
    for parser in (tr, ev):
        parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
        parser.add_argument("--device", default=None); parser.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    tr.add_argument("--feature-dim", type=int, choices=[12, 20], default=20)
    tr.add_argument("--hidden", type=int, default=64); tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--patience", type=int, default=8); tr.add_argument("--seed", type=int, default=42)
    ev.add_argument("--checkpoint", required=True)
    args = vars(p.parse_args()); command = args.pop("command")
    (train if command == "train" else evaluate)(**args)


if __name__ == "__main__":
    main()
