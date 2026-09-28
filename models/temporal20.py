"""Small recurrent classifiers, with train-only normalization in checkpoints."""
import torch
from torch import nn
from core.features20 import SCHEMA, FEATURE_NAMES


class Temporal20(nn.Module):
    def __init__(self, classes, hidden=64, cell="gru", feature_dim=20, dropout=0.2):
        super().__init__()
        if feature_dim not in (12, 20) or cell not in ("gru", "lstm"):
            raise ValueError("Use 12/20 features and gru/lstm")
        self.config = dict(classes=list(classes), hidden=hidden, cell=cell,
                           feature_dim=feature_dim, dropout=dropout)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("scale", torch.ones(feature_dim))
        self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(
            feature_dim, hidden, num_layers=1, batch_first=True)
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, len(classes)))

    def forward(self, x):
        # Always accept full schema so both ablation variants share the mask.
        if x.ndim != 3 or x.shape[-1] != 20:
            raise ValueError("Temporal20 expects (batch,time,20) geometric20-v2")
        valid = x[..., 19] > 0.5
        if not valid.any(dim=1).all():
            raise ValueError("Cannot classify a window without a face")
        z = (x[..., :len(self.mean)] - self.mean) / self.scale
        z = z.clamp(-10, 10).masked_fill(~valid.unsqueeze(-1), 0)
        states, _ = self.rnn(z)
        weights = self.attention(states).squeeze(-1).masked_fill(~valid, -1e4).softmax(1)
        pooled = (states * weights.unsqueeze(-1)).sum(1)
        return self.head(pooled), weights

    def predict(self, x, confidence_threshold=0.55):
        self.eval()
        with torch.inference_mode():
            logits, weights = self(x)
            confidence, indices = logits.softmax(-1).max(-1)
        return [(int(i) if float(c) >= confidence_threshold else -1,
                 self.config["classes"][int(i)] if float(c) >= confidence_threshold else "UNCERTAIN",
                 float(c), w) for i, c, w in zip(indices, confidence, weights)]


def save_checkpoint(path, model, preprocessing, **metadata):
    torch.save(dict(model_class="Temporal20", schema=SCHEMA,
                    feature_names=FEATURE_NAMES, config=model.config,
                    preprocessing=preprocessing, state_dict=model.state_dict(),
                    **metadata), path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("model_class") != "Temporal20":
        raise ValueError("Expected a versioned Temporal20 checkpoint")
    model = Temporal20(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
