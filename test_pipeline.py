import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

print('=== TEST 1: Import modules ===')
from core.face_analyzer import FaceAnalyzer, FaceFeatures
from core.attention_scorer import AttentionScorer, ATTENTION_STATES
from models.lstm_classifier import DriverStateLSTM, DriverStateLSTM_Lite, CLASSES
from training.dataset_builder import DatasetBuilder, FEATURE_NAMES
print('  All imports OK')

print()
print('=== TEST 2: FaceFeatures + AttentionScorer ===')
feats = FaceFeatures()
feats.ear_avg = 0.15
feats.mar = 0.12
feats.yaw = 5.0
feats.pitch = -5.0
feats.gaze_x = 0.1
feats.gaze_y = 0.1
feats.perclos = 0.45
feats.eyes_closed_duration = 2.5
feats.blink_rate = 8.0
feats.gaze_zone = 'EYES_CLOSED'

scorer = AttentionScorer()
score, state, label = scorer.update(feats)
print(f'  DROWSY scenario: score={score}%, state={state}, label={label}')
assert state in ('DROWSY', 'DANGER'), f'Expected DROWSY/DANGER, got {state}'
print('  Drowsy detection: OK')

feats2 = FaceFeatures()
feats2.ear_avg = 0.35
feats2.mar = 0.08
feats2.yaw = 2.0
feats2.pitch = 1.0
feats2.gaze_x = 0.05
feats2.perclos = 0.02
feats2.eyes_closed_duration = 0.0
feats2.blink_rate = 16.0
feats2.gaze_zone = 'FORWARD'
score2, state2, label2 = scorer.update(feats2)
print(f'  FOCUSED scenario: score={score2}%, state={state2}')

print()
print('=== TEST 3: LSTM Model ===')
import torch
model = DriverStateLSTM()
n = sum(p.numel() for p in model.parameters())
print(f'  DriverStateLSTM params: {n:,}')

model_lite = DriverStateLSTM_Lite()
n_lite = sum(p.numel() for p in model_lite.parameters())
print(f'  DriverStateLSTM_Lite params: {n_lite:,}')

x = torch.randn(4, 30, 12)
logits, attn = model(x)
print(f'  Forward pass: input {x.shape} -> logits {logits.shape}, attn {attn.shape}')

results = model.predict(x)
print(f'  Predictions: {[(r[1], f"{r[2]*100:.0f}%") for r in results]}')

print()
print('=== TEST 4: Synthetic Dataset + Bias Analysis ===')
import numpy as np
from training.train_lstm import generate_synthetic_dataset
X, y = generate_synthetic_dataset(n_per_class=100)
print(f'  Synthetic dataset: X={X.shape}, y={y.shape}')
print(f'  Class distribution: {dict(zip(CLASSES, [int((y==i).sum()) for i in range(4)]))}')

builder = DatasetBuilder()
bias = builder.analyze_dataset_bias(X, y)

print()
print('=== TEST 5: Augmentation ===')
X_aug, y_aug = DatasetBuilder.augment_sequences(X, y)
print(f'  After augmentation: X={X_aug.shape}, y={y_aug.shape}')
print(f'  Augmented classes: {dict(zip(CLASSES, [int((y_aug==i).sum()) for i in range(4)]))}')

print()
print('ALL TESTS PASSED')
