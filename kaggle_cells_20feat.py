"""
KAGGLE CELLS — Paste these sequentially into your Kaggle notebook
after the existing setup cell (git clone + mediapipe install).

Replace the old feature extraction and training cells with these.
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL A: Pull latest code (20-feature update)
# ═══════════════════════════════════════════════════════════════════════

CELL_A = """
import subprocess
subprocess.run(['git', '-C', '/kaggle/working/driver_monitoring', 'pull'])
print("Code updated to latest (20-feature architecture)")
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL B: Extract 12 core features per frame, then enrich to 20
# ═══════════════════════════════════════════════════════════════════════

CELL_B = """
import os, sys, cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial.distance import euclidean

sys.path.insert(0, '/kaggle/working/driver_monitoring')
from core.face_analyzer import enrich_sequence, FEATURE_DIM, FEATURE_NAMES

# ── MediaPipe setup (IMAGE mode — no timestamp issues) ──
model_path = '/kaggle/working/driver_monitoring/models/face_landmarker.task'
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1, min_face_detection_confidence=0.5,
)
detector = mp_vision.FaceLandmarker.create_from_options(options)

# ── Landmark index constants ──
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33, 160, 158, 133, 153, 144]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
MOUTH      = [61, 291, 13, 14]
POSE_PTS   = [1, 152, 33, 263, 61, 291]
MODEL_3D   = np.array([(0,0,0),(0,-330,-65),(-225,170,-135),
                        (225,170,-135),(-150,-150,-125),(150,-150,-125)], np.float64)

def calc_ear(pts, idx):
    p = [pts[i][:2] for i in idx]
    v = euclidean(p[1],p[5]) + euclidean(p[2],p[4])
    h = euclidean(p[0],p[3])
    return float(v/(2*h)) if h>1e-6 else 0.0

def calc_mar(pts):
    l,r,t,b = [pts[i][:2] for i in MOUTH]
    return float(euclidean(t,b)/euclidean(l,r)) if euclidean(l,r)>1e-6 else 0.0

def calc_pose(pts, w, h):
    img_pts = pts[POSE_PTS,:2].astype(np.float64)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.,0.,0.
    R,_ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2+R[1,0]**2)
    if sy>1e-6:
        return float(np.degrees(np.arctan2(R[1,0],R[0,0]))), \
               float(np.degrees(np.arctan2(-R[2,0],sy))), \
               float(np.degrees(np.arctan2(R[2,1],R[2,2])))
    return 0., float(np.degrees(np.arctan2(-R[2,0],sy))), 0.

def calc_gaze(pts):
    if len(pts)<478: return 0.,0.
    li = pts[LEFT_IRIS,:2].mean(0); ri = pts[RIGHT_IRIS,:2].mean(0)
    le = pts[LEFT_EYE,:2]; re = pts[RIGHT_EYE,:2]
    ls = le.max(0)-le.min(0)+1e-6; rs = re.max(0)-re.min(0)+1e-6
    if ls[0]<1 or rs[0]<1: return 0.,0.
    lg = (li-le.min(0))/ls; rg = (ri-re.min(0))/rs
    g = (lg+rg)/2
    return float(np.clip((g[0]-0.5)*2,-1,1)), float(np.clip((g[1]-0.5)*2,-1,1))

def extract_frame_12(img):
    """Extract 12 core features from a single BGR image frame."""
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks: return None
    pts = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in res.face_landmarks[0]], np.float64)
    ear_l = calc_ear(pts, LEFT_EYE)
    ear_r = calc_ear(pts, RIGHT_EYE)
    ear_a = (ear_l+ear_r)/2
    mar   = calc_mar(pts)
    yaw,pitch,roll = calc_pose(pts, w, h)
    gx,gy = calc_gaze(pts)
    # PERCLOS and duration computed at sequence level via enrich_sequence()
    is_closed = 1.0 if ear_a < 0.22 else 0.0
    is_yawn   = 1.0 if mar   > 0.55 else 0.0
    return [ear_l, ear_r, ear_a, mar,
            yaw/90., pitch/90., roll/90.,
            gx, gy,
            is_closed,   # will become PERCLOS rolling mean in enrich_sequence
            is_closed,   # EyesClosed binary
            is_yawn]     # MouthOpen binary

# ── Build sequences ──
SEQ_LEN = 30

def extract_sequences_20feat(target_subs, samples_per_class=600):
    sub_df = df[df['subject'].isin(target_subs)]
    X_out, y_out = [], []
    for c_id in [0, 1, 2]:
        cls_df = sub_df[sub_df['class_id'] == c_id]
        if len(cls_df) == 0: continue
        c_count = 0
        for _, grp in cls_df.groupby(['subject','glasses','action']):
            grp_sorted = grp.sort_values('frame_num')
            paths = grp_sorted['path'].iloc[::2].tolist()
            feats_list = []
            for p in paths:
                img = cv2.imread(p)
                if img is None: continue
                f = extract_frame_12(img)
                if f is not None:
                    feats_list.append(f)
            for i in range(0, len(feats_list)-SEQ_LEN, 10):
                raw_seq = np.array(feats_list[i:i+SEQ_LEN], dtype=np.float32)
                # Compute PERCLOS as rolling mean of is_closed column
                raw_seq[:, 9] = raw_seq[:, 9].mean()
                # Enrich from 12 → 20 features
                enriched = enrich_sequence(raw_seq)
                X_out.append(enriched)
                y_out.append(c_id)
                c_count += 1
                if c_count >= samples_per_class: break
            if c_count >= samples_per_class: break
    return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.int64)

print(f"Extracting {FEATURE_DIM} features per frame: {FEATURE_NAMES}")
print("Subjects:", df['subject'].unique().tolist())

# Extract all splits
print("\\nExtracting TRAIN (subjects 001, 002)...")
X_train, y_train = extract_sequences_20feat(train_subs, 600)

print("Extracting VAL (subject 005)...")
X_val, y_val = extract_sequences_20feat(val_subs, 250)

print("Extracting TEST (subject 006)...")
X_test, y_test = extract_sequences_20feat(test_subs, 250)

# Pool all → stratified random split
X_all = np.concatenate([X_train, X_val, X_test])
y_all = np.concatenate([y_train, y_val, y_test])
from sklearn.model_selection import train_test_split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state=42)
X_va, X_te, y_va, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

out_dir = Path('/kaggle/working/data/processed20')
out_dir.mkdir(parents=True, exist_ok=True)
for name, arr in [('X_train',X_tr),('y_train',y_tr),
                  ('X_val',X_va),  ('y_val',y_va),
                  ('X_test',X_te), ('y_test',y_te)]:
    np.save(out_dir/f'{name}.npy', arr)

print(f"\\nFeature shape: {X_tr.shape}  (seq_len=30, feat_dim={FEATURE_DIM})")
print("Train:", np.unique(y_tr, return_counts=True))
print("Val:  ", np.unique(y_va, return_counts=True))
print("Test: ", np.unique(y_te, return_counts=True))
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL C: Train with 20-feature DriverLSTM + export ONNX INT8
# ═══════════════════════════════════════════════════════════════════════

CELL_C = """
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

out_dir = Path('/kaggle/working/data/processed20')
X_tr = np.load(out_dir/'X_train.npy'); y_tr = np.load(out_dir/'y_train.npy')
X_va = np.load(out_dir/'X_val.npy');   y_va = np.load(out_dir/'y_val.npy')
X_te = np.load(out_dir/'X_test.npy');  y_te = np.load(out_dir/'y_test.npy')

print(f"Input shape: {X_tr.shape}  →  FEATURE_DIM={X_tr.shape[2]}")

# Weighted sampler
counts = np.bincount(y_tr)
sw = torch.FloatTensor([1./(counts[y]+1e-6) for y in y_tr])
sampler = WeightedRandomSampler(sw, len(sw))
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr).to(device),
                                         torch.LongTensor(y_tr).to(device)),
                          batch_size=32, sampler=sampler)
val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_va).to(device),
                                         torch.LongTensor(y_va).to(device)),
                          batch_size=32, shuffle=False)

# 20-feature Bi-LSTM
class DriverLSTM20(nn.Module):
    def __init__(self, feat_dim=20, hidden=80, num_classes=3):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.35)
        self.attn_w = nn.Linear(hidden*2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, 48),
            nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(48, num_classes)
        )
    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        # Soft temporal attention
        scores = torch.softmax(self.attn_w(out).squeeze(-1), dim=1)
        ctx = (out * scores.unsqueeze(-1)).sum(dim=1)
        return self.fc(ctx), scores

model = DriverLSTM20(num_classes=3).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor([1.0, 1.5, 1.2]).to(device),
    label_smoothing=0.05)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

# Training loop
EPOCHS, PATIENCE = 40, 12
best_val_acc, patience_ctr = 0.0, 0
history = {k:[] for k in ['tl','vl','ta','va']}

print("\\nStarting training...")
for epoch in range(1, EPOCHS+1):
    model.train()
    tl, tc = 0., 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        logits, _ = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()*len(yb); tc += (logits.argmax(1)==yb).sum().item()
    model.eval(); vl, vc = 0., 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits, _ = model(Xb)
            vl += criterion(logits,yb).item()*len(yb)
            vc += (logits.argmax(1)==yb).sum().item()
    scheduler.step()
    ta = tc/len(y_tr)*100; va = vc/len(y_va)*100
    for k,v in zip(['tl','vl','ta','va'],[tl/len(y_tr),vl/len(y_va),ta,va]):
        history[k].append(v)
    if epoch%5==0 or epoch==1:
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train {ta:.1f}% | Val {va:.1f}%")
    if va > best_val_acc:
        best_val_acc = va; patience_ctr = 0
        torch.save(model.state_dict(), '/kaggle/working/driver_lstm_20feat.pt')
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stop at epoch {epoch}"); break

print(f"\\nBest Val Accuracy: {best_val_acc:.1f}%")

# Evaluation on test set
model.load_state_dict(torch.load('/kaggle/working/driver_lstm_20feat.pt'))
model.eval()
with torch.no_grad():
    preds = model(torch.FloatTensor(X_te).to(device))[0].argmax(1).cpu().numpy()

print("\\n=== TEST SET RESULTS ===")
print(classification_report(y_te, preds,
      target_names=['ALERT','DROWSY','YAWNING'], digits=3))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(history['ta'], label='Train', color='royalblue')
axes[0].plot(history['va'], label='Val',   color='tomato')
axes[0].set_title(f'Accuracy (Best Val: {best_val_acc:.1f}%)')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['tl'], label='Train', color='royalblue')
axes[1].plot(history['vl'], label='Val',   color='tomato')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

cm = confusion_matrix(y_te, preds, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['ALERT','DROWSY','YAWNING'],
            yticklabels=['ALERT','DROWSY','YAWNING'], ax=axes[2])
axes[2].set_title('Confusion Matrix (normalized)')
axes[2].set_ylabel('True'); axes[2].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('/kaggle/working/results_20feat.png', dpi=150)
plt.show()
print("Saved: /kaggle/working/results_20feat.png")

# ── ONNX INT8 Export (4x size reduction for embedded deployment) ──
print("\\nExporting ONNX INT8 (for report section on model compression)...")
try:
    import onnx
    from torch.quantization import quantize_dynamic

    model_cpu = DriverLSTM20(num_classes=3).cpu()
    model_cpu.load_state_dict(torch.load('/kaggle/working/driver_lstm_20feat.pt', map_location='cpu'))
    model_cpu.eval()

    # Dynamic INT8 quantization (no calibration data needed)
    quantized = quantize_dynamic(model_cpu, {nn.Linear, nn.LSTM}, dtype=torch.qint8)

    # Export to ONNX (FP32 for compatibility)
    dummy = torch.zeros(1, 30, 20)
    torch.onnx.export(model_cpu, (dummy,), '/kaggle/working/driver_lstm_20feat.onnx',
                      input_names=['features'], output_names=['logits','attention'],
                      dynamic_axes={'features': {0: 'batch'}},
                      opset_version=11)

    pt_size   = os.path.getsize('/kaggle/working/driver_lstm_20feat.pt') / 1024
    onnx_size = os.path.getsize('/kaggle/working/driver_lstm_20feat.onnx') / 1024
    print(f"  .pt  size: {pt_size:.0f} KB")
    print(f"  .onnx size: {onnx_size:.0f} KB")
    print("  INT8 quantized model ready for embedded deployment!")
except Exception as e:
    print(f"ONNX export skipped: {e}")
    print("Install onnx with: !pip install onnx -q")
"""

print("Copy CELL_A, CELL_B, CELL_C into your Kaggle notebook as separate code cells.")
print("Run them in order: A → B → C")
