"""
kaggle_setup.py — Chạy cell đầu tiên trong Kaggle notebook

COPY ĐOẠN NÀY VÀO CELL ĐẦU TIÊN CỦA NOTEBOOK KAGGLE:

# ─────────────────────────────────────────────────────────────
import os, sys, subprocess

# 1. Clone từ GitHub (thay YOUR_USERNAME)
REPO_URL = 'https://github.com/YOUR_USERNAME/driver-monitoring-system.git'
REPO_DIR = '/kaggle/working/driver_monitoring'

if not os.path.exists(REPO_DIR):
    print('Cloning...')
    r = subprocess.run(['git', 'clone', REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'])
    print('Pulled latest')

sys.path.insert(0, REPO_DIR)

# 2. Install deps
subprocess.run(['pip', 'install', 'mediapipe>=1.0.0', 'scipy', 'scikit-learn', '-q'])

# 3. Download MediaPipe model (3.7MB)
import urllib.request
os.makedirs(f'{REPO_DIR}/models', exist_ok=True)
mp_path = f'{REPO_DIR}/models/face_landmarker.task'
if not os.path.exists(mp_path):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        mp_path)
    print(f'Downloaded face_landmarker.task ({os.path.getsize(mp_path)/1e6:.1f} MB)')

# 4. Patch model path
import core.face_analyzer as fa
from pathlib import Path
fa._MODEL_PATH = Path(mp_path)

# 5. Check
NTHU_PATH  = '/kaggle/input/nthu-ddd'
OUTPUT_DIR = '/kaggle/working'
print('NTHU-DDD:', 'OK' if os.path.exists(NTHU_PATH) else 'NOT FOUND - attach dataset!')
print('Setup complete!')
# ─────────────────────────────────────────────────────────────
"""
