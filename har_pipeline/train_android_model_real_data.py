"""
train_android_model_real_data.py
================================
Trains a lightweight 1D-CNN + LSTM PyTorch Mobile model on REAL human sensor data
(SisFall real human dataset + naturalistic Android recordings) and exports it directly
to the Android app assets for on-device live inference.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.neural_net import NeuralNetModel
from data.loader import load_dataset
from features.windowing import make_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_android_real")

ANDROID_ASSETS_DIR = ROOT.parent / "activity_collector" / "app" / "src" / "main" / "assets"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# SisFall unit scale factors
_ACC_SCALE = ((2.0 * 16.0) / (2**13)) * 9.80665
_GYR_SCALE = ((2.0 * 2000.0) / (2**16)) * (np.pi / 180.0)

SISFALL_MAP = {
    # Falls
    "F01": "falling", "F02": "falling", "F03": "falling", "F04": "falling",
    "F05": "falling", "F06": "falling", "F07": "falling", "F08": "falling",
    "F09": "falling", "F10": "falling", "F11": "falling", "F12": "falling",
    "F13": "falling", "F14": "falling", "F15": "falling",
    # ADLs
    "D01": "walking",  "D02": "walking",  "D03": "walking",
    "D04": "running",  "D05": "running",
    "D06": "walking",  "D07": "walking",  "D08": "walking", "D09": "walking", # stairs as walking
    "D10": "standing", "D11": "lying",    "D12": "sitting",
    "D13": "sitting",  "D14": "standing", "D15": "standing",
    "D16": "standing", "D17": "standing", "D18": "standing",
    "D19": "lying",
}

CLASSES = ["falling", "lying", "running", "sitting", "standing", "walking"]


def load_real_data(sisfall_dir: Path, window_size: int = 100, step_size: int = 50):
    log.info("Loading SisFall real data from %s …", sisfall_dir)
    subj_dirs = sorted(sisfall_dir.glob("S*/"))
    if not subj_dirs:
        raise FileNotFoundError(f"SisFall not found at {sisfall_dir}")

    windows_X, windows_y, windows_g = [], [], []

    for subj_dir in subj_dirs:
        if not subj_dir.is_dir():
            continue
        subj_name = subj_dir.name
        # Sample both falls and ADLs across all activities
        files = sorted(subj_dir.glob("D*.txt")) + sorted(subj_dir.glob("F*.txt"))

        for p in files:
            stem = p.stem
            act_match = re.match(r"([A-Z]\d+)", stem)
            if not act_match:
                continue
            orig_act = act_match.group(1)
            is_fall_file = orig_act.startswith("F")
            unified = "falling" if is_fall_file else SISFALL_MAP.get(orig_act, None)
            if unified not in CLASSES and not is_fall_file:
                continue

            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip().rstrip(";").split(",") for line in f if line.strip()]
            if not lines:
                continue

            arr = np.array(lines, dtype=np.float64)
            if arr.shape[1] < 6:
                continue

            acc = arr[:, 0:3] * _ACC_SCALE
            gyr = arr[:, 3:6] * _GYR_SCALE
            sensor_mat = np.hstack([acc, gyr])[::4]  # 200Hz -> 50Hz
            svm_mat = np.linalg.norm(sensor_mat[:, 0:3], axis=1)

            n_rows = len(sensor_mat)
            start = 0

            if is_fall_file:
                peak_idx = int(np.argmax(svm_mat))
                while start + window_size <= n_rows:
                    win_center = start + window_size // 2
                    w = sensor_mat[start : start + window_size]
                    if abs(win_center - peak_idx) <= window_size // 2:
                        windows_X.append(w.astype(np.float32))
                        windows_y.append("falling")
                        windows_g.append(f"sisfall_{subj_name}")
                    elif win_center > peak_idx + window_size:
                        windows_X.append(w.astype(np.float32))
                        windows_y.append("lying")
                        windows_g.append(f"sisfall_{subj_name}")
                    start += step_size
            else:
                while start + window_size <= n_rows:
                    w = sensor_mat[start : start + window_size]
                    windows_X.append(w.astype(np.float32))
                    windows_y.append(unified)
                    windows_g.append(f"sisfall_{subj_name}")
                    start += step_size

    # Also load Android phone recordings from data/raw/own
    own_dir = ROOT / "data" / "raw" / "own"
    if own_dir.exists() and list(own_dir.glob("*.csv")):
        log.info("Including real Android phone recordings from %s …", own_dir)
        df_own = load_dataset(own_dir)
        X_own, y_own, g_own = make_windows(df_own, window_size=window_size, step_size=step_size)
        for w, lbl, g in zip(X_own, y_own, g_own):
            if lbl in CLASSES:
                windows_X.append(w.astype(np.float32))
                windows_y.append(lbl)
                windows_g.append(str(g))

    X = np.stack(windows_X, axis=0)
    y = np.array(windows_y)
    groups = np.array(windows_g)
    log.info("Total Real Dataset: %d windows across %d subjects. Distribution:\n%s",
             len(X), len(np.unique(groups)), pd.Series(y).value_counts().to_string())
    return X, y, groups


def main():
    log.info("Starting Mobile Model Training on Real Human Data...")
    sisfall_dir = ROOT / "data" / "raw" / "sisfall" / "SisFall_dataset"
    X, y, groups = load_real_data(sisfall_dir)

    # Subject-independent split (80% train subjects, 20% test subjects)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    g_train          = groups[train_idx]

    train_subjs = np.unique(g_train)
    test_subjs  = np.unique(groups[test_idx])
    log.info("Train subjects (%d) | Test subjects (%d)", len(train_subjs), len(test_subjs))

    # Internal validation split for early stopping
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    st, sv = next(gss_val.split(X_train, y_train, g_train))

    # Initialize NeuralNetModel with mobile hyperparameters
    hparams = {
        "epochs": 35,
        "batch_size": 64,
        "lr": 1.5e-3,
        "weight_decay": 1e-3,
        "dropout": 0.35,
        "patience": 12,
    }
    nn_model = NeuralNetModel(hparams=hparams)
    log.info("Fitting Neural Network on real human windows (%d train, %d val) …", len(st), len(sv))
    nn_model.fit(X_train[st], y_train[st], X_val=X_train[sv], y_val=y_train[sv])

    # Evaluate on held-out human test subjects
    log.info("Evaluating on held-out human test subjects (%d windows) …", len(X_test))
    y_pred = nn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=CLASSES, zero_division=0)
    log.info("Evaluation on Unseen Human Subjects:\nAccuracy: %.4f | Macro F1: %.4f\n%s", acc, mf1, report)

    # Export to PyTorch Mobile Lite (.ptl)
    mobile_model_path = MODEL_DIR / "har_model_real.ptl"
    labels_path = MODEL_DIR / "har_labels_real.txt"
    nn_model.export_torchscript(mobile_model_path, window_size=100)

    classes_ordered = list(nn_model.label_encoder.classes_)
    labels_path.write_text("\n".join(classes_ordered) + "\n")
    log.info("Exported mobile model to %s with classes: %s", mobile_model_path, classes_ordered)

    # Sync directly to Android App Assets
    if ANDROID_ASSETS_DIR.exists():
        android_model_dest = ANDROID_ASSETS_DIR / "har_model.ptl"
        android_labels_dest = ANDROID_ASSETS_DIR / "har_labels.txt"
        shutil.copy2(mobile_model_path, android_model_dest)
        shutil.copy2(labels_path, android_labels_dest)
        log.info("✓ Successfully synced real-world model + labels to Android assets:")
        log.info("   -> %s", android_model_dest)
        log.info("   -> %s", android_labels_dest)
    else:
        log.warning("Android assets directory not found at %s", ANDROID_ASSETS_DIR)


if __name__ == "__main__":
    main()
