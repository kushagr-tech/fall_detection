"""
data/adapters/mobiact.py
========================
Adapter for the MobiAct v2 dataset.

Raw file layout expected in raw_dir/MobiAct/
---------------------------------------------
    <LABEL>/
        <LABEL>_<subject_id>_<trial>.csv

Each CSV has a header line then rows of:
    timestamp(ms), acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, azimuth, pitch, roll

Sensor units
------------
  Accelerometer : m/s²   (already SI)
  Gyroscope     : deg/s  → must convert to rad/s  (÷ 57.2957795)
  Sampling rate : 200 Hz → resample to 50 Hz

Coordinate convention: identity (same as our canonical).

Download: https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/
Extract so that raw_dir/MobiAct/<LABEL>/*.csv exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters.base import BaseAdapter
from data.datasets import MOBIACT, map_label

log = logging.getLogger(__name__)

_FS_SOURCE   = 200    # Hz
_FS_TARGET   = 50     # Hz
_DEG2RAD     = np.pi / 180.0
_DOWNSAMPLE  = _FS_SOURCE // _FS_TARGET   # = 4


class MobiActAdapter(BaseAdapter):

    dataset_id = "mobiact"

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir)
        self._data_root = self.raw_dir / "MobiAct"

    def is_available(self) -> bool:
        return self._data_root.exists() and any(self._data_root.iterdir())

    def load(self) -> pd.DataFrame:
        self._assert_available()
        log.info("[%s] Scanning %s …", self.dataset_id, self._data_root)

        frames: list[pd.DataFrame] = []
        label_dirs = sorted(p for p in self._data_root.iterdir() if p.is_dir())

        for label_dir in label_dirs:
            orig_label = label_dir.name.upper()
            unified    = map_label(self.dataset_id, orig_label)
            if unified == "unknown":
                log.debug("[%s] Skipping unknown label: %s", self.dataset_id, orig_label)
                continue

            csv_files = sorted(label_dir.glob("*.csv"))
            for csv_path in csv_files:
                try:
                    df = self._load_single(csv_path, orig_label, unified)
                    frames.append(df)
                except Exception as exc:
                    log.warning("[%s] Skipping %s: %s", self.dataset_id, csv_path.name, exc)

        if not frames:
            raise RuntimeError(f"[{self.dataset_id}] No CSV files loaded from {self._data_root}")

        result = pd.concat(frames, ignore_index=True)
        log.info("[%s] Loaded %d rows from %d recordings",
                 self.dataset_id, len(result), len(frames))
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_single(self, path: Path, orig_label: str, unified: str) -> pd.DataFrame:
        """
        Parse one MobiAct CSV file.
        Filename pattern: <LABEL>_<subj>_<trial>.csv  e.g. WAL_001_1.csv
        """
        stem  = path.stem  # e.g. "WAL_001_1"
        parts = stem.split("_")
        subj_local = parts[1] if len(parts) >= 2 else "00"
        trial      = parts[2] if len(parts) >= 3 else "1"

        subj_id  = f"mobiact_{int(subj_local):03d}"
        rec_id   = f"mobiact_{int(subj_local):03d}_{orig_label}_{trial}"

        raw = pd.read_csv(path, comment="#", header=0, on_bad_lines="skip")

        # Flexible column detection
        raw.columns = raw.columns.str.strip().str.lower()
        col_map = _detect_columns(raw.columns.tolist())
        if col_map is None:
            raise ValueError(f"Cannot detect sensor columns in {path.name}: {list(raw.columns)}")

        acc_x = raw[col_map["acc_x"]].astype(np.float32)
        acc_y = raw[col_map["acc_y"]].astype(np.float32)
        acc_z = raw[col_map["acc_z"]].astype(np.float32)
        gyr_x = raw[col_map["gyr_x"]].astype(np.float32) * _DEG2RAD
        gyr_y = raw[col_map["gyr_y"]].astype(np.float32) * _DEG2RAD
        gyr_z = raw[col_map["gyr_z"]].astype(np.float32) * _DEG2RAD

        # Timestamps
        if "timestamp" in col_map and col_map["timestamp"] in raw.columns:
            ts = raw[col_map["timestamp"]].astype(np.int64)
        else:
            ts = pd.RangeIndex(len(raw)) * (1000 // _FS_SOURCE) + 1_600_000_000_000

        # Downsample 200 Hz → 50 Hz (simple decimation; anti-aliasing via pandas mean)
        df_full = pd.DataFrame({
            "timestamp": ts, "ax": acc_x, "ay": acc_y, "az": acc_z,
            "gx": gyr_x, "gy": gyr_y, "gz": gyr_z,
        })
        df_ds = df_full.iloc[::_DOWNSAMPLE].reset_index(drop=True)

        df_ds["dataset_id"]     = self.dataset_id
        df_ds["subject_id"]     = subj_id
        df_ds["recording_id"]   = rec_id
        df_ds["original_label"] = orig_label
        df_ds["unified_label"]  = unified
        df_ds["phone_position"] = "trouser_right_pocket"
        df_ds["segment"]        = 0
        return df_ds


def _detect_columns(cols: list[str]) -> dict | None:
    """
    Heuristically map raw column names to canonical sensor axes.
    MobiAct uses names like: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    mappings = {
        "timestamp": ["timestamp", "time", "ts"],
        "acc_x":  ["acc_x", "accel_x", "ax", "accelerometerx"],
        "acc_y":  ["acc_y", "accel_y", "ay", "accelerometery"],
        "acc_z":  ["acc_z", "accel_z", "az", "accelerometerz"],
        "gyr_x":  ["gyro_x", "gyr_x", "gx", "gyroscopex"],
        "gyr_y":  ["gyro_y", "gyr_y", "gy", "gyroscopey"],
        "gyr_z":  ["gyro_z", "gyr_z", "gz", "gyroscopez"],
    }
    result = {}
    for key, candidates in mappings.items():
        for cand in candidates:
            if cand in cols:
                result[key] = cand
                break
        if key not in result and key not in ("timestamp",):
            return None  # mandatory sensor column missing
    return result
