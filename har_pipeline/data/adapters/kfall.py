"""
data/adapters/kfall.py
======================
Adapter for the KFall dataset.

Raw file layout expected in raw_dir/KFall/
-------------------------------------------
    KFall_dataset/
      SubjectID/
        <SubjectID>_<ActivityCode>_<TrialID>_<SensorPos>.csv
        e.g.  S01_ADL03_T01_waist.csv

Each CSV has a header with columns:
    SampleTimeFine, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z

Sensor units
------------
  Accelerometer : m/s²   (already SI — no conversion needed)
  Gyroscope     : rad/s  (already SI — no conversion needed)
  Sampling rate : 100 Hz → resample to 50 Hz

Two sensor positions per recording: waist, wrist.
We load both and tag phone_position accordingly.

Download: https://github.com/ylysa/KFall-Dataset
License : CC BY 4.0
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters.base import BaseAdapter
from data.datasets import KFALL, map_label

log = logging.getLogger(__name__)

_FS_SOURCE   = 100
_FS_TARGET   = 50
_DOWNSAMPLE  = _FS_SOURCE // _FS_TARGET


class KFallAdapter(BaseAdapter):

    dataset_id = "kfall"

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir)
        self._data_root = self.raw_dir / "KFall" / "KFall_dataset"

    def is_available(self) -> bool:
        return self._data_root.exists() and any(self._data_root.glob("S*/*.csv"))

    def load(self) -> pd.DataFrame:
        self._assert_available()
        log.info("[%s] Scanning %s …", self.dataset_id, self._data_root)

        frames: list[pd.DataFrame] = []
        for subj_dir in sorted(self._data_root.glob("S*/")):
            if not subj_dir.is_dir():
                continue
            subj_match = re.search(r"S(\d+)", subj_dir.name)
            if not subj_match:
                continue
            subj_num = int(subj_match.group(1))

            for csv_path in sorted(subj_dir.glob("*.csv")):
                try:
                    df = self._load_single(csv_path, subj_num)
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

    def _load_single(self, path: Path, subj_num: int) -> pd.DataFrame:
        """
        Parse one KFall CSV recording.
        Filename: <SubjID>_<ActivityCode>_<TrialID>_<SensorPos>.csv
        """
        stem   = path.stem   # e.g. "S01_ADL03_T01_waist"
        parts  = stem.split("_")

        act_code     = parts[1] if len(parts) > 1 else "UNK"
        trial_id     = parts[2] if len(parts) > 2 else "T01"
        sensor_pos   = parts[3] if len(parts) > 3 else "unknown"

        unified = map_label(self.dataset_id, act_code)

        subj_id  = f"kfall_{subj_num:02d}"
        rec_id   = f"kfall_{subj_num:02d}_{act_code}_{trial_id}_{sensor_pos}"

        raw = pd.read_csv(path, on_bad_lines="skip")
        raw.columns = raw.columns.str.strip().str.lower()

        col_map = _detect_columns(raw.columns.tolist())
        if col_map is None:
            raise ValueError(f"Cannot detect sensor columns in {path.name}")

        # KFall is already in m/s² and rad/s — no conversion needed
        df_full = pd.DataFrame({
            "timestamp": _make_timestamps(len(raw), _FS_SOURCE),
            "ax": raw[col_map["acc_x"]].astype(np.float32).values,
            "ay": raw[col_map["acc_y"]].astype(np.float32).values,
            "az": raw[col_map["acc_z"]].astype(np.float32).values,
            "gx": raw[col_map["gyr_x"]].astype(np.float32).values,
            "gy": raw[col_map["gyr_y"]].astype(np.float32).values,
            "gz": raw[col_map["gyr_z"]].astype(np.float32).values,
        })

        # Downsample 100 → 50 Hz
        df_ds = df_full.iloc[::_DOWNSAMPLE].reset_index(drop=True)

        df_ds["dataset_id"]     = self.dataset_id
        df_ds["subject_id"]     = subj_id
        df_ds["recording_id"]   = rec_id
        df_ds["original_label"] = act_code
        df_ds["unified_label"]  = unified
        df_ds["phone_position"] = _normalise_position(sensor_pos)
        df_ds["segment"]        = 0
        return df_ds


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_columns(cols: list[str]) -> dict | None:
    mappings = {
        "acc_x": ["acc_x", "accel_x", "ax", "accelerometerx"],
        "acc_y": ["acc_y", "accel_y", "ay", "accelerometery"],
        "acc_z": ["acc_z", "accel_z", "az", "accelerometerz"],
        "gyr_x": ["gyr_x", "gyro_x", "gx", "gyroscopex"],
        "gyr_y": ["gyr_y", "gyro_y", "gy", "gyroscopey"],
        "gyr_z": ["gyr_z", "gyro_z", "gz", "gyroscopez"],
    }
    result = {}
    for key, candidates in mappings.items():
        for cand in candidates:
            if cand in cols:
                result[key] = cand
                break
        if key not in result:
            return None
    return result


def _normalise_position(raw: str) -> str:
    raw = raw.lower().strip()
    if "waist" in raw:
        return "waist_belt"
    if "wrist" in raw:
        return "wrist"
    return raw


def _make_timestamps(n: int, fs: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) * (1000 // fs) + 1_600_000_000_000
