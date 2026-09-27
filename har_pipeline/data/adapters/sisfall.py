"""
data/adapters/sisfall.py
========================
Adapter for the SisFall dataset.

Raw file layout expected in raw_dir/SisFall/
---------------------------------------------
    SA01/  SA02/ … SA23/    (older adults, subjects 1-23)
    SE01/  SE02/ … SE15/    (young adults, subjects 1-15)
        <ActivityCode><n>.txt   e.g. D01n01.txt, F01n01.txt

Each .txt file is a recording of one activity trial for one subject.
Each line: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
Values are raw ADC integers.

Unit conversions required
--------------------------
  Accelerometer: raw_adc × (16 / 32768) → g  then × 9.81 → m/s²
  Gyroscope    : raw_adc × (2000 / 32768) → deg/s  then ÷ 57.2958 → rad/s
  Sampling rate: 200 Hz → resample to 50 Hz

Coordinate convention: identity (same as our canonical).
Subject populations: SA = older adults (60-75), SE = young adults (19-30).

Download: http://sistemic.udea.edu.co/en/research/projects/english-falls/
Extract so that raw_dir/SisFall/SA01/D01n01.txt etc. exist.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters.base import BaseAdapter
from data.datasets import SISFALL, map_label

log = logging.getLogger(__name__)

_FS_SOURCE      = 200
_FS_TARGET      = 50
_DOWNSAMPLE     = _FS_SOURCE // _FS_TARGET

# ADC scale factors (from SisFall official specification: Range*2 / 2^Resolution)
_ACC_SCALE      = ((2.0 * 16.0) / (2**13)) * 9.80665    # ADXL345 (13-bit, ±16g) → m/s²
_GYR_SCALE      = ((2.0 * 2000.0) / (2**16)) * (np.pi / 180.0)  # ITG3200 (16-bit, ±2000°/s) → rad/s


class SisFallAdapter(BaseAdapter):

    dataset_id = "sisfall"

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir)
        # Check either SisFall or SisFall_dataset
        candidate = self.raw_dir / "SisFall_dataset"
        if candidate.exists():
            self._data_root = candidate
        else:
            self._data_root = self.raw_dir / "SisFall"

    def is_available(self) -> bool:
        return self._data_root.exists() and any(self._data_root.glob("S*/*.txt"))

    def load(self, max_recordings_per_subject: int | None = None) -> pd.DataFrame:
        self._assert_available()
        log.info("[%s] Scanning %s …", self.dataset_id, self._data_root)

        frames: list[pd.DataFrame] = []
        for subj_dir in sorted(self._data_root.glob("S*/")):
            if not subj_dir.is_dir():
                continue
            age_group = _parse_age_group(subj_dir.name)
            subj_num_match = re.search(r"\d+", subj_dir.name)
            if not subj_num_match:
                continue
            subj_num  = int(subj_num_match.group())

            txt_files = sorted(subj_dir.glob("*.txt"))
            if max_recordings_per_subject:
                txt_files = txt_files[:max_recordings_per_subject]

            for txt_path in txt_files:
                try:
                    df = self._load_single(txt_path, subj_dir.name, subj_num, age_group)
                    if df is not None and not df.empty:
                        frames.append(df)
                except Exception as exc:
                    log.warning("[%s] Skipping %s: %s", self.dataset_id, txt_path.name, exc)

        if not frames:
            raise RuntimeError(f"[{self.dataset_id}] No .txt files loaded from {self._data_root}")

        result = pd.concat(frames, ignore_index=True)
        log.info("[%s] Loaded %d rows from %d recordings",
                 self.dataset_id, len(result), len(frames))
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_single(
        self, path: Path, subj_dir_name: str, subj_num: int, age_group: str
    ) -> pd.DataFrame:
        """
        Parse one SisFall .txt recording.
        Filename: <ActivityCode>_<SubjectID>_<TrialCode>.txt e.g. D01_SA01_R01.txt
        """
        stem        = path.stem
        act_match   = re.match(r"([A-Z]\d+)", stem)
        trial_match = re.search(r"[Rn](\d+)", stem)

        orig_label = act_match.group(1) if act_match else "UNK"
        trial_num  = int(trial_match.group(1)) if trial_match else 0
        unified    = map_label(self.dataset_id, orig_label)

        # Subject ID encodes age group for later stratification analysis
        subj_id  = f"sisfall_{age_group}{subj_num:02d}"
        rec_id   = f"sisfall_{age_group}{subj_num:02d}_{orig_label}_r{trial_num:02d}"

        # Fast line-by-line reading to handle trailing semicolon
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip().rstrip(";").split(",") for line in f if line.strip()]
        if not lines:
            return pd.DataFrame()

        arr = np.array(lines, dtype=np.float64)
        if arr.shape[1] < 6:
            return pd.DataFrame()

        # ADXL345 (cols 0,1,2) and ITG3200 (cols 3,4,5)
        acc = arr[:, 0:3] * _ACC_SCALE
        gyr = arr[:, 3:6] * _GYR_SCALE

        n     = len(acc)
        ts_ms = np.arange(n) * (1000 // _FS_SOURCE) + 1_600_000_000_000

        df_full = pd.DataFrame({
            "timestamp": ts_ms,
            "ax": acc[:, 0].astype(np.float32),
            "ay": acc[:, 1].astype(np.float32),
            "az": acc[:, 2].astype(np.float32),
            "gx": gyr[:, 0].astype(np.float32),
            "gy": gyr[:, 1].astype(np.float32),
            "gz": gyr[:, 2].astype(np.float32),
        })

        # Downsample 200 → 50 Hz via [::4]
        df_ds = df_full.iloc[::_DOWNSAMPLE].reset_index(drop=True)

        df_ds["dataset_id"]     = self.dataset_id
        df_ds["subject_id"]     = subj_id
        df_ds["recording_id"]   = rec_id
        df_ds["original_label"] = orig_label
        df_ds["unified_label"]  = unified
        df_ds["phone_position"] = "waist_belt"
        df_ds["segment"]        = 0
        return df_ds


def _parse_age_group(dir_name: str) -> str:
    """'SA01' → 'older', 'SE01' → 'young'."""
    prefix = dir_name[:2].upper()
    return "older" if prefix == "SA" else "young"
