"""
data/adapters/unimib_shar.py
============================
Adapter for the UniMiB SHAR dataset.

Raw file layout expected in raw_dir/
-------------------------------------
    UniMiB-SHAR/
      adl_data.mat          — ADL windows, shape (n_adl, 453)
      adl_labels.mat        — ADL labels, shape (n_adl, 2)  [label_idx, subject_id]
      adl_names.mat         — ADL class names (9 classes)
      fall_data.mat         — Fall windows, shape (n_fall, 453)
      fall_labels.mat       — Fall labels, shape (n_fall, 2)
      fall_names.mat        — Fall class names (9 fall types)

Each window is 453 samples of 3-axis accelerometer at 50 Hz ≈ 9.06 s.
The dataset has NO gyroscope.

Download: https://www.sal.disco.unimib.it/technologies/unimib-shar/
After downloading, extract so the .mat files are at raw_dir/UniMiB-SHAR/*.mat

Unit note: accelerometer values are in m/s² (already SI).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters.base import BaseAdapter, CANONICAL_COLUMNS
from data.datasets import UNIMIB_SHAR, map_label

log = logging.getLogger(__name__)

# Window length for this dataset (samples)
_WINDOW_LEN = 453
_FS         = 50      # Hz
_ACC_UNIT   = "m/s2"  # already SI — no conversion needed
_NO_GYRO    = float("nan")


class UniMiBSHARAdapter(BaseAdapter):

    dataset_id = "unimib_shar"

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir)
        self._data_root = self.raw_dir / "UniMiB-SHAR"

    def is_available(self) -> bool:
        return (self._data_root / "adl_data.mat").exists()

    def load(self) -> pd.DataFrame:
        self._assert_available()
        log.info("[%s] Loading …", self.dataset_id)

        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required to load .mat files: pip install scipy")

        adl_data   = sio.loadmat(self._data_root / "adl_data.mat")["adl_data"]
        adl_labels = sio.loadmat(self._data_root / "adl_labels.mat")["adl_labels"]
        adl_names  = sio.loadmat(self._data_root / "adl_names.mat")["adl_names"]

        fall_data   = sio.loadmat(self._data_root / "fall_data.mat")["fall_data"]
        fall_labels = sio.loadmat(self._data_root / "fall_labels.mat")["fall_labels"]
        fall_names  = sio.loadmat(self._data_root / "fall_names.mat")["fall_names"]

        adl_class_names  = [str(n[0][0]) for n in adl_names]
        fall_class_names = [str(n[0][0]) for n in fall_names]

        rows_adl  = self._expand_windows(adl_data,  adl_labels,  adl_class_names,  "adl")
        rows_fall = self._expand_windows(fall_data, fall_labels, fall_class_names, "fall")

        df = pd.concat([rows_adl, rows_fall], ignore_index=True)
        log.info("[%s] Loaded %d rows from %d windows", self.dataset_id, len(df),
                 len(adl_data) + len(fall_data))
        return df

    # ── Internals ─────────────────────────────────────────────────────────────

    def _expand_windows(
        self,
        data:   np.ndarray,   # (n_windows, 453)   — interleaved ax,ay,az
        labels: np.ndarray,   # (n_windows, 2)     — [class_idx, subject_id]
        names:  list[str],
        kind:   str,
    ) -> pd.DataFrame:
        """
        Each window in UniMiB-SHAR is stored as a flat 453-element vector:
        [ax0,ay0,az0, ax1,ay1,az1, …, ax150,ay150,az150]
        (151 samples × 3 axes — some windows are longer; we reshape robustly)
        """
        n_windows = data.shape[0]
        rows = []
        base_ts = 1_600_000_000_000  # synthetic base Unix ms

        for w_idx in range(n_windows):
            flat          = data[w_idx].astype(np.float32)
            class_idx     = int(labels[w_idx, 0]) - 1   # 1-indexed → 0-indexed
            subject_local = int(labels[w_idx, 1])
            orig_label    = names[class_idx] if class_idx < len(names) else "unknown"
            unified       = map_label(self.dataset_id, orig_label)

            subj_id  = f"unimib_{subject_local:02d}"
            rec_id   = f"unimib_{subject_local:02d}_{kind}_{w_idx:04d}"

            # Reshape: UniMiB interleaves axes in triplicates
            n_samples = len(flat) // 3
            xyz = flat[: n_samples * 3].reshape(n_samples, 3)

            for s_idx, (ax, ay, az) in enumerate(xyz):
                rows.append({
                    "dataset_id":     self.dataset_id,
                    "subject_id":     subj_id,
                    "recording_id":   rec_id,
                    "timestamp":      base_ts + w_idx * _WINDOW_LEN * (1000 // _FS) + s_idx * (1000 // _FS),
                    "ax": ax, "ay": ay, "az": az,
                    "gx": _NO_GYRO, "gy": _NO_GYRO, "gz": _NO_GYRO,
                    "original_label": orig_label,
                    "unified_label":  unified,
                    "phone_position": "trouser_front_pocket",
                    "segment":        w_idx,
                })

        return pd.DataFrame(rows)
