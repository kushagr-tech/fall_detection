"""
features/windowing.py
=====================
Converts a continuous sensor stream into fixed-length, overlapping windows.

Key design choices
------------------
* Windows NEVER span a recording-pause boundary (segment column from loader).
* Windows NEVER span a subject boundary (subject column).
* The label for a window is determined by majority vote of its rows.
* A window is only kept if at least `min_label_purity` of rows share the
  majority label (default 0.9).  This prevents mixed-activity windows from
  adding noise to training.
* For fall-detection the fall event is very short (< 1 s).  A window is
  automatically tagged 'falling' if it contains ANY falling row, even if
  falling rows are the minority.  This is the correct behaviour for safety-
  critical recall.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from data.loader import SENSOR_COLS

log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE = 100   # rows @ 50 Hz  =  2 seconds
DEFAULT_STEP_SIZE   = 50    # 50 % overlap
MIN_LABEL_PURITY    = 0.6   # minimum fraction for majority-vote label
FALL_LABEL          = "falling"


# ── Public API ─────────────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    min_purity: float = MIN_LABEL_PURITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over every (subject, segment) group and collect windows.

    Parameters
    ----------
    df          : cleaned DataFrame from ``data.loader``
    window_size : number of sensor rows per window
    step_size   : stride between window starts
    min_purity  : minimum label homogeneity; windows below this are discarded
                  (except windows containing any 'falling' row — always kept)

    Returns
    -------
    X : np.ndarray of shape (n_windows, window_size, n_sensors=6)
    y : np.ndarray of shape (n_windows,) — string labels
    groups : np.ndarray of shape (n_windows,) — subject IDs for GroupKFold
    """
    windows_X: list[np.ndarray] = []
    windows_y: list[str] = []
    windows_g: list[str] = []

    df = df.reset_index(drop=True)
    sensor_data = df[SENSOR_COLS].values.astype(np.float32)

    for (subject, segment), grp_idx in df.groupby(
        ["subject", "segment"], sort=False
    ).groups.items():
        grp_idx_sorted = df.loc[grp_idx].sort_values("timestamp").index
        local_X = sensor_data[grp_idx_sorted]
        local_y = df.loc[grp_idx_sorted, "label"].values

        n = len(local_X)
        if n < window_size:
            continue

        start = 0
        while start + window_size <= n:
            w_X = local_X[start : start + window_size]
            w_y = local_y[start : start + window_size]

            label, purity = _majority_label(w_y)

            # Always keep windows that contain even a single fall row
            has_fall = FALL_LABEL in w_y
            if has_fall:
                label = FALL_LABEL

            if not has_fall and purity < min_purity:
                start += step_size
                continue

            windows_X.append(w_X)
            windows_y.append(label)
            windows_g.append(str(subject))
            start += step_size

    if not windows_X:
        raise RuntimeError(
            "No windows were created. Check window_size vs. dataset length."
        )

    X = np.stack(windows_X, axis=0)
    y = np.array(windows_y)
    groups = np.array(windows_g)

    _log_window_summary(y, groups)
    return X, y, groups


# ── Helpers ────────────────────────────────────────────────────────────────────

def _majority_label(labels: np.ndarray) -> tuple[str, float]:
    values, counts = np.unique(labels, return_counts=True)
    idx = counts.argmax()
    return str(values[idx]), counts[idx] / len(labels)


def _log_window_summary(y: np.ndarray, groups: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    log.info(
        "Windows created: %d total | %d subjects | distribution:\n%s",
        len(y),
        len(np.unique(groups)),
        "\n".join(f"  {lbl}: {cnt}" for lbl, cnt in zip(unique, counts)),
    )
