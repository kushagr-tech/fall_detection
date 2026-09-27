"""
data/loader.py
==============
Loads one or many CSV files produced by the Android Activity Collector app,
cleans sensor readings, and returns a single tidy DataFrame.

Expected CSV columns (case-insensitive):
    timestamp, ax, ay, az, gx, gy, gz, label

Subject IDs are inferred from the filename stem so that the train/test split
can be done per person (preventing data leakage).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
REQUIRED_COLS = {"timestamp"} | set(SENSOR_COLS) | {"label"}

VALID_LABELS = {"standing", "walking", "running", "sitting", "lying", "falling"}

# Physical sanity bounds (SI units)
# Accelerometer: ±2g ≈ ±19.6 m/s²; allow ±4g for vigorous activity
ACC_BOUND = 40.0   # m/s²
# Gyroscope: ±2000 °/s ≈ ±34.9 rad/s; allow ±35 rad/s
GYR_BOUND = 35.0   # rad/s

# Maximum time gap between consecutive rows within one label segment (ms).
# A gap larger than this suggests the recording was paused.
MAX_GAP_MS = 500


# ── Public API ─────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load all CSV files matching *pattern* from *data_dir* and return a single
    cleaned, time-sorted DataFrame with a ``subject`` column added.

    Parameters
    ----------
    data_dir : path to folder containing one CSV per recording session.
               Files should be named ``<subject_id>_*.csv`` or just
               ``<anything>.csv`` (subject inferred from stem before first '_').
    pattern  : glob pattern relative to *data_dir*.

    Returns
    -------
    pd.DataFrame with columns:
        subject, timestamp, ax, ay, az, gx, gy, gz, label
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}."
        )

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = _load_single(f)
            frames.append(df)
            log.info("Loaded %s  →  %d rows", f.name, len(df))
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", f.name, exc)

    if not frames:
        raise RuntimeError("All CSV files failed to load.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["subject", "timestamp"]).reset_index(drop=True)

    _log_summary(combined)
    return combined


def load_single_file(path: str | Path) -> pd.DataFrame:
    """Convenience wrapper for a single CSV."""
    return _load_single(Path(path))


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_single(path: Path) -> pd.DataFrame:
    """Load, validate, and clean one CSV file."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    _check_columns(df, path)

    # Infer subject from filename: "subject01_walking.csv" → "subject01"
    subject = path.stem.split("_")[0]
    df.insert(0, "subject", subject)

    df = _coerce_types(df)
    df = _clean_labels(df)
    df = _remove_physical_outliers(df)
    df = _interpolate_small_gaps(df)
    df = _drop_gap_boundaries(df)

    return df


def _check_columns(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[df["label"].isin(VALID_LABELS)].copy()
    dropped = before - len(df)
    if dropped:
        log.debug("  Dropped %d rows with unknown labels.", dropped)
    return df


def _remove_physical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sensor readings that violate physical bounds with NaN, then
    drop rows that cannot be recovered.
    """
    acc_mask = (df[["ax", "ay", "az"]].abs() > ACC_BOUND).any(axis=1)
    gyr_mask = (df[["gx", "gy", "gz"]].abs() > GYR_BOUND).any(axis=1)
    bad_mask = acc_mask | gyr_mask

    n_bad = bad_mask.sum()
    if n_bad:
        log.debug("  Removing %d rows with out-of-range sensor values.", n_bad)
    df = df[~bad_mask].copy()

    # Drop rows where any sensor value is NaN (coercion failures above)
    before = len(df)
    df = df.dropna(subset=SENSOR_COLS + ["timestamp"])
    dropped = before - len(df)
    if dropped:
        log.debug("  Dropped %d rows with NaN sensor values.", dropped)

    return df


def _interpolate_small_gaps(df: pd.DataFrame, max_gap_rows: int = 3) -> pd.DataFrame:  # noqa: E501
    """
    Linearly interpolate sensor columns over short NaN stretches that arise
    from individual bad readings (not full-segment gaps).  Works per subject
    to avoid bleeding across recordings.
    """
    def _interp_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        g[SENSOR_COLS] = (
            g[SENSOR_COLS]
            .interpolate(method="linear", limit=max_gap_rows, limit_direction="both")
        )
        return g

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return (
            df.groupby("subject", group_keys=False)
            .apply(_interp_group)
            .reset_index(drop=True)
        )


def _drop_gap_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that immediately follow a large timestamp gap (recording pause).
    This prevents the windowing step from creating windows that bridge pauses.
    We mark such rows with a 'segment' counter that the windowing code can use.
    """
    df = df.copy()
    df["segment"] = 0

    def _label_segments(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()
        gaps = g["timestamp"].diff() > MAX_GAP_MS
        g["segment"] = gaps.cumsum().astype(int)
        return g

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return (
            df.groupby("subject", group_keys=False)
            .apply(_label_segments)
            .reset_index(drop=True)
        )


def _log_summary(df: pd.DataFrame) -> None:
    log.info(
        "Dataset: %d rows | %d subjects | label distribution:\n%s",
        len(df),
        df["subject"].nunique(),
        df["label"].value_counts().to_string(),
    )
