"""
data/harmonizer.py
==================
Merges output from multiple adapters into one analysis-ready DataFrame.

Responsibilities
----------------
1.  Load each available adapter (skips datasets not yet downloaded).
2.  Validate that every row has exactly the canonical columns.
3.  Handle gyroscope-less datasets (fill gx/gy/gz with 0.0 and set a flag
    so windowing / features can treat them specially).
4.  Drop windows labelled "unknown" — they carry no training signal.
5.  Add a global unique `subject_id` (already prefixed by each adapter).
6.  Compute and log a diversity summary.
7.  Save the merged parquet to processed/ for fast re-loading.

This module does NOT window, normalise, or feature-engineer.
Those steps remain in features/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters import (
    UniMiBSHARAdapter, MobiActAdapter, SisFallAdapter,
    KFallAdapter, OwnRecordingsAdapter, CANONICAL_COLUMNS,
)

log = logging.getLogger(__name__)

PROCESSED_PARQUET = Path(__file__).parent / "processed" / "harmonized.parquet"

# Columns added during harmonization (on top of CANONICAL_COLUMNS)
EXTRA_COLUMNS = ["has_gyro", "age_group"]

# Labels we actually want to train on (drop the rest)
TRAINABLE_LABELS = {
    "standing", "walking", "running", "sitting", "lying",
    "falling", "stairs_up", "stairs_down", "transition",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def harmonize(
    raw_root:    Path,
    own_raw_dir: Path | None = None,
    cache:       bool        = True,
) -> pd.DataFrame:
    """
    Load all available datasets, merge, clean, and return a unified DataFrame.

    Parameters
    ----------
    raw_root    : parent of unimib_shar/, mobiact/, sisfall/, kfall/ subdirs
    own_raw_dir : directory with our own Android CSV recordings (optional)
    cache       : if True, write/read a parquet cache at PROCESSED_PARQUET

    Returns
    -------
    DataFrame with CANONICAL_COLUMNS + ["has_gyro", "age_group"]
    """
    cache_path = PROCESSED_PARQUET
    if cache and cache_path.exists():
        log.info("Loading cached harmonized dataset from %s", cache_path)
        return pd.read_parquet(cache_path)

    adapters = [
        UniMiBSHARAdapter(raw_root / "unimib_shar"),
        MobiActAdapter(raw_root / "mobiact"),
        SisFallAdapter(raw_root / "sisfall"),
        KFallAdapter(raw_root / "kfall"),
    ]
    if own_raw_dir is not None:
        adapters.append(OwnRecordingsAdapter(own_raw_dir))

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for adapter in adapters:
        if not adapter.is_available():
            skipped.append(adapter.dataset_id)
            log.warning(
                "[%s] Not available — skipping. "
                "Download from data/datasets.py → download_url.",
                adapter.dataset_id,
            )
            continue
        try:
            df = adapter.load()
            df = _validate_and_clean(df, adapter.dataset_id)
            frames.append(df)
            log.info("[%s] ✓  %d rows, %d subjects, %d recordings",
                     adapter.dataset_id, len(df),
                     df["subject_id"].nunique(), df["recording_id"].nunique())
        except Exception as exc:
            log.error("[%s] Failed to load: %s", adapter.dataset_id, exc)
            skipped.append(adapter.dataset_id)

    if not frames:
        raise RuntimeError(
            "No datasets were loaded. Either download at least one public dataset "
            "or run with --generate_demo_data to use synthetic data.\n"
            f"Skipped: {skipped}"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = _add_derived_columns(merged)
    merged = _drop_untrained_labels(merged)

    _log_diversity_summary(merged, skipped)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        log.info("Harmonized dataset cached → %s", cache_path)

    return merged


def invalidate_cache() -> None:
    if PROCESSED_PARQUET.exists():
        PROCESSED_PARQUET.unlink()
        log.info("Cache invalidated: %s", PROCESSED_PARQUET)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_and_clean(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    # Ensure canonical columns exist
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[CANONICAL_COLUMNS].copy()

    # Coerce sensor values
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(np.int64)

    # Remove rows where all acc values are NaN
    before = len(df)
    df = df.dropna(subset=["ax", "ay", "az"])
    if len(df) < before:
        log.debug("[%s] Dropped %d rows with NaN accelerometer", dataset_id, before - len(df))

    return df.reset_index(drop=True)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Mark rows that have valid gyroscope data
    df["has_gyro"] = (~df["gx"].isna()).astype(bool)
    # For datasets with no gyroscope, fill with 0.0 so features still work
    for col in ["gx", "gy", "gz"]:
        df[col] = df[col].fillna(0.0)

    # Age group from SisFall subject_id prefix (sisfall_older_XX / sisfall_young_XX)
    df["age_group"] = df["subject_id"].apply(_infer_age_group)
    return df


def _drop_untrained_labels(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["unified_label"].isin(TRAINABLE_LABELS)].copy()
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d rows with non-trainable labels (e.g. 'unknown').", dropped)
    return df.reset_index(drop=True)


def _infer_age_group(subject_id: str) -> str:
    if "older" in subject_id:
        return "older_adult"
    if "young" in subject_id or "se" in subject_id.lower():
        return "young_adult"
    if "own" in subject_id:
        return "unknown"
    return "adult"   # default


def _log_diversity_summary(df: pd.DataFrame, skipped: list[str]) -> None:
    log.info("\n" + "═" * 70)
    log.info("HARMONIZED DATASET DIVERSITY SUMMARY")
    log.info("═" * 70)
    log.info("  Total rows         : %d", len(df))
    log.info("  Unique subjects    : %d", df["subject_id"].nunique())
    log.info("  Unique recordings  : %d", df["recording_id"].nunique())
    log.info("  Datasets loaded    : %s", df["dataset_id"].unique().tolist())
    if skipped:
        log.info("  Datasets skipped   : %s", skipped)
    log.info("  Unified labels     : %s", sorted(df["unified_label"].unique()))
    log.info("  Phone positions    : %s", sorted(df["phone_position"].dropna().unique()))
    log.info("  Has gyroscope      : %d / %d rows", df["has_gyro"].sum(), len(df))
    log.info("  Age groups         : %s", df["age_group"].value_counts().to_dict())
    log.info("")
    log.info("  Label distribution:")
    for label, count in df["unified_label"].value_counts().items():
        subj_count = df[df["unified_label"] == label]["subject_id"].nunique()
        rec_count  = df[df["unified_label"] == label]["recording_id"].nunique()
        log.info("    %-14s : %7d rows | %3d subjects | %4d recordings",
                 label, count, subj_count, rec_count)
    log.info("═" * 70)
