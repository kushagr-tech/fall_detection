"""
data/balance.py
===============
Dataset diversity reporting and class-weight computation.

Reports generated
-----------------
  dataset_summary.csv        — one row per dataset: subjects, recordings, rows, duration
  class_distribution.csv     — rows per (label, dataset, phone_position)
  subject_distribution.csv   — rows per (subject_id, label)
  sensor_statistics.csv      — mean/std/min/max per sensor axis per dataset
  phone_position_distribution.csv — rows and subjects per position

Class-weight strategy
---------------------
Simple inverse-frequency weights with a floor to prevent extreme values.
These weights are used for:
  - RandomForest class_weight parameter
  - Focal loss alpha in the neural network

We do NOT balance by oversampling identical windows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, floor: float = 0.1) -> dict[str, float]:
    """
    Return a dict {label: weight} using inverse-class-frequency, floored at
    `floor` × max_weight to prevent the minority class from dominating entirely.

    Parameters
    ----------
    y     : array of string labels
    floor : minimum weight relative to the maximum (0–1)

    Returns
    -------
    dict mapping each unique label to its weight
    """
    labels, counts = np.unique(y, return_counts=True)
    freqs   = counts / counts.sum()
    weights = 1.0 / freqs
    weights = weights / weights.sum() * len(labels)    # normalise so mean = 1

    max_w = weights.max()
    weights = np.maximum(weights, floor * max_w)

    result = {str(lbl): float(w) for lbl, w in zip(labels, weights)}
    log.info("Class weights: %s", {k: f"{v:.3f}" for k, v in result.items()})
    return result


def class_weight_array(y: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    Return a float array of shape (n_classes,) in the order given by `classes`.
    Suitable for use as torch tensor alpha in focal loss.
    """
    weight_dict = compute_class_weights(y)
    return np.array([weight_dict.get(c, 1.0) for c in classes], dtype=np.float32)


def generate_reports(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Generate all diversity reports and save them as CSV files.

    Parameters
    ----------
    df         : harmonized OR synthetic DataFrame
    output_dir : directory where CSV files are written

    Returns
    -------
    dict of report_name → DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure required columns exist for synthetic DataFrames
    if "dataset_id" not in df.columns:
        df = df.copy()
        df["dataset_id"] = "synthetic"
    if "subject_id" not in df.columns:
        df = df.copy()
        df["subject_id"] = df.get("subject", "unknown")
    if "recording_id" not in df.columns:
        df = df.copy()
        df["recording_id"] = df["subject_id"].astype(str) + "_s0"
    if "unified_label" not in df.columns:
        df = df.copy()
        df["unified_label"] = df.get("label", "unknown")
    if "phone_position" not in df.columns:
        df = df.copy()
        df["phone_position"] = "unknown"
    if "has_gyro" not in df.columns:
        df = df.copy()
        df["has_gyro"] = (~df["gx"].isna()) if "gx" in df.columns else True

    reports: dict[str, pd.DataFrame] = {}

    reports["dataset_summary"]             = _dataset_summary(df)
    reports["class_distribution"]          = _class_distribution(df)
    reports["subject_distribution"]        = _subject_distribution(df)
    reports["sensor_statistics"]           = _sensor_statistics(df)
    reports["phone_position_distribution"] = _position_distribution(df)

    for name, table in reports.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        log.info("Report saved → %s  (%d rows)", path, len(table))

    _log_balance_overview(df)
    return reports


# ── Report builders ────────────────────────────────────────────────────────────

def _dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds_id, grp in df.groupby("dataset_id"):
        n_rows     = len(grp)
        n_subj     = grp["subject_id"].nunique()
        n_recs     = grp["recording_id"].nunique()
        n_labels   = grp["unified_label"].nunique()
        dur_min    = n_rows / 50 / 60    # assumes 50 Hz
        has_gyro_pct = grp["has_gyro"].mean() * 100 if "has_gyro" in grp else 0
        rows.append({
            "dataset_id":         ds_id,
            "n_rows":             n_rows,
            "n_subjects":         n_subj,
            "n_recordings":       n_recs,
            "n_unique_labels":    n_labels,
            "est_duration_min":   round(dur_min, 1),
            "has_gyro_pct":       round(has_gyro_pct, 1),
        })
    return pd.DataFrame(rows).sort_values("dataset_id")


def _class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["unified_label", "dataset_id", "phone_position"])
        .agg(
            n_rows      = ("ax", "count"),
            n_subjects  = ("subject_id", "nunique"),
            n_recordings= ("recording_id", "nunique"),
        )
        .reset_index()
        .sort_values(["unified_label", "dataset_id"])
    )


def _subject_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["subject_id", "unified_label"])
        .agg(n_rows=("ax", "count"))
        .reset_index()
        .sort_values(["subject_id", "unified_label"])
    )


def _sensor_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds_id, grp in df.groupby("dataset_id"):
        for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
            vals = grp[col].dropna()
            rows.append({
                "dataset_id": ds_id,
                "channel":    col,
                "mean":       round(float(vals.mean()), 4),
                "std":        round(float(vals.std()), 4),
                "min":        round(float(vals.min()), 4),
                "max":        round(float(vals.max()), 4),
                "pct5":       round(float(vals.quantile(0.05)), 4),
                "pct95":      round(float(vals.quantile(0.95)), 4),
            })
    return pd.DataFrame(rows)


def _position_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("phone_position")
        .agg(
            n_rows      = ("ax", "count"),
            n_subjects  = ("subject_id", "nunique"),
            n_recordings= ("recording_id", "nunique"),
            n_datasets  = ("dataset_id", "nunique"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )


def _log_balance_overview(df: pd.DataFrame) -> None:
    log.info("\n" + "─" * 60)
    log.info("CLASS BALANCE OVERVIEW")
    log.info("─" * 60)
    total = len(df)
    for label, cnt in df["unified_label"].value_counts().items():
        bar = "█" * int(cnt / total * 40)
        log.info("  %-14s %7d  (%5.1f%%)  %s", label, cnt, cnt / total * 100, bar)
    log.info("─" * 60)
