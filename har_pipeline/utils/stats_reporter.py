"""
utils/stats_reporter.py
=======================
Generates the full suite of dataset statistics plots and CSV reports.

Plots produced
--------------
  class_distribution.png          — bar chart of rows per class
  subjects_per_dataset.png        — grouped bar per dataset
  sampling_rates.png              — known sampling rates from registry
  sensor_ranges.png               — box plots of acc/gyro ranges per dataset
  recording_duration_distribution.png — histogram of recording lengths

All plots are saved to outputs/plots/dataset_stats/.
All tables are saved to data/metadata/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from data.datasets import ALL_DATASETS

log = logging.getLogger(__name__)

PLOT_DIR     = Path(__file__).parent.parent / "outputs" / "plots" / "dataset_stats"
METADATA_DIR = Path(__file__).parent.parent / "data" / "metadata"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_all(df: pd.DataFrame) -> None:
    """
    Run all reporters. `df` is the harmonized DataFrame from data.harmonizer.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    _plot_class_distribution(df)
    _plot_subjects_per_dataset(df)
    _plot_sensor_ranges(df)
    _plot_recording_durations(df)
    _plot_phone_positions(df)
    _plot_known_sampling_rates()
    _save_label_mapping_csv()
    log.info("All dataset statistics reports generated.")


# ── Individual plots ───────────────────────────────────────────────────────────

def _plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df.groupby(["unified_label", "dataset_id"]).size().unstack(fill_value=0)
    counts = counts.sort_values(counts.columns[0] if len(counts.columns) else "unified_label",
                                ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Samples per Activity Class and Dataset", fontsize=13)
    ax.set_xlabel("Activity")
    ax.set_ylabel("Sample count")
    ax.set_xticklabels(counts.index, rotation=35, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    _save(fig, "class_distribution.png")


def _plot_subjects_per_dataset(df: pd.DataFrame) -> None:
    sub_df = (
        df.groupby(["dataset_id", "unified_label"])["subject_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sub_df.T.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Unique Subjects per Activity Class and Dataset", fontsize=13)
    ax.set_xlabel("Activity")
    ax.set_ylabel("Subject count")
    ax.set_xticklabels(sub_df.columns, rotation=35, ha="right")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    _save(fig, "subjects_per_dataset.png")


def _plot_sensor_ranges(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, channels, title in [
        (axes[0], ["ax", "ay", "az"], "Accelerometer (m/s²)"),
        (axes[1], ["gx", "gy", "gz"], "Gyroscope (rad/s)"),
    ]:
        data_by_ds = []
        labels_ds  = []
        for ds_id, grp in df.groupby("dataset_id"):
            vals = grp[channels].values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals):
                data_by_ds.append(vals)
                labels_ds.append(ds_id)

        ax.boxplot(data_by_ds, labels=labels_ds, showfliers=False, patch_artist=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Sensor Value Distributions per Dataset", fontsize=13)
    plt.tight_layout()
    _save(fig, "sensor_ranges.png")


def _plot_recording_durations(df: pd.DataFrame) -> None:
    durations = (
        df.groupby("recording_id")
        .size()
        .apply(lambda n: n / 50 / 60)   # rows → minutes at 50 Hz
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(durations, bins=40, color="#3b82f6", edgecolor="white", linewidth=0.5)
    ax.set_title("Recording Duration Distribution", fontsize=13)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of recordings")
    median = durations.median()
    ax.axvline(median, color="#ef4444", linestyle="--", label=f"Median {median:.1f} min")
    ax.legend()
    plt.tight_layout()
    _save(fig, "recording_duration_distribution.png")


def _plot_phone_positions(df: pd.DataFrame) -> None:
    pos_counts = (
        df.groupby("phone_position")
        .agg(n_rows=("ax", "count"), n_subjects=("subject_id", "nunique"))
        .reset_index()
        .sort_values("n_rows", ascending=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in [
        (axes[0], "n_rows",     "Samples per Phone Position"),
        (axes[1], "n_subjects", "Subjects per Phone Position"),
    ]:
        ax.barh(pos_counts["phone_position"], pos_counts[col], color="#8b5cf6")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(col)
    plt.tight_layout()
    _save(fig, "phone_position_distribution.png")


def _plot_known_sampling_rates() -> None:
    names = []
    rates = []
    for ds in ALL_DATASETS.values():
        names.append(ds.dataset_id)
        rates.append(ds.sensor_spec.sampling_rate_hz)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#22c55e" if r == 50 else "#f97316" for r in rates]
    ax.bar(names, rates, color=colors)
    ax.axhline(50, color="#1a1d23", linestyle="--", linewidth=1, label="Target 50 Hz")
    ax.set_title("Original Sampling Rates by Dataset\n(orange = requires resampling)", fontsize=11)
    ax.set_ylabel("Hz")
    ax.legend()
    plt.tight_layout()
    _save(fig, "sampling_rates.png")


def _save_label_mapping_csv() -> None:
    rows = []
    for ds in ALL_DATASETS.values():
        for orig, unified in ds.unified_label_map.items():
            description = ds.original_labels.get(orig, "")
            rows.append({
                "dataset_id":     ds.dataset_id,
                "original_label": orig,
                "description":    description,
                "unified_label":  unified,
            })
    path = METADATA_DIR / "label_mapping.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Label mapping CSV saved → %s", path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved → %s", path)
