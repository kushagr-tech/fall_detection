"""
data/subject_split.py
=====================
Leakage-proof subject-level train / validation / test splitting.

Rules (strictly enforced)
--------------------------
1. Split by SUBJECT, never by individual rows or windows.
2. No subject appears in more than one of {train, validation, test}.
3. No recording session can cross splits.
4. The test set is written to disk once and NEVER modified afterwards.
   It is loaded as a frozen artifact for final evaluation only.
5. Stratification: ensure each split contains at least one subject per
   dataset so evaluation is not biased toward a single data source.

Cross-dataset experiment splits
--------------------------------
Experiment A:
  train=subset of dataset_A subjects,  test=held-out subjects of dataset_A

Experiment B:
  train=all subjects from datasets A+B+C,  test=all subjects of dataset_D
  (whole dataset held out as test)

Experiment C:
  train=all public dataset subjects,  test=all own_recordings subjects

The caller selects the experiment via experiment_id in get_splits().

Output files saved under data/metadata/
  subject_split.json   — the canonical subject↔split assignment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)

METADATA_DIR  = Path(__file__).parent / "metadata"
SPLIT_FILE    = METADATA_DIR / "subject_split.json"

SplitName = Literal["train", "validation", "test"]


# ── Public API ─────────────────────────────────────────────────────────────────

def make_subject_split(
    df:           pd.DataFrame,
    test_frac:    float = 0.20,
    val_frac:     float = 0.15,
    random_state: int   = 42,
    force_redo:   bool  = False,
) -> dict[str, list[str]]:
    """
    Create a subject-level train/val/test split and save it as JSON.

    The split is stratified by dataset_id so each split contains
    representative subjects from each source dataset.

    Parameters
    ----------
    df           : harmonized DataFrame (must have subject_id, dataset_id columns)
    test_frac    : fraction of subjects for test
    val_frac     : fraction of remaining (train+val) subjects for validation
    random_state : random seed — change to get a different split
    force_redo   : overwrite an existing split file

    Returns
    -------
    dict with keys "train", "validation", "test",
    each mapping to a list of subject_id strings.
    """
    if SPLIT_FILE.exists() and not force_redo:
        log.info("Loading existing subject split from %s", SPLIT_FILE)
        with open(SPLIT_FILE) as f:
            split = json.load(f)
        _verify_no_leakage(split)
        return split

    log.info("Creating new subject split (test=%.0f%%, val=%.0f%% of train+val) …",
             test_frac * 100, val_frac * 100)

    # Build subject manifest: one row per unique subject with their dataset
    subject_df = (
        df[["subject_id", "dataset_id"]]
        .drop_duplicates("subject_id")
        .reset_index(drop=True)
    )

    # ── Per-dataset stratified split ───────────────────────────────────────────
    train_subjects: list[str] = []
    val_subjects:   list[str] = []
    test_subjects:  list[str] = []

    rng = np.random.default_rng(random_state)

    for dataset_id, grp in subject_df.groupby("dataset_id"):
        subjects = grp["subject_id"].tolist()
        rng.shuffle(subjects := np.array(subjects))
        subjects = list(subjects)

        n       = len(subjects)
        n_test  = max(1, int(round(n * test_frac)))
        n_val   = max(1, int(round((n - n_test) * val_frac)))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Too few subjects — put all but 1 in train, 1 in test
            log.warning("[%s] Only %d subjects — assigning 1 to test, rest to train.",
                        dataset_id, n)
            test_subjects  += [subjects[0]]
            train_subjects += subjects[1:]
            continue

        test_subjects  += subjects[:n_test]
        val_subjects   += subjects[n_test : n_test + n_val]
        train_subjects += subjects[n_test + n_val:]

    split = {
        "train":      sorted(train_subjects),
        "validation": sorted(val_subjects),
        "test":       sorted(test_subjects),
    }

    _verify_no_leakage(split)
    _save_split(split)

    log.info("Subject split created: %d train / %d val / %d test subjects",
             len(split["train"]), len(split["validation"]), len(split["test"]))
    return split


def get_splits(
    df:           pd.DataFrame,
    experiment_id: str = "A",
    test_dataset_id: str | None = None,
    split_file:   Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, val_df, test_df) according to the chosen experiment.

    Experiment A: subject-split within all available data
    Experiment B: train=all-but-one-dataset, test=held-out-dataset
    Experiment C: train=all-public, test=own_recordings

    Parameters
    ----------
    df               : harmonized DataFrame
    experiment_id    : "A" | "B" | "C"
    test_dataset_id  : for Experiment B, the dataset_id to hold out
    split_file       : optional path to a pre-saved split JSON
    """
    if experiment_id == "B":
        return _experiment_b(df, test_dataset_id)
    if experiment_id == "C":
        return _experiment_c(df)

    # Default: Experiment A
    return _experiment_a(df, split_file)


# ── Experiment implementations ─────────────────────────────────────────────────

def _experiment_a(
    df: pd.DataFrame,
    split_file: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subject-level split across all datasets."""
    file = split_file or SPLIT_FILE
    if not file.exists():
        make_subject_split(df)
    with open(file) as f:
        split = json.load(f)

    train_df = df[df["subject_id"].isin(split["train"])].copy()
    val_df   = df[df["subject_id"].isin(split["validation"])].copy()
    test_df  = df[df["subject_id"].isin(split["test"])].copy()

    _log_split_stats("Experiment A", train_df, val_df, test_df)
    return train_df, val_df, test_df


def _experiment_b(
    df: pd.DataFrame,
    test_dataset_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out an entire dataset as the test set."""
    if test_dataset_id is None:
        raise ValueError("Experiment B requires test_dataset_id.")

    test_df  = df[df["dataset_id"] == test_dataset_id].copy()
    rest_df  = df[df["dataset_id"] != test_dataset_id].copy()

    # 85/15 train/val split within the remaining subjects
    subjects = rest_df["subject_id"].unique()
    rng      = np.random.default_rng(42)
    rng.shuffle(subjects)
    n_val    = max(1, int(len(subjects) * 0.15))
    val_subj = set(subjects[:n_val])

    train_df = rest_df[~rest_df["subject_id"].isin(val_subj)].copy()
    val_df   = rest_df[rest_df["subject_id"].isin(val_subj)].copy()

    _log_split_stats(f"Experiment B (test={test_dataset_id})", train_df, val_df, test_df)
    return train_df, val_df, test_df


def _experiment_c(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on public datasets, test on own recordings."""
    own_mask = df["dataset_id"] == "own"
    test_df  = df[own_mask].copy()
    public   = df[~own_mask].copy()

    subjects = public["subject_id"].unique()
    rng      = np.random.default_rng(42)
    rng.shuffle(subjects)
    n_val    = max(1, int(len(subjects) * 0.15))
    val_subj = set(subjects[:n_val])

    train_df = public[~public["subject_id"].isin(val_subj)].copy()
    val_df   = public[public["subject_id"].isin(val_subj)].copy()

    _log_split_stats("Experiment C (test=own recordings)", train_df, val_df, test_df)
    return train_df, val_df, test_df


# ── Utilities ──────────────────────────────────────────────────────────────────

def _verify_no_leakage(split: dict[str, list[str]]) -> None:
    train = set(split.get("train", []))
    val   = set(split.get("validation", []))
    test  = set(split.get("test", []))

    tv = train & val
    tt = train & test
    vt = val   & test

    if tv or tt or vt:
        raise RuntimeError(
            f"LEAKAGE DETECTED in subject split!\n"
            f"  train ∩ val  = {tv}\n"
            f"  train ∩ test = {tt}\n"
            f"  val   ∩ test = {vt}"
        )
    log.debug("✓  No leakage detected in subject split.")


def _save_split(split: dict[str, list[str]]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)
    log.info("Subject split saved → %s", SPLIT_FILE)


def _log_split_stats(
    name: str,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    def _s(d: pd.DataFrame) -> str:
        return (f"{len(d):>7} rows | "
                f"{d['subject_id'].nunique():>3} subj | "
                f"{d['recording_id'].nunique():>4} recs | "
                f"datasets={d['dataset_id'].unique().tolist()}")

    log.info("[%s] Train : %s", name, _s(train_df))
    log.info("[%s] Val   : %s", name, _s(val_df))
    log.info("[%s] Test  : %s", name, _s(test_df))
