"""
data/adapters/own_recordings.py
================================
Adapter for recordings collected with our own Activity Collector Android app.

These recordings are already in our canonical CSV format:
  timestamp, ax, ay, az, gx, gy, gz, label

The filenames must follow the naming convention used by the Android app:
  <subject_id>_<session_id>.csv   e.g.  subject01_session.csv
  or the richer naming from the extended recording UI:
  <subject_id>_<session_id>_<phone_position>_<phone_orientation>.csv

Phone position and orientation metadata are embedded in the filename if available,
otherwise they default to "unknown".

This adapter adds the dataset_id "own", subject_id "own_<stem>", etc.
so that own recordings can be included in the multi-dataset merge or held
out for Experiment C (train on public → test on own recordings).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from data.adapters.base import BaseAdapter
from data.datasets import LABEL_ALIASES, UNIFIED_LABELS

log = logging.getLogger(__name__)

DATASET_ID = "own"

# Labels accepted verbatim from our own app
_OWN_LABELS = set(UNIFIED_LABELS.keys()) | set(LABEL_ALIASES.keys())


class OwnRecordingsAdapter(BaseAdapter):

    dataset_id = DATASET_ID

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir)

    def is_available(self) -> bool:
        return self.raw_dir.exists() and any(self.raw_dir.glob("*.csv"))

    def load(self) -> pd.DataFrame:
        self._assert_available()
        files = sorted(self.raw_dir.glob("*.csv"))
        log.info("[%s] Loading %d CSV files from %s …", self.dataset_id, len(files), self.raw_dir)

        frames: list[pd.DataFrame] = []
        for f in files:
            try:
                df = self._load_single(f)
                frames.append(df)
            except Exception as exc:
                log.warning("[%s] Skipping %s: %s", self.dataset_id, f.name, exc)

        if not frames:
            raise RuntimeError(f"[{self.dataset_id}] No CSV files loaded from {self.raw_dir}")

        result = pd.concat(frames, ignore_index=True)
        log.info("[%s] Loaded %d rows", self.dataset_id, len(result))
        return result

    def _load_single(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip().str.lower()

        required = {"timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label"}
        missing  = required - set(raw.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path.name}")

        # Parse subject + session from filename
        parts = path.stem.split("_")
        subj_local = parts[0] if parts else "unknown"
        session    = parts[1] if len(parts) > 1 else "s01"
        position   = parts[2] if len(parts) > 2 else "unknown"
        orientation= parts[3] if len(parts) > 3 else "unknown"

        subj_id = f"own_{subj_local}"
        rec_id  = f"own_{subj_local}_{session}"

        raw["dataset_id"]     = self.dataset_id
        raw["subject_id"]     = subj_id
        raw["recording_id"]   = rec_id
        raw["original_label"] = raw["label"].astype(str).str.strip().str.lower()
        raw["unified_label"]  = raw["original_label"].apply(_normalise_own_label)
        raw["phone_position"] = position
        raw["segment"]        = 0

        # Ensure sensor columns are float32
        for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float32)

        return raw


def _normalise_own_label(label: str) -> str:
    if label in UNIFIED_LABELS:
        return label
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]
    return "unknown"
