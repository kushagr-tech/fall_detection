"""
data/adapters/base.py
=====================
Abstract base class for all dataset adapters.

Each adapter is responsible for:
  1. Verifying the raw dataset is present on disk
  2. Loading the raw files into a common intermediate format
  3. Returning a tidy DataFrame with exactly these columns:

      dataset_id  | str  — source dataset identifier
      subject_id  | str  — globally unique: "<dataset_id>_<local_id>"
      recording_id| str  — globally unique: "<dataset_id>_<local_id>_<session>"
      timestamp   | int  — Unix time in milliseconds (or synthetic if unavailable)
      ax ay az    | f32  — m/s²
      gx gy gz    | f32  — rad/s  (NaN if dataset has no gyroscope)
      original_label | str — the label code from the source dataset
      unified_label  | str — mapped to UNIFIED_LABELS
      phone_position | str — from PhonePosition names
      segment     | int  — intra-recording segment index (from loader)

All unit conversions, resampling, and axis remapping are done inside the adapter
so that harmonizer.py sees a clean, consistent schema.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


CANONICAL_COLUMNS = [
    "dataset_id", "subject_id", "recording_id",
    "timestamp",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "original_label", "unified_label",
    "phone_position", "segment",
]


class BaseAdapter(ABC):

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    @property
    @abstractmethod
    def dataset_id(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the raw dataset files are present on disk."""
        ...

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load all recordings, apply unit conversion + axis remap,
        and return a DataFrame with CANONICAL_COLUMNS.
        """
        ...

    def _assert_available(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(
                f"[{self.dataset_id}] Raw data not found in {self.raw_dir}. "
                f"Please download the dataset and place it there.\n"
                f"See har_pipeline/data/datasets.py for the download URL."
            )

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add any missing canonical columns with sensible defaults."""
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[CANONICAL_COLUMNS].copy()
