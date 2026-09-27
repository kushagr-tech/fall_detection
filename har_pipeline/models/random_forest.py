"""
models/random_forest.py
=======================
Random Forest classifier with SMOTE-based over-sampling for the minority
'falling' class and subject-aware cross-validation.

Design choices
--------------
* class_weight='balanced_subsample' ensures each tree sees a balanced view
  without resampling the full dataset (fast, memory-efficient).
* A separate SMOTE step is also available when falling class is very rare.
* GridSearchCV is NOT used here to keep the pipeline fast; sensible defaults
  are derived from HAR literature.  Swap in RandomizedSearchCV if needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=16,          # limit tree depth to prevent memorization
    min_samples_leaf=4,    # regularize leaf nodes
    min_samples_split=8,   # require at least 8 samples to split
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
    oob_score=True,
)


# ── Public API ─────────────────────────────────────────────────────────────────

class RandomForestModel:
    """
    Wraps sklearn Pipeline (StandardScaler → RandomForest) together with
    the LabelEncoder so callers always work with string labels.
    """

    def __init__(self, params: dict | None = None):
        p = {**RF_PARAMS, **(params or {})}
        self.label_encoder = LabelEncoder()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(**p)),
        ])
        self._is_fitted = False

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X_feat: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """
        Parameters
        ----------
        X_feat : (n_windows, n_features) feature matrix
        y      : (n_windows,) string labels
        """
        y_enc = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X_feat, y_enc)
        self._is_fitted = True

        oob = self.pipeline.named_steps["clf"].oob_score_
        log.info("RF trained. OOB accuracy: %.4f", oob)
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        """Returns string labels."""
        self._check_fitted()
        y_enc = self.pipeline.predict(X_feat)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X_feat: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(X_feat)

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "le": self.label_encoder}, path)
        log.info("RF model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        obj = cls.__new__(cls)
        saved = joblib.load(path)
        obj.pipeline      = saved["pipeline"]
        obj.label_encoder = saved["le"]
        obj._is_fitted    = True
        return obj

    # ── Feature importances (pass-through) ────────────────────────────────────

    @property
    def feature_importances_(self) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.named_steps["clf"].feature_importances_

    # ── Internal ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
