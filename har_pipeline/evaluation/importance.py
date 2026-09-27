"""
evaluation/importance.py
========================
Explains which features are most important for the Random Forest model and,
specifically, which are most discriminative for detecting falls.

Two analyses are performed:

1. Global MDI importance (Mean Decrease in Impurity, built into RF) — fast,
   shows overall feature utility across all classes.

2. Fall-vs-rest permutation importance — a binary RF is retrained on the
   same feature matrix with labels {falling=1, other=0}.  Permutation
   importance on this binary model pinpoints features that specifically
   distinguish falls from every other activity.

Both results are saved as bar charts and as CSV tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

FALL_LABEL = "falling"
TOP_N      = 20   # features to show in charts


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_importance(
    rf_model,                      # RandomForestModel instance (already fitted)
    X_feat:    np.ndarray,
    y:         np.ndarray,
    feat_names: list[str],
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Run both analyses and save plots + CSV tables.

    Parameters
    ----------
    rf_model   : fitted RandomForestModel
    X_feat     : (n_windows, n_features) feature matrix used to train the RF
    y          : (n_windows,) string labels
    feat_names : ordered list of feature names from features.extractor.feature_names()
    output_dir : directory for output files

    Returns
    -------
    DataFrame with columns [feature, mdi_importance, fall_perm_importance]
    sorted by fall_perm_importance descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_names = list(feat_names)

    # ── 1. Global MDI importance ───────────────────────────────────────────────
    mdi = rf_model.feature_importances_
    mdi_df = pd.DataFrame({"feature": feat_names, "mdi_importance": mdi})
    mdi_df = mdi_df.sort_values("mdi_importance", ascending=False).reset_index(drop=True)

    _plot_bar(
        mdi_df["feature"].head(TOP_N).tolist(),
        mdi_df["mdi_importance"].head(TOP_N).tolist(),
        title=f"Top {TOP_N} Features — Global MDI Importance (Random Forest)",
        xlabel="MDI Importance",
        path=output_dir / "feature_importance_global.png",
        color="#3b82f6",
    )
    log.info("Top 10 global features:\n%s", mdi_df.head(10).to_string(index=False))

    # ── 2. Fall-vs-rest permutation importance ─────────────────────────────────
    y_binary = (y == FALL_LABEL).astype(int)

    fall_rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )),
    ])
    fall_rf_pipe.fit(X_feat, y_binary)

    perm = permutation_importance(
        fall_rf_pipe, X_feat, y_binary,
        n_repeats=10, random_state=42, scoring="f1",
        n_jobs=-1,
    )
    perm_mean = perm.importances_mean
    perm_df = pd.DataFrame({
        "feature":              feat_names,
        "fall_perm_importance": perm_mean,
    }).sort_values("fall_perm_importance", ascending=False).reset_index(drop=True)

    _plot_bar(
        perm_df["feature"].head(TOP_N).tolist(),
        perm_df["fall_perm_importance"].head(TOP_N).tolist(),
        title=f"Top {TOP_N} Features for FALL Detection (Permutation Importance)",
        xlabel="Mean F1 drop when feature is permuted",
        path=output_dir / "feature_importance_fall.png",
        color="#ef4444",
    )
    log.info(
        "Top 10 fall-discriminative features:\n%s",
        perm_df.head(10).to_string(index=False),
    )

    # ── Merge and save ─────────────────────────────────────────────────────────
    combined = perm_df.merge(mdi_df, on="feature")
    combined = combined.sort_values("fall_perm_importance", ascending=False)
    combined.to_csv(output_dir / "feature_importance.csv", index=False)
    log.info("Feature importance table saved → %s", output_dir / "feature_importance.csv")

    _print_narrative(combined)
    return combined


# ── Helpers ────────────────────────────────────────────────────────────────────

def _plot_bar(
    features: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(features) + 1.5))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values[::-1], color=color, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[::-1], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Importance chart saved → %s", path)


def _print_narrative(df: pd.DataFrame) -> None:
    """Log a plain-English explanation of the top fall features."""
    top5 = df["feature"].head(5).tolist()

    groups = {
        "SVM / jerk": ["acc_svm", "gyr_svm", "acc_jerk", "gyr_jerk", "resultant"],
        "Spectral":   ["band_high", "band_mid", "dom_freq", "spec_centroid", "spec_entropy"],
        "Statistical":["az_kurt", "az_skew", "acc_sma", "az_std", "resultant_peak"],
    }

    mentioned = []
    for feat in top5:
        for group, keywords in groups.items():
            if any(k in feat for k in keywords):
                mentioned.append((feat, group))
                break

    log.info(
        "\n──────────────────────────────────────────────────────\n"
        "FEATURE IMPORTANCE NARRATIVE (FALL DETECTION)\n"
        "──────────────────────────────────────────────────────\n"
        "The most important features for distinguishing falls from\n"
        "normal activities are:\n\n"
        "  %s\n\n"
        "Key patterns typically seen:\n"
        "  • SVM (Signal Vector Magnitude) peaks and jerk values\n"
        "    capture the sudden acceleration spike that occurs when\n"
        "    a person hits the ground or loses balance.\n"
        "  • High-frequency spectral power (band_high) reflects the\n"
        "    impulsive, broadband energy of an impact event — absent\n"
        "    in normal locomotion.\n"
        "  • Kurtosis and skewness of the vertical axis (az) are\n"
        "    high during falls because the signal distribution becomes\n"
        "    impulsive (heavy-tailed) rather than Gaussian.\n"
        "  • Gyroscope SVM peak is large during uncontrolled rotation\n"
        "    when a person falls sideways or forward.\n"
        "──────────────────────────────────────────────────────",
        "\n  ".join(f"{f}  [{g}]" for f, g in mentioned) if mentioned else
        "\n  ".join(top5),
    )
