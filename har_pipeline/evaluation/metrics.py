"""
evaluation/metrics.py
=====================
Classification metrics, confusion-matrix plot, and per-class report.

Special treatment for the 'falling' class
------------------------------------------
Fall detection is a safety-critical task.  A missed fall (false negative) is
far more dangerous than a false alarm.  We therefore:

  1. Always print fall-specific recall separately and flag if it drops
     below FALL_RECALL_THRESHOLD.
  2. Include a normalised confusion matrix so misclassification patterns
     are immediately visible.
  3. Return a ``results`` dict that the pipeline can compare across models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

log = logging.getLogger(__name__)

FALL_LABEL           = "falling"
FALL_RECALL_THRESHOLD = 0.90   # warn if fall recall drops below this


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    classes: np.ndarray | list[str],
    output_dir: str | Path | None = None,
) -> dict:
    """
    Compute and display all metrics.  Optionally saves plots.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1,
        fall_precision, fall_recall, fall_f1,
        per_class  (dict label → {precision, recall, f1, support})
    """
    classes = list(classes)
    output_dir = Path(output_dir) if output_dir else None

    acc   = float((y_true == y_pred).mean())
    mf1   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    wf1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = {
        lbl: dict(precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for lbl, p, r, f, s in zip(classes, prec_arr, rec_arr, f1_arr, sup_arr)
    }

    # ── Console report ─────────────────────────────────────────────────────────
    log.info("\n═══ %s Results ═══", model_name.upper())
    log.info("Accuracy : %.4f", acc)
    log.info("Macro F1 : %.4f", mf1)
    log.info(
        "\n%s",
        classification_report(y_true, y_pred, labels=classes, zero_division=0),
    )

    # ── Fall-specific warning ──────────────────────────────────────────────────
    fall_metrics = per_class.get(FALL_LABEL, {})
    fall_recall  = fall_metrics.get("recall", 0.0)
    fall_prec    = fall_metrics.get("precision", 0.0)
    fall_f1      = fall_metrics.get("f1", 0.0)

    if fall_recall < FALL_RECALL_THRESHOLD:
        log.warning(
            "⚠  %s: Fall recall = %.3f  (below target %.2f). "
            "Consider adjusting class weights or lowering fall threshold.",
            model_name, fall_recall, FALL_RECALL_THRESHOLD,
        )
    else:
        log.info(
            "✓  %s: Fall recall = %.3f  ≥ target %.2f",
            model_name, fall_recall, FALL_RECALL_THRESHOLD,
        )

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    cm_path = None
    if output_dir:
        cm_path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        _plot_confusion_matrix(y_true, y_pred, classes, model_name, cm_path)

    return dict(
        model_name       = model_name,
        accuracy         = acc,
        macro_f1         = mf1,
        weighted_f1      = wf1,
        fall_precision   = fall_prec,
        fall_recall      = fall_recall,
        fall_f1          = fall_f1,
        per_class        = per_class,
        confusion_matrix_path = str(cm_path) if cm_path else None,
    )


def print_comparison_table(results: list[dict]) -> None:
    """Pretty-print a side-by-side comparison of two or more model results."""
    header = f"{'Model':<22} {'Accuracy':>9} {'Macro F1':>9} {'Fall Rec':>9} {'Fall F1':>9}"
    sep    = "─" * len(header)
    log.info("\n%s\n%s\n%s", sep, header, sep)
    for r in results:
        log.info(
            "%-22s %9.4f %9.4f %9.4f %9.4f",
            r["model_name"],
            r["accuracy"],
            r["macro_f1"],
            r["fall_recall"],
            r["fall_f1"],
        )
    log.info(sep)


def save_report(results: list[dict], output_dir: str | Path) -> None:
    """Save JSON summary of all model results."""
    import json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"

    # per_class dicts are already JSON-serialisable
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Evaluation report saved → %s", report_path)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    model_name: str,
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Normalised Confusion Matrix — {model_name}", fontsize=13)

    # Highlight the 'falling' row in a different colour
    if FALL_LABEL in classes:
        fall_idx = classes.index(FALL_LABEL)
        for col_idx in range(len(classes)):
            ax.add_patch(
                plt.Rectangle(
                    (col_idx, fall_idx), 1, 1,
                    fill=False, edgecolor="red", lw=2, clip_on=False,
                )
            )

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", path)
