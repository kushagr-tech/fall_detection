"""
evaluation/cross_dataset.py
===========================
Runs and reports the three generalisation experiments.

Experiment A — Within-dataset subject split
  Train: subjects from dataset(s) assigned to train split
  Test:  held-out subjects from the same dataset(s)
  Purpose: baseline — how well does the model generalise to NEW PEOPLE
           from data it was trained on?

Experiment B — Leave-one-dataset-out
  Train: all subjects from every dataset EXCEPT one
  Test:  ALL subjects from the held-out dataset
  Purpose: does the model generalise across different:
           • device placements (pocket vs. waist)
           • recording protocols
           • subject demographics

Experiment C — Train on public, test on own recordings
  Train: all subjects from public datasets
  Test:  subjects from our own Android recordings
  Purpose: does the model generalise from lab/controlled data
           to real-world, naturalistic behaviour?

Results are logged and saved to outputs/reports/cross_dataset_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.subject_split import get_splits
from features.windowing  import make_windows
from features.extractor  import extract_features, feature_names
from models.random_forest import RandomForestModel
from models.neural_net    import NeuralNetModel
from evaluation.metrics   import evaluate, print_comparison_table

log = logging.getLogger(__name__)

RESULTS_FILE = Path(__file__).parent.parent / "outputs" / "reports" / "cross_dataset_results.json"


# ── Public API ─────────────────────────────────────────────────────────────────

def run_all_experiments(
    df:              pd.DataFrame,
    window_size:     int   = 100,
    step:            int   = 50,
    nn_epochs:       int   = 40,
    skip_nn:         bool  = False,
    test_dataset_id: str   = "kfall",   # for Experiment B
    output_dir:      Path  = Path("outputs/plots"),
) -> dict[str, list[dict]]:
    """
    Run all three cross-dataset generalisation experiments.

    Returns
    -------
    dict with keys "A", "B", "C" → list of result dicts per model
    """
    all_results: dict[str, list[dict]] = {}

    experiments = [
        ("A", None),
        ("B", test_dataset_id),
    ]

    # Only run Experiment C if own recordings are present
    if "own" in df["dataset_id"].unique():
        experiments.append(("C", None))
    else:
        log.info("Skipping Experiment C — no own recordings found (dataset_id='own').")

    for exp_id, test_ds in experiments:
        log.info("\n" + "═" * 65)
        log.info("EXPERIMENT %s", exp_id)
        log.info("═" * 65)
        try:
            results = _run_experiment(
                df, exp_id, test_ds, window_size, step,
                nn_epochs, skip_nn, output_dir,
            )
            all_results[exp_id] = results
        except Exception as exc:
            log.error("Experiment %s failed: %s", exp_id, exc)
            all_results[exp_id] = [{"error": str(exc)}]

    _save_results(all_results)
    _print_summary(all_results)
    return all_results


# ── Internal ───────────────────────────────────────────────────────────────────

def _run_experiment(
    df:              pd.DataFrame,
    experiment_id:   str,
    test_dataset_id: str | None,
    window_size:     int,
    step:            int,
    nn_epochs:       int,
    skip_nn:         bool,
    output_dir:      Path,
) -> list[dict]:
    train_df, val_df, test_df = get_splits(
        df, experiment_id=experiment_id, test_dataset_id=test_dataset_id
    )

    # Convert DataFrames to the format make_windows expects:
    # add 'subject' and 'segment' columns if not present
    train_df = _prep_for_windowing(train_df)
    test_df  = _prep_for_windowing(test_df)
    val_df   = _prep_for_windowing(val_df)

    # Only window on 6 core sensor columns + required meta
    log.info("  Windowing train (%d rows) …", len(train_df))
    X_tr, y_tr, g_tr = make_windows(train_df, window_size, step)

    log.info("  Windowing test  (%d rows) …", len(test_df))
    X_te, y_te, _    = make_windows(test_df,  window_size, step)

    log.info("  Windowing val   (%d rows) …", len(val_df))
    X_va, y_va, _    = make_windows(val_df,   window_size, step)

    all_classes = sorted(set(np.unique(y_tr)) | set(np.unique(y_te)))
    log.info("  Classes: %s", all_classes)

    # Feature extraction for RF
    X_tr_feat = extract_features(X_tr)
    X_te_feat = extract_features(X_te)

    results: list[dict] = []

    # ── Random Forest ──────────────────────────────────────────────────────────
    log.info("  Training RF …")
    rf = RandomForestModel()
    rf.fit(X_tr_feat, y_tr)
    y_pred_rf = rf.predict(X_te_feat)

    res_rf = evaluate(
        y_te, y_pred_rf,
        model_name = f"RF Exp-{experiment_id}",
        classes    = all_classes,
        output_dir = output_dir,
    )
    res_rf["experiment"] = experiment_id
    results.append(res_rf)

    # ── Neural Network ─────────────────────────────────────────────────────────
    if not skip_nn:
        log.info("  Training NN (max %d epochs) …", nn_epochs)
        nn = NeuralNetModel(hparams={"epochs": nn_epochs})
        nn.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        y_pred_nn = nn.predict(X_te)

        res_nn = evaluate(
            y_te, y_pred_nn,
            model_name = f"NN Exp-{experiment_id}",
            classes    = all_classes,
            output_dir = output_dir,
        )
        res_nn["experiment"] = experiment_id
        results.append(res_nn)

    print_comparison_table(results)
    return results


def _prep_for_windowing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'subject' and 'segment' columns in the format expected by make_windows(),
    and ensure sensor columns match data/loader.SENSOR_COLS order.
    """
    out = df.copy()
    if "subject" not in out.columns:
        out["subject"] = out["subject_id"]
    if "segment" not in out.columns:
        out["segment"] = 0
    if "label" not in out.columns:
        out["label"] = out["unified_label"]
    # make_windows uses 'timestamp' — ensure it exists
    if "timestamp" not in out.columns:
        out["timestamp"] = np.arange(len(out), dtype=np.int64) * 20
    return out


def _save_results(results: dict[str, list[dict]]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Cross-dataset results saved → %s", RESULTS_FILE)


def _print_summary(results: dict[str, list[dict]]) -> None:
    log.info("\n" + "═" * 80)
    log.info("CROSS-DATASET GENERALISATION SUMMARY")
    log.info("═" * 80)
    header = f"{'Experiment':<8} {'Model':<20} {'Accuracy':>9} {'MacroF1':>9} {'FallRec':>9}"
    log.info(header)
    log.info("─" * 80)
    for exp_id, exp_results in results.items():
        for r in exp_results:
            if "error" in r:
                log.info("  Exp %s: ERROR — %s", exp_id, r["error"])
                continue
            log.info(
                "  %-6s  %-20s  %9.4f  %9.4f  %9.4f",
                exp_id, r.get("model_name", "?"),
                r.get("accuracy", 0), r.get("macro_f1", 0), r.get("fall_recall", 0),
            )
    log.info("═" * 80)
