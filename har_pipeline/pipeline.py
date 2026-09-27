"""
pipeline.py
===========
Main entry point. Orchestrates the full HAR / fall-detection pipeline.

Two data modes
--------------
1. SYNTHETIC (default / --generate_demo_data)
   Generates physically-plausible CSVs and runs a quick train/eval cycle.
   Good for smoke-testing and CI.

2. MULTI-DATASET (--multi_dataset)
   Loads and harmonizes all available public datasets (UniMiB-SHAR, MobiAct,
   SisFall, KFall) plus any own Android recordings found in data/raw/own/.
   Runs three cross-dataset generalisation experiments (A, B, C).
   Requires that raw dataset files have been downloaded; see data/datasets.py.

Steps
-----
  1. Data loading & harmonization
  2. Diversity reporting (dataset_summary.csv, class_distribution.csv …)
  3. Subject-aware splits (no leakage across train/val/test)
  4. Sliding-window segmentation
  5. Feature extraction (RF path)
  6. Subject-grouped cross-validation
  7. Final holdout evaluation
  8. Cross-dataset generalisation experiments (multi-dataset mode only)
  9. Model saving + TorchScript export
  10. Feature importance analysis

Usage
-----
  # Synthetic demo (works with no downloads):
  python pipeline.py --generate_demo_data

  # Multi-dataset (after downloading public data):
  python pipeline.py --multi_dataset --raw_root data/raw

  # Full multi-dataset with cross-experiment, skip NN:
  python pipeline.py --multi_dataset --raw_root data/raw --skip_nn

  # Invalidate the parquet cache and re-harmonize:
  python pipeline.py --multi_dataset --invalidate_cache
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.loader          import load_dataset
from features.windowing   import make_windows
from features.extractor   import extract_features, feature_names
from models.random_forest import RandomForestModel
from models.neural_net    import NeuralNetModel
from evaluation.metrics   import evaluate, print_comparison_table, save_report
from evaluation.importance import analyze_importance
from utils.data_generator  import generate_demo_data

# Multi-dataset imports (only used when --multi_dataset is set)
from data.harmonizer   import harmonize, invalidate_cache
from data.subject_split import make_subject_split, get_splits
from data.balance      import generate_reports as generate_balance_reports
from evaluation.cross_dataset import run_all_experiments
from utils.stats_reporter     import generate_all as generate_stats_plots

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT               = Path(__file__).parent
OUTPUT_DIR         = ROOT / "outputs"
MODEL_DIR          = OUTPUT_DIR / "models"
PLOT_DIR           = OUTPUT_DIR / "plots"
REPORT_DIR         = OUTPUT_DIR / "reports"
METADATA_DIR       = ROOT / "data" / "metadata"
ANDROID_ASSETS_DIR = ROOT.parent / "activity_collector" / "app" / "src" / "main" / "assets"


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HAR + Fall Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data source ────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",           type=str,  default="data/raw",
                   help="[Synthetic mode] Folder with synthetic CSV files.")
    p.add_argument("--multi_dataset",      action="store_true",
                   help="Use harmonized public datasets instead of synthetic CSVs.")
    p.add_argument("--raw_root",           type=str,  default="data/raw",
                   help="[Multi-dataset] Parent directory containing dataset subdirs.")
    p.add_argument("--own_dir",            type=str,  default=None,
                   help="[Multi-dataset] Directory with own Android CSV recordings.")
    p.add_argument("--invalidate_cache",   action="store_true",
                   help="Delete the parquet cache and re-harmonize from raw files.")
    # ── Demo data ──────────────────────────────────────────────────────────────
    p.add_argument("--generate_demo_data", action="store_true",
                   help="Generate synthetic CSV data.")
    p.add_argument("--n_subjects",         type=int,  default=16)
    p.add_argument("--duration_s",         type=int,  default=180)
    # ── Windowing ──────────────────────────────────────────────────────────────
    p.add_argument("--window_size",        type=int,  default=100)
    p.add_argument("--step",               type=int,  default=50)
    # ── Training ───────────────────────────────────────────────────────────────
    p.add_argument("--test_size",          type=float, default=0.20)
    p.add_argument("--val_size",           type=float, default=0.15)
    p.add_argument("--cv_folds",           type=int,  default=5)
    p.add_argument("--skip_cv",            action="store_true")
    p.add_argument("--nn_epochs",          type=int,  default=40)
    p.add_argument("--skip_nn",            action="store_true")
    # ── Cross-dataset experiments ──────────────────────────────────────────────
    p.add_argument("--run_experiments",    action="store_true",
                   help="Run cross-dataset generalisation experiments A, B, C.")
    p.add_argument("--exp_b_test_dataset", type=str,  default="kfall",
                   help="Dataset held out as the Experiment B test set.")
    return p.parse_args()


# ── Cross-Validation ───────────────────────────────────────────────────────────

def run_cross_validation(
    X_raw:       np.ndarray,
    X_feat:      np.ndarray,
    y:           np.ndarray,
    groups:      np.ndarray,
    all_classes: list[str],
    n_splits:    int  = 5,
    skip_nn:     bool = False,
    nn_epochs:   int  = 40,
) -> dict[str, dict[str, float]]:
    import json
    n_unique_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_unique_groups)

    log.info("\n════════════════════════════════════════════════════════════════════════")
    log.info("Subject-Grouped %d-Fold Cross-Validation  (%d subjects)",
             n_splits, n_unique_groups)
    log.info("════════════════════════════════════════════════════════════════════════")

    gkf = GroupKFold(n_splits=n_splits)
    rf_fold_val: list[dict] = []
    rf_fold_train: list[dict] = []
    nn_fold_val: list[dict] = []
    nn_fold_train: list[dict] = []

    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups), 1):
        val_subjects   = np.unique(groups[val_idx])
        train_subjects = np.unique(groups[train_idx])
        log.info("Fold %d/%d — train=%d subjects (%d windows), val=%d subjects (%d windows) [%s]",
                 fold_idx, n_splits, len(train_subjects), len(train_idx),
                 len(val_subjects), len(val_idx), ", ".join(val_subjects[:4]))

        # ── Random Forest ──
        rf = RandomForestModel()
        rf.fit(X_feat[train_idx], y[train_idx])

        rf_tr_eval = evaluate(
            y[train_idx], rf.predict(X_feat[train_idx]),
            model_name=f"RF Train (Fold {fold_idx})", classes=all_classes,
        )
        rf_va_eval = evaluate(
            y[val_idx], rf.predict(X_feat[val_idx]),
            model_name=f"RF Val (Fold {fold_idx})", classes=all_classes,
        )
        rf_fold_train.append(rf_tr_eval)
        rf_fold_val.append(rf_va_eval)
        rf_gap = rf_tr_eval["accuracy"] - rf_va_eval["accuracy"]

        nn_tr_eval, nn_va_eval = None, None
        nn_gap = 0.0

        if not skip_nn:
            gss_sub = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_idx)
            st, sv  = next(gss_sub.split(train_idx, y[train_idx], groups[train_idx]))
            nn      = NeuralNetModel(hparams={"epochs": nn_epochs})
            nn.fit(X_raw[train_idx[st]], y[train_idx[st]],
                   X_val=X_raw[train_idx[sv]], y_val=y[train_idx[sv]])

            nn_tr_eval = evaluate(
                y[train_idx[st]], nn.predict(X_raw[train_idx[st]]),
                model_name=f"NN Train (Fold {fold_idx})", classes=all_classes,
            )
            nn_va_eval = evaluate(
                y[val_idx], nn.predict(X_raw[val_idx]),
                model_name=f"NN Val (Fold {fold_idx})", classes=all_classes,
            )
            nn_fold_train.append(nn_tr_eval)
            nn_fold_val.append(nn_va_eval)
            nn_gap = nn_tr_eval["accuracy"] - nn_va_eval["accuracy"]

        fold_details.append({
            "fold": fold_idx,
            "rf_train_acc": rf_tr_eval["accuracy"],
            "rf_val_acc":   rf_va_eval["accuracy"],
            "rf_gap":       rf_gap,
            "nn_train_acc": nn_tr_eval["accuracy"] if nn_tr_eval else None,
            "nn_val_acc":   nn_va_eval["accuracy"] if nn_va_eval else None,
            "nn_gap":       nn_gap if nn_tr_eval else None,
        })

    summary: dict[str, dict[str, float]] = {}

    def _summarize(name: str, train_res: list[dict], val_res: list[dict]) -> None:
        tr_accs = [r["accuracy"]    for r in train_res]
        va_accs = [r["accuracy"]    for r in val_res]
        gaps    = [t - v for t, v in zip(tr_accs, va_accs)]
        mf1s    = [r["macro_f1"]    for r in val_res]
        rec     = [r["fall_recall"] for r in val_res]
        f1f     = [r["fall_f1"]     for r in val_res]
        summary[name] = {
            "train_acc_mean":   float(np.mean(tr_accs)), "train_acc_std":   float(np.std(tr_accs)),
            "val_acc_mean":     float(np.mean(va_accs)), "val_acc_std":     float(np.std(va_accs)),
            "gap_mean":         float(np.mean(gaps)),    "gap_std":         float(np.std(gaps)),
            "macro_f1_mean":    float(np.mean(mf1s)),    "macro_f1_std":    float(np.std(mf1s)),
            "fall_recall_mean": float(np.mean(rec)),     "fall_recall_std": float(np.std(rec)),
            "fall_f1_mean":     float(np.mean(f1f)),     "fall_f1_std":     float(np.std(f1f)),
        }

    _summarize("Random Forest", rf_fold_train, rf_fold_val)
    if not skip_nn:
        _summarize("Neural Network", nn_fold_train, nn_fold_val)

    # ── Console Diagnostics ────────────────────────────────────────────────────
    log.info("\n════════════════════════════════════════════════════════════════════════════════════════════════")
    log.info("5-FOLD CROSS-VALIDATION DIAGNOSTIC (Underfitting & Overfitting Check)")
    log.info("════════════════════════════════════════════════════════════════════════════════════════════════")
    header = f"{'Model':<16} | {'Train Acc':<15} | {'Val Acc':<15} | {'Gap (Tr-Val)':<14} | {'Underfitting?':<15} | {'Overfitting?'}"
    log.info(header)
    log.info("─" * len(header))
    for model_name, s in summary.items():
        underfit_status = "NO (Train >= 80%)" if s["train_acc_mean"] >= 0.80 else "YES (Train < 80%)"
        overfit_status  = "NO (Gap < 5%)"    if s["gap_mean"] < 0.05        else "YES (Gap >= 5%)"
        log.info(
            "%-16s | %.4f ± %.4f   | %.4f ± %.4f   | %+.4f ± %.4f | %-15s | %s",
            model_name,
            s["train_acc_mean"], s["train_acc_std"],
            s["val_acc_mean"],   s["val_acc_std"],
            s["gap_mean"],       s["gap_std"],
            underfit_status,
            overfit_status,
        )
    log.info("─" * len(header))

    # Save CV report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cv_report_path = REPORT_DIR / "cv_5fold_report.json"
    with open(cv_report_path, "w") as f:
        json.dump({"summary": summary, "folds": fold_details}, f, indent=2)
    log.info("CV 5-fold report saved → %s", cv_report_path)

    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Choose data path ───────────────────────────────────────────────────────
    if args.multi_dataset:
        df = _load_multi_dataset(args)
    else:
        df = _load_synthetic(args)

    # ── Diversity reports ──────────────────────────────────────────────────────
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_balance_reports(df, output_dir=METADATA_DIR)
    if args.multi_dataset:
        generate_stats_plots(df)

    # ── Prepare df for windowing ───────────────────────────────────────────────
    # Ensure 'subject' and 'label' columns exist (windowing reads these)
    if "subject" not in df.columns:
        df["subject"] = df.get("subject_id", df.get("subject", "unknown"))
    if "label" not in df.columns:
        df["label"] = df.get("unified_label", df.get("label", "unknown"))
    if "segment" not in df.columns:
        df["segment"] = 0

    # ── Windowing ──────────────────────────────────────────────────────────────
    log.info("Creating windows (size=%d, step=%d) …", args.window_size, args.step)
    X_raw, y, groups = make_windows(df, args.window_size, args.step)
    all_classes = sorted(np.unique(y).tolist())
    log.info("Windows: %d | Classes (%d): %s", len(X_raw), len(all_classes), all_classes)

    # ── Feature extraction ─────────────────────────────────────────────────────
    log.info("Extracting features …")
    X_feat    = extract_features(X_raw)
    feat_names_list = feature_names()
    log.info("Feature matrix: %s", X_feat.shape)

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv_summary = None
    if not args.skip_cv:
        cv_summary = run_cross_validation(
            X_raw, X_feat, y, groups, all_classes,
            n_splits=args.cv_folds, skip_nn=args.skip_nn, nn_epochs=args.nn_epochs,
        )

    # ── Holdout split ──────────────────────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X_raw, y, groups))
    X_tr_raw, X_te_raw   = X_raw[train_idx], X_raw[test_idx]
    X_tr_feat, X_te_feat = X_feat[train_idx], X_feat[test_idx]
    y_tr, y_te           = y[train_idx], y[test_idx]
    g_tr                 = groups[train_idx]

    log.info("Holdout split: %d train (%d subj) | %d test (%d subj)",
             len(train_idx), len(np.unique(g_tr)),
             len(test_idx),  len(np.unique(groups[test_idx])))

    # ── Train Random Forest ────────────────────────────────────────────────────
    log.info("Training final Random Forest …")
    rf = RandomForestModel()
    rf.fit(X_tr_feat, y_tr)
    rf_train_res = evaluate(y_tr, rf.predict(X_tr_feat),
                            model_name="Random Forest (Train)", classes=all_classes)
    results_rf = evaluate(y_te, rf.predict(X_te_feat),
                          model_name="Random Forest", classes=all_classes,
                          output_dir=PLOT_DIR)

    # ── Train Neural Network ───────────────────────────────────────────────────
    results_nn = None
    nn_train_res = None
    nn = None
    if not args.skip_nn:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=0)
        st, sv  = next(gss_val.split(X_tr_raw, y_tr, g_tr))
        log.info("Training final Neural Network …")
        nn = NeuralNetModel(hparams={"epochs": args.nn_epochs})
        nn.fit(X_tr_raw[st], y_tr[st], X_val=X_tr_raw[sv], y_val=y_tr[sv])
        nn_train_res = evaluate(y_tr[st], nn.predict(X_tr_raw[st]),
                                model_name="Neural Network (Train)", classes=all_classes)
        results_nn = evaluate(y_te, nn.predict(X_te_raw),
                              model_name="Neural Network", classes=all_classes,
                              output_dir=PLOT_DIR)

    # ── Bias-Variance / Underfitting & Overfitting Diagnostic ─────────────────
    log.info("\n════════════════════════════════════════════════════════════════════════════════════════════════")
    log.info("HOLDOUT BIAS-VARIANCE DIAGNOSTIC (Underfitting & Overfitting Check on Unseen Subjects)")
    log.info("════════════════════════════════════════════════════════════════════════════════════════════════")
    diag_results = []
    for m_name, tr_res, te_res in [("Random Forest", rf_train_res, results_rf),
                                   ("Neural Network", nn_train_res, results_nn)]:
        if te_res is None or tr_res is None:
            continue
        tr_acc = tr_res["accuracy"]
        te_acc = te_res["accuracy"]
        gap    = tr_acc - te_acc
        underfit_status = "NO (Train >= 80%)" if tr_acc >= 0.80 else "YES (Train < 80%)"
        overfit_status  = "NO (Gap < 5%)"    if abs(gap) < 0.05 else "YES (Gap >= 5%)"
        target_status   = "PASSED (>= 80%)"  if te_acc >= 0.80 else "FAILED (< 80%)"
        overall_status  = "OPTIMAL FIT" if (tr_acc >= 0.80 and abs(gap) < 0.05) else "NEEDS TUNING"
        diag_results.append({
            "model": m_name,
            "train_accuracy": tr_acc,
            "test_accuracy": te_acc,
            "generalization_gap": gap,
            "underfitting": underfit_status,
            "overfitting": overfit_status,
            "target_80_percent": target_status,
            "overall_status": overall_status,
        })
        log.info("%-16s | Train: %.4f | Test: %.4f | Gap: %+.4f | Underfit: %s | Overfit: %s | %s",
                 m_name, tr_acc, te_acc, gap, underfit_status, overfit_status, overall_status)
    log.info("════════════════════════════════════════════════════════════════════════════════════════════════\n")

    # Save bias-variance & overfitting reports
    import json
    with open(REPORT_DIR / "overfitting_analysis.json", "w") as f:
        json.dump(diag_results, f, indent=2)
    with open(REPORT_DIR / "bias_variance_analysis.json", "w") as f:
        json.dump(diag_results, f, indent=2)

    # ── Compare & select ───────────────────────────────────────────────────────
    all_results = [r for r in [results_rf, results_nn] if r is not None]
    print_comparison_table(all_results)
    best = max(all_results, key=lambda r: (r["fall_recall"], r["macro_f1"]))
    log.info("Best model: %s  (fall recall=%.3f, macro_f1=%.3f)",
             best["model_name"], best["fall_recall"], best["macro_f1"])

    # ── Save models ────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rf.save(MODEL_DIR / "random_forest.joblib")

    if nn is not None:
        nn.save(MODEL_DIR / "neural_net.pt")
        mobile_path  = MODEL_DIR / "neural_net_mobile.ptl"
        labels_path  = MODEL_DIR / "har_labels.txt"
        nn.export_torchscript(mobile_path, window_size=args.window_size)
        labels_path.write_text("\n".join(nn.label_encoder.classes_) + "\n")
        if ANDROID_ASSETS_DIR.exists():
            shutil.copy2(mobile_path, ANDROID_ASSETS_DIR / "har_model.ptl")
            shutil.copy2(labels_path, ANDROID_ASSETS_DIR / "har_labels.txt")
            log.info("✓ Copied .ptl + labels to Android assets.")

    (MODEL_DIR / "best_model.txt").write_text(
        best["model_name"].lower().replace(" ", "_")
    )

    # ── Feature importance ─────────────────────────────────────────────────────
    log.info("Analysing feature importance …")
    analyze_importance(rf_model=rf, X_feat=X_tr_feat, y=y_tr,
                       feat_names=feat_names_list, output_dir=PLOT_DIR)

    # ── Cross-dataset experiments (multi-dataset mode only) ────────────────────
    if args.multi_dataset and args.run_experiments:
        log.info("Running cross-dataset generalisation experiments …")
        run_all_experiments(
            df          = df,
            window_size = args.window_size,
            step        = args.step,
            nn_epochs   = args.nn_epochs,
            skip_nn     = args.skip_nn,
            test_dataset_id = args.exp_b_test_dataset,
            output_dir  = PLOT_DIR,
        )

    # ── Final report ───────────────────────────────────────────────────────────
    save_report(all_results, REPORT_DIR)
    log.info("Pipeline complete.  Outputs in %s", OUTPUT_DIR)


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_synthetic(args: argparse.Namespace) -> "pd.DataFrame":
    import pandas as pd
    data_dir = Path(args.data_dir)
    if args.generate_demo_data or not data_dir.exists() or not list(data_dir.glob("*.csv")):
        log.info("Generating demo data (%d subjects, %ds) …",
                 args.n_subjects, args.duration_s)
        generate_demo_data(data_dir, n_subjects=args.n_subjects,
                           duration_s=args.duration_s, fs=50)
    return load_dataset(data_dir)


def _load_multi_dataset(args: argparse.Namespace) -> "pd.DataFrame":
    raw_root = Path(args.raw_root)
    own_dir  = Path(args.own_dir) if args.own_dir else None

    if args.invalidate_cache:
        invalidate_cache()

    df = harmonize(raw_root, own_raw_dir=own_dir, cache=True)

    # If harmonizer returned nothing (all datasets missing), fall back to demo data
    if len(df) == 0:
        log.warning("No public datasets available — falling back to synthetic demo data.")
        return _load_synthetic(args)

    # Subject split (creates data/metadata/subject_split.json if not present)
    make_subject_split(df, force_redo=False)
    return df


if __name__ == "__main__":
    main()
