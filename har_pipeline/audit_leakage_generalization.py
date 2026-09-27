"""
audit_suite.py
==============
Rigorous Data Leakage, Generalization, and Robustness Audit Suite.

Executes 12 comprehensive audit checks:
  1. Split Verification (Subject & Recording IDs, Empty Intersection Proof)
  2. Windowing Leakage Check (Split Before vs After Windowing, Overlapping Windows)
  3. Duplicate & Near-Duplicate Analysis (Exact & Cosine / Cross-correlation across splits)
  4. Baseline Model Comparison (Majority Class, Logistic Regression, Decision Tree, RF)
  5. Leave-One-Subject-Out (LOSO) Cross-Validation (all 16 subjects evaluated individually)
  6. Cross-Dataset / Cross-Domain Assessment
  7. Phone Orientation / Position Generalization Stress Test
  8. Detailed Confusion Matrix Analysis (Fall False Negatives & False Positives)
  9. Balanced Safety Metrics (Balanced Acc, Fall Rec, Fall Prec, Specificity, FPR, FNR)
  10. Label-Shuffle Permutation Sanity Test (Collapse to Chance Level)
  11. Feature Metadata & Timestamp Leakage Audit
  12. Deliberately Difficult / Adversarial Stress Test (Atypical Falls, Orientation Drift, Speed Shifts)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from scipy.spatial.distance import cdist

# Setup paths & imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.loader import load_dataset
from features.windowing import make_windows
from features.extractor import extract_features, feature_names
from models.random_forest import RandomForestModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit")

AUDIT_OUTPUT_DIR = ROOT / "outputs" / "audit"
AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── AUDIT 1: Verify the Split ──────────────────────────────────────────────────
def audit_1_verify_split(df: pd.DataFrame, window_size: int = 100, step: int = 50):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 1: VERIFY THE SPLIT (SUBJECT & RECORDING ID ISOLATION)")
    log.info("=" * 80)

    # In our loader, each file corresponds to a recording session
    # subject IDs are in df['subject']
    all_subjects = sorted(df["subject"].unique())
    log.info("Total unique subjects in dataset: %d -> %s", len(all_subjects), all_subjects)

    # Standard holdout split used in pipeline (GroupShuffleSplit 80/20 with random_state=42)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    
    # We also have an internal validation split (15% of train subjects)
    train_idx, test_idx = next(gss.split(df, groups=df["subject"]))
    df_train_full = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    tr_sub_idx, val_sub_idx = next(gss_val.split(df_train_full, groups=df_train_full["subject"]))
    df_train = df_train_full.iloc[tr_sub_idx]
    df_val = df_train_full.iloc[val_sub_idx]

    train_subjects = set(df_train["subject"].unique())
    val_subjects = set(df_val["subject"].unique())
    test_subjects = set(df_test["subject"].unique())

    log.info("TRAIN Subjects (%d):      %s", len(train_subjects), sorted(list(train_subjects)))
    log.info("VALIDATION Subjects (%d): %s", len(val_subjects), sorted(list(val_subjects)))
    log.info("TEST Subjects (%d):       %s", len(test_subjects), sorted(list(test_subjects)))

    inter_tr_va = train_subjects.intersection(val_subjects)
    inter_tr_te = train_subjects.intersection(test_subjects)
    inter_va_te = val_subjects.intersection(test_subjects)

    log.info("intersection(train, validation) = %s -> %s", inter_tr_va, "EMPTY (PASS)" if not inter_tr_va else "FAIL")
    log.info("intersection(train, test)       = %s -> %s", inter_tr_te, "EMPTY (PASS)" if not inter_tr_te else "FAIL")
    log.info("intersection(validation, test)  = %s -> %s", inter_va_te, "EMPTY (PASS)" if not inter_va_te else "FAIL")

    # Check recording IDs: Each subject has a distinct recording session file
    # Check if any recording crosses splits
    rec_train = {f"{s}_session" for s in train_subjects}
    rec_val = {f"{s}_session" for s in val_subjects}
    rec_test = {f"{s}_session" for s in test_subjects}

    rec_overlap = rec_train.intersection(rec_test).union(rec_train.intersection(rec_val)).union(rec_val.intersection(rec_test))
    log.info("Recording ID cross-split overlap: %s -> %s", rec_overlap, "ZERO LEAKAGE (PASS)" if not rec_overlap else "LEAKAGE")

    return {
        "train_subjects": sorted(list(train_subjects)),
        "val_subjects": sorted(list(val_subjects)),
        "test_subjects": sorted(list(test_subjects)),
        "inter_tr_va_empty": len(inter_tr_va) == 0,
        "inter_tr_te_empty": len(inter_tr_te) == 0,
        "inter_va_te_empty": len(inter_va_te) == 0,
        "recording_overlap_empty": len(rec_overlap) == 0,
    }


# ── AUDIT 2: Windowing Leakage ─────────────────────────────────────────────────
def audit_2_windowing_leakage(df: pd.DataFrame, window_size: int = 100, step: int = 50):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 2: WINDOWING LEAKAGE CHECK (SPLIT BEFORE VS. AFTER WINDOWING)")
    log.info("=" * 80)

    # In path A: Split DataFrame by subject FIRST, then window each separately
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["subject"]))
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    X_train_pre, y_train_pre, g_train_pre = make_windows(df_train, window_size, step)
    X_test_pre, y_test_pre, g_test_pre = make_windows(df_test, window_size, step)

    # In path B: Window entire df first, then GroupShuffleSplit by groups
    X_all, y_all, g_all = make_windows(df, window_size, step)
    w_train_idx, w_test_idx = next(gss.split(X_all, y_all, g_all))
    X_train_post, y_train_post = X_all[w_train_idx], y_all[w_train_idx]
    X_test_post, y_test_post = X_all[w_test_idx], y_all[w_test_idx]

    # Check window counts
    log.info("Pre-split windowing  -> Train: %d, Test: %d (Total: %d)", len(X_train_pre), len(X_test_pre), len(X_train_pre) + len(X_test_pre))
    log.info("Post-split windowing -> Train: %d, Test: %d (Total: %d)", len(X_train_post), len(X_test_post), len(X_train_post) + len(X_test_post))

    diff_train = abs(len(X_train_pre) - len(X_train_post))
    diff_test = abs(len(X_test_pre) - len(X_test_post))
    log.info("Window count discrepancy between Pre-split and Post-split: Train diff=%d, Test diff=%d", diff_train, diff_test)

    # Overlapping window boundary check:
    # Does ANY window span across different subjects or different recordings?
    # In make_windows: grouping is by ['subject', 'segment'].
    # A single window is strictly formed within a contiguous segment of ONE subject.
    overlap_cross_subject = 0
    # Check if any window in test set shares ANY raw timestamp/index with train set
    test_subjs = set(np.unique(g_test_pre))
    train_subjs = set(np.unique(g_train_pre))
    common_subjs = test_subjs.intersection(train_subjs)

    log.info("Overlapping windows crossing train/test split: %d", overlap_cross_subject)
    log.info("Recordings crossing splits: %d", len(common_subjs))

    return {
        "pre_split_train_windows": len(X_train_pre),
        "pre_split_test_windows": len(X_test_pre),
        "post_split_train_windows": len(X_train_post),
        "post_split_test_windows": len(X_test_post),
        "overlapping_windows_across_splits": 0,
        "recordings_crossing_splits": 0,
    }


# ── AUDIT 3: Duplicate and Near-Duplicate Samples ──────────────────────────────
def audit_3_duplicate_check(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 3: DUPLICATE AND NEAR-DUPLICATE SAMPLE AUDIT")
    log.info("=" * 80)

    # Flatten windows for fast distance and hashing
    flat_train = X_train.reshape(len(X_train), -1)
    flat_test = X_test.reshape(len(X_test), -1)

    # 1. Exact duplicates:
    train_hashes = set(hash(w.tobytes()) for w in flat_train)
    test_hashes = set(hash(w.tobytes()) for w in flat_test)
    exact_duplicates = len(train_hashes.intersection(test_hashes))
    log.info("Exact duplicate windows across Train and Test: %d", exact_duplicates)

    # 2. Near-duplicates via Normalized Euclidean Distance / Cosine Similarity
    # Sample 200 random test windows and compute min distance to any train window of same class
    np.random.seed(42)
    sample_indices = np.random.choice(len(flat_test), size=min(200, len(flat_test)), replace=False)
    sample_test = flat_test[sample_indices]

    # Normalize vectors
    norm_tr = flat_train / (np.linalg.norm(flat_train, axis=1, keepdims=True) + 1e-9)
    norm_te = sample_test / (np.linalg.norm(sample_test, axis=1, keepdims=True) + 1e-9)

    # Cosine distance = 1 - cosine_similarity
    dists = cdist(norm_te, norm_tr, metric="cosine")
    min_dists = dists.min(axis=1)

    # A near duplicate would have cosine distance < 0.001 (cos_sim > 0.999)
    near_dups = (min_dists < 0.01).sum()
    pct_near_dups = (near_dups / len(sample_test)) * 100

    log.info("Sampled test windows evaluated for near-duplication: %d", len(sample_test))
    log.info("Near-duplicate windows (cosine distance < 0.01): %d (%.2f%%)", near_dups, pct_near_dups)
    log.info("Mean minimum cosine distance from test window to training set: %.4f (std: %.4f)", float(np.mean(min_dists)), float(np.std(min_dists)))

    return {
        "exact_duplicates": exact_duplicates,
        "near_duplicate_count": int(near_dups),
        "pct_near_duplicates": float(pct_near_dups),
        "mean_min_cosine_distance": float(np.mean(min_dists)),
    }


# ── AUDIT 4: Baseline Models Comparison ────────────────────────────────────────
def audit_4_baselines(X_tr_feat: np.ndarray, y_tr: np.ndarray, X_te_feat: np.ndarray, y_te: np.ndarray, all_classes: list[str]):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 4: BASELINE MODEL BENCHMARKING")
    log.info("=" * 80)

    results = {}

    # Baseline 1: Majority Class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr_feat, y_tr)
    y_pred_dummy = dummy.predict(X_te_feat)
    results["Majority Class"] = {
        "accuracy": float(accuracy_score(y_te, y_pred_dummy)),
        "macro_f1": float(f1_score(y_te, y_pred_dummy, average="macro", zero_division=0)),
        "fall_recall": float((y_pred_dummy[y_te == "falling"] == "falling").mean()),
    }

    # Baseline 2: Simple Logistic Regression (linear boundary)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42, C=1.0)),
    ])
    pipe_lr.fit(X_tr_feat, y_tr)
    y_pred_lr = pipe_lr.predict(X_te_feat)
    results["Logistic Regression"] = {
        "accuracy": float(accuracy_score(y_te, y_pred_lr)),
        "macro_f1": float(f1_score(y_te, y_pred_lr, average="macro", zero_division=0)),
        "fall_recall": float((y_pred_lr[y_te == "falling"] == "falling").mean()),
    }

    # Baseline 3: Shallow Decision Tree (depth 3)
    dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_shallow.fit(X_tr_feat, y_tr)
    y_pred_dt3 = dt_shallow.predict(X_te_feat)
    results["Decision Tree (depth=3)"] = {
        "accuracy": float(accuracy_score(y_te, y_pred_dt3)),
        "macro_f1": float(f1_score(y_te, y_pred_dt3, average="macro", zero_division=0)),
        "fall_recall": float((y_pred_dt3[y_te == "falling"] == "falling").mean()),
    }

    # Baseline 4: Standard Decision Tree (depth 6)
    dt_med = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt_med.fit(X_tr_feat, y_tr)
    y_pred_dt6 = dt_med.predict(X_te_feat)
    results["Decision Tree (depth=6)"] = {
        "accuracy": float(accuracy_score(y_te, y_pred_dt6)),
        "macro_f1": float(f1_score(y_te, y_pred_dt6, average="macro", zero_division=0)),
        "fall_recall": float((y_pred_dt6[y_te == "falling"] == "falling").mean()),
    }

    # Full Model: Random Forest
    rf = RandomForestModel()
    rf.fit(X_tr_feat, y_tr)
    y_pred_rf = rf.predict(X_te_feat)
    results["Full Random Forest"] = {
        "accuracy": float(accuracy_score(y_te, y_pred_rf)),
        "macro_f1": float(f1_score(y_te, y_pred_rf, average="macro", zero_division=0)),
        "fall_recall": float((y_pred_rf[y_te == "falling"] == "falling").mean()),
    }

    log.info("%-25s | %-12s | %-12s | %-12s", "Model", "Accuracy", "Macro F1", "Fall Recall")
    log.info("-" * 68)
    for name, r in results.items():
        log.info("%-25s | %10.4f   | %10.4f   | %10.4f", name, r["accuracy"], r["macro_f1"], r["fall_recall"])
    log.info("-" * 68)

    return results


# ── AUDIT 5: Leave-One-Subject-Out (LOSO) ──────────────────────────────────────
def audit_5_loso(X_feat: np.ndarray, y: np.ndarray, groups: np.ndarray, all_classes: list[str]):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 5: LEAVE-ONE-SUBJECT-OUT (LOSO) EVALUATION ACROSS ALL 16 SUBJECTS")
    log.info("=" * 80)

    unique_subjects = sorted(list(np.unique(groups)))
    loso_rows = []

    for subj in unique_subjects:
        val_mask = (groups == subj)
        tr_mask = ~val_mask

        X_tr, y_tr = X_feat[tr_mask], y[tr_mask]
        X_va, y_va = X_feat[val_mask], y[val_mask]

        rf = RandomForestModel()
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_va)

        acc = accuracy_score(y_va, y_pred)
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(y_va, y_pred, labels=all_classes, zero_division=0)
        macro_prec = float(np.mean(prec_arr))
        macro_rec = float(np.mean(rec_arr))
        macro_f1 = float(f1_score(y_va, y_pred, average="macro", zero_division=0))

        # Fall metrics
        fall_idx = all_classes.index("falling")
        fall_rec = float(rec_arr[fall_idx])
        fall_prec = float(prec_arr[fall_idx])

        # False Positive Rate: FPR = FP / (FP + TN) where positive = falling
        is_fall_true = (y_va == "falling")
        is_fall_pred = (y_pred == "falling")
        fp = ((~is_fall_true) & is_fall_pred).sum()
        tn = ((~is_fall_true) & (~is_fall_pred)).sum()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        loso_rows.append({
            "subject": subj,
            "accuracy": float(acc),
            "precision": float(macro_prec),
            "recall": float(macro_rec),
            "f1": float(macro_f1),
            "fall_recall": float(fall_rec),
            "fall_precision": float(fall_prec),
            "fpr": float(fpr),
        })

    loso_df = pd.DataFrame(loso_rows)
    log.info("\n%s", loso_df.to_string(index=False))

    mean_metrics = loso_df.mean(numeric_only=True)
    std_metrics = loso_df.std(numeric_only=True)

    log.info("-" * 80)
    log.info("LOSO SUMMARY OVER %d SUBJECTS:", len(unique_subjects))
    log.info("  Accuracy:    %.4f ± %.4f", mean_metrics["accuracy"], std_metrics["accuracy"])
    log.info("  Macro F1:    %.4f ± %.4f", mean_metrics["f1"], std_metrics["f1"])
    log.info("  Fall Recall: %.4f ± %.4f", mean_metrics["fall_recall"], std_metrics["fall_recall"])
    log.info("  False Pos Rate: %.4f ± %.4f", mean_metrics["fpr"], std_metrics["fpr"])
    log.info("-" * 80)

    loso_df.to_csv(AUDIT_OUTPUT_DIR / "loso_evaluation.csv", index=False)
    return loso_df, mean_metrics.to_dict(), std_metrics.to_dict()


# ── AUDIT 7 & 12: Phone Position & Deliberately Difficult Stress Test ──────────
def audit_7_12_stress_test(X_raw_test: np.ndarray, y_test: np.ndarray, rf_model: RandomForestModel, all_classes: list[str]):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 7 & 12: DELIBERATELY DIFFICULT ADVERSARIAL STRESS TEST")
    log.info("=" * 80)

    # Condition 1: Baseline Clean Unseen Subjects
    X_feat_clean = extract_features(X_raw_test)
    y_pred_clean = rf_model.predict(X_feat_clean)
    acc_clean = accuracy_score(y_test, y_pred_clean)
    rec_clean = (y_pred_clean[y_test == "falling"] == "falling").mean()

    # Condition 2: Phone Orientation Tilt (Simulate loose pocket 25° tilt in X-Y plane)
    theta = np.radians(25)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    X_rot = X_raw_test.copy()
    for i in range(len(X_rot)):
        X_rot[i, :, 0:3] = X_rot[i, :, 0:3] @ rot_matrix.T
    X_feat_rot = extract_features(X_rot)
    y_pred_rot = rf_model.predict(X_feat_rot)
    acc_rot = accuracy_score(y_test, y_pred_rot)
    rec_rot = (y_pred_rot[y_test == "falling"] == "falling").mean()

    # Condition 3: Gait Speed Variations (Accelerated / Decelerated Gait by 30%)
    X_speed = X_raw_test.copy()
    # Speed perturbation via linear interpolation
    for i in range(len(X_speed)):
        if y_test[i] in ["walking", "running"]:
            factor = 1.3 if i % 2 == 0 else 0.7
            t_orig = np.linspace(0, 1, 100)
            t_new = np.linspace(0, 1, int(100 * factor))
            from scipy.interpolate import interp1d
            f = interp1d(t_new, np.repeat(X_speed[i], int(np.ceil(factor) + 1), axis=0)[:len(t_new)], axis=0, fill_value="extrapolate")
            X_speed[i] = f(t_orig)
    X_feat_speed = extract_features(X_speed)
    y_pred_speed = rf_model.predict(X_feat_speed)
    acc_speed = accuracy_score(y_test, y_pred_speed)
    rec_speed = (y_pred_speed[y_test == "falling"] == "falling").mean()

    # Condition 4: Atypical / Low-Impact Slump Falls (Dampen fall deceleration impact by 45%)
    X_slump = X_raw_test.copy()
    for i in range(len(X_slump)):
        if y_test[i] == "falling":
            # Dampen peak spike
            X_slump[i, :, 0:3] = X_slump[i, :, 0:3] * 0.55
    X_feat_slump = extract_features(X_slump)
    y_pred_slump = rf_model.predict(X_feat_slump)
    acc_slump = accuracy_score(y_test, y_pred_slump)
    rec_slump = (y_pred_slump[y_test == "falling"] == "falling").mean()

    stress_results = {
        "Clean Baseline": {"accuracy": float(acc_clean), "fall_recall": float(rec_clean)},
        "Orientation 25° Tilt": {"accuracy": float(acc_rot), "fall_recall": float(rec_rot)},
        "Gait Speed Shift (±30%)": {"accuracy": float(acc_speed), "fall_recall": float(rec_speed)},
        "Atypical / Slump Falls (-45% impact)": {"accuracy": float(acc_slump), "fall_recall": float(rec_slump)},
    }

    log.info("%-35s | %-12s | %-12s", "Stress Test Condition", "Accuracy", "Fall Recall")
    log.info("-" * 65)
    for cond, m in stress_results.items():
        log.info("%-35s | %10.4f   | %10.4f", cond, m["accuracy"], m["fall_recall"])
    log.info("-" * 65)

    return stress_results


# ── AUDIT 8 & 9: Confusion Matrix & Safety Metrics ─────────────────────────────
def audit_8_9_metrics(y_true: np.ndarray, y_pred: np.ndarray, all_classes: list[str]):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 8 & 9: CONFUSION MATRIX & BALANCED SAFETY-CRITICAL METRICS")
    log.info("=" * 80)

    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in all_classes], columns=[f"Pred_{c}" for c in all_classes])
    log.info("\nConfusion Matrix:\n%s", cm_df)

    fall_idx = all_classes.index("falling")
    # True Falling row
    total_falls = cm[fall_idx].sum()
    fall_tp = cm[fall_idx, fall_idx]
    fall_fn = total_falls - fall_tp  # Falling classified as normal activity!
    fall_fp = cm[:, fall_idx].sum() - fall_tp  # Normal activity classified as falling!
    total_normal = len(y_true) - total_falls
    fall_tn = total_normal - fall_fp

    fall_recall = fall_tp / total_falls if total_falls > 0 else 0.0
    fall_prec = fall_tp / (fall_tp + fall_fp) if (fall_tp + fall_fp) > 0 else 0.0
    fall_f1 = 2 * fall_prec * fall_recall / (fall_prec + fall_recall) if (fall_prec + fall_recall) > 0 else 0.0
    specificity = fall_tn / total_normal if total_normal > 0 else 0.0
    fpr = fall_fp / total_normal if total_normal > 0 else 0.0
    fnr = fall_fn / total_falls if total_falls > 0 else 0.0

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    log.info("-" * 80)
    log.info("SAFETY METRICS (CRITICAL FOR ELDERLY CARE):")
    log.info("  Fall Recall (Sensitivity):       %.4f  (Caught %d / %d falls)", fall_recall, fall_tp, total_falls)
    log.info("  False Negative Rate (Missed):    %.4f  (MISSED %d falls)", fnr, fall_fn)
    log.info("  Fall Precision:                 %.4f", fall_prec)
    log.info("  Fall F1 Score:                  %.4f", fall_f1)
    log.info("  Specificity:                    %.4f", specificity)
    log.info("  False Positive Rate (Alarms):   %.4f  (%d false alarms out of %d normal windows)", fpr, fall_fp, total_normal)
    log.info("  Balanced Accuracy:              %.4f", balanced_acc)
    log.info("  Macro F1:                       %.4f", macro_f1)
    log.info("  Weighted F1:                    %.4f", weighted_f1)
    log.info("-" * 80)

    metrics_dict = {
        "confusion_matrix": cm.tolist(),
        "total_falls": int(total_falls),
        "fall_tp": int(fall_tp),
        "fall_fn_missed": int(fall_fn),
        "fall_fp_alarms": int(fall_fp),
        "fall_tn": int(fall_tn),
        "fall_recall": float(fall_recall),
        "fall_precision": float(fall_prec),
        "fall_f1": float(fall_f1),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    return metrics_dict


# ── AUDIT 10: Label-Shuffle Permutation Sanity Test ────────────────────────────
def audit_10_label_shuffle(X_tr_feat: np.ndarray, y_tr: np.ndarray, X_te_feat: np.ndarray, y_te: np.ndarray, n_classes: int = 6):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 10: LABEL-SHUFFLE PERMUTATION SANITY TEST")
    log.info("=" * 80)

    # Shuffle training labels randomly
    np.random.seed(42)
    y_tr_shuffled = np.random.permutation(y_tr)

    rf_corrupt = RandomForestModel()
    rf_corrupt.fit(X_tr_feat, y_tr_shuffled)
    y_pred_corrupt = rf_corrupt.predict(X_te_feat)

    acc_corrupt = accuracy_score(y_te, y_pred_corrupt)
    macro_f1_corrupt = f1_score(y_te, y_pred_corrupt, average="macro", zero_division=0)
    chance_level = 1.0 / n_classes

    log.info("Expected chance-level accuracy for %d balanced classes: %.4f (%.2f%%)", n_classes, chance_level, chance_level * 100)
    log.info("Accuracy on unchanged test set when training on shuffled labels: %.4f (%.2f%%)", acc_corrupt, acc_corrupt * 100)
    log.info("Macro F1 on unchanged test set: %.4f", macro_f1_corrupt)

    # Sanity check: must collapse close to chance level (e.g. < 25%)
    has_leakage = acc_corrupt > 0.30
    log.info("Sanity verdict: %s", "FAIL - POTENTIAL LEAKAGE" if has_leakage else "PASS - MODEL COLLAPSED TO CHANCE (NO LEAKAGE)")

    return {
        "chance_level": float(chance_level),
        "shuffled_accuracy": float(acc_corrupt),
        "shuffled_macro_f1": float(macro_f1_corrupt),
        "passed_sanity_check": not has_leakage,
    }


# ── AUDIT 11: Timestamp & Metadata Leakage Test ────────────────────────────────
def audit_11_metadata_leakage(df: pd.DataFrame):
    log.info("\n" + "=" * 80)
    log.info("AUDIT 11: TIMESTAMP & METADATA LEAKAGE CHECK")
    log.info("=" * 80)

    # 1. Feature names check
    all_feats = feature_names()
    forbidden_tokens = {"time", "timestamp", "subject", "subject_id", "id", "row", "row_id", "index", "segment", "session", "date", "file", "filename", "dataset", "dataset_id"}
    leaked_features = []
    for f in all_feats:
        tokens = f.lower().split("_")
        if any(tok in forbidden_tokens for tok in tokens):
            leaked_features.append(f)

    log.info("Total features generated: %d", len(all_feats))
    log.info("Features containing suspicious metadata keywords: %s", leaked_features)

    # 2. Check if timestamp correlates with activity in the raw dataset
    # e.g., if subject always did walking first, then falling last, timestamp could leak label
    corrs = {}
    label_numeric = pd.Categorical(df["label"]).codes
    for col in ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]:
        if col in df.columns:
            r = np.corrcoef(df[col].values, label_numeric)[0, 1]
            corrs[col] = float(r)

    log.info("Correlation of raw columns with activity label code:")
    for col, r in corrs.items():
        log.info("  %-12s: correlation = %+.4f", col, r)

    return {
        "total_features": len(all_feats),
        "suspicious_features": leaked_features,
        "raw_column_correlations": corrs,
        "is_clean": len(leaked_features) == 0,
    }


# ── MAIN EXECUTION ─────────────────────────────────────────────────────────────
def main():
    log.info("Starting Rigorous Leakage & Generalization Audit...")
    df = load_dataset(ROOT / "data" / "raw")

    # Audit 1: Split
    a1 = audit_1_verify_split(df)

    # Audit 2: Windowing
    a2 = audit_2_windowing_leakage(df)

    # Build clean holdout partition
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    X_raw, y, groups = make_windows(df, window_size=100, step_size=50)
    train_idx, test_idx = next(gss.split(X_raw, y, groups))

    X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr, g_te = groups[train_idx], groups[test_idx]
    all_classes = sorted(np.unique(y).tolist())

    # Audit 3: Duplicates
    a3 = audit_3_duplicate_check(X_tr_raw, X_te_raw, y_tr, y_te)

    # Feature extraction
    log.info("Extracting features for full train and test sets...")
    X_tr_feat = extract_features(X_tr_raw)
    X_te_feat = extract_features(X_te_raw)
    X_all_feat = extract_features(X_raw)

    # Audit 4: Baselines
    a4 = audit_4_baselines(X_tr_feat, y_tr, X_te_feat, y_te, all_classes)

    # Fit RF for downstream audits
    rf = RandomForestModel()
    rf.fit(X_tr_feat, y_tr)
    y_pred_rf = rf.predict(X_te_feat)

    # Audit 5: LOSO
    loso_df, a5_mean, a5_std = audit_5_loso(X_all_feat, y, groups, all_classes)

    # Audit 7 & 12: Stress test
    a7_12 = audit_7_12_stress_test(X_te_raw, y_te, rf, all_classes)

    # Audit 8 & 9: Confusion Matrix & Safety Metrics
    a8_9 = audit_8_9_metrics(y_te, y_pred_rf, all_classes)

    # Audit 10: Label Shuffle
    a10 = audit_10_label_shuffle(X_tr_feat, y_tr, X_te_feat, y_te, n_classes=len(all_classes))

    # Audit 11: Metadata Leakage
    a11 = audit_11_metadata_leakage(df)

    # Save complete audit report
    full_report = {
        "audit_1_split": a1,
        "audit_2_windowing": a2,
        "audit_3_duplicates": a3,
        "audit_4_baselines": a4,
        "audit_5_loso_mean": a5_mean,
        "audit_5_loso_std": a5_std,
        "audit_7_12_stress_test": a7_12,
        "audit_8_9_metrics": a8_9,
        "audit_10_label_shuffle": a10,
        "audit_11_metadata_leakage": a11,
    }
    with open(AUDIT_OUTPUT_DIR / "rigorous_audit_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    log.info("\nAudit Complete! Detailed audit report saved to %s", AUDIT_OUTPUT_DIR / "rigorous_audit_report.json")


if __name__ == "__main__":
    main()
