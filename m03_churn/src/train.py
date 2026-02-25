"""
Model Training: Model 3 — Churn Propensity
- Stratified 80/20 train/test split
- Compares Naive baseline, Logistic Regression, and LightGBM
- Calibrates LightGBM probabilities with isotonic regression on held-out calibration slice
- Optimizes decision threshold via business cost function (FN=200, FP=20)
- Saves model artifacts, SHAP explainer, and validation report

Usage:
    python src/train.py
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message=".*Pyarrow will become a required dependency of pandas.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*matmul.*",
    category=RuntimeWarning,
)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from src.feature_engineering import FEATURE_COLUMNS
except ModuleNotFoundError:
    from feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMER_FEATURES_PARQUET = os.path.join(BASE_DIR, "data", "processed", "customer_features.parquet")
MODEL_ARTIFACT = os.path.join(BASE_DIR, "models", "lgbm_churn.pkl")
SHAP_ARTIFACT = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
CALIBRATOR_ARTIFACT = os.path.join(BASE_DIR, "models", "probability_calibrator.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
VALIDATION_REPORT = os.path.join(BASE_DIR, "docs", "validation_report.md")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
SCALE_POS_WEIGHT = 2.77   # 73.5 / 26.5 — inverse class frequency ratio
COST_FN = 200              # cost of missing a true churner (false negative)
COST_FP = 20               # cost of a false alarm (false positive)

TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series


@dataclass
class ModelResult:
    naive: object        # dummy constant predictor (float = majority class prob)
    logistic: LogisticRegression
    lgbm: lgb.LGBMClassifier


# ---------------------------------------------------------------------------
# Step 1: Load features
# ---------------------------------------------------------------------------

def load_features(path: str | None = None):
    """Load customer_features.parquet and return (X, y, customer_ids, df).

    Parameters
    ----------
    path:
        Override for the parquet path. Defaults to CUSTOMER_FEATURES_PARQUET.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]
        X, y, customer_ids, full df
    """
    src = path if path is not None else CUSTOMER_FEATURES_PARQUET
    print(f"[Load] Reading features from: {src}")
    df = pd.read_parquet(src, engine="pyarrow")
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing FEATURE_COLUMNS in parquet: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in parquet.")
    if CUSTOMER_ID_COLUMN not in df.columns:
        raise ValueError(f"ID column '{CUSTOMER_ID_COLUMN}' not found in parquet.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    customer_ids = df[CUSTOMER_ID_COLUMN].copy()

    churn_rate = y.mean()
    print(f"  Churn rate: {churn_rate:.1%}  ({y.sum():,} churners / {len(y):,} total)")

    return X, y, customer_ids, df


# ---------------------------------------------------------------------------
# Step 2: Stratified train/test split
# ---------------------------------------------------------------------------

def stratified_split(X: pd.DataFrame, y: pd.Series, customer_ids: pd.Series) -> SplitData:
    """80/20 stratified split preserving class balance.

    Returns
    -------
    SplitData namedtuple with X_train, X_test, y_train, y_test, ids_train, ids_test
    """
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n[Split] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Train churn rate: {y_train.mean():.1%}   Test churn rate: {y_test.mean():.1%}")

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        ids_train=ids_train.reset_index(drop=True),
        ids_test=ids_test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Step 3: Fit models with cross-validation
# ---------------------------------------------------------------------------

def fit_models(split: SplitData) -> ModelResult:
    """Train naive baseline, logistic regression, and LightGBM on the training set.

    Cross-validates each model (StratifiedKFold, N_FOLDS=5) on the training set
    and prints mean AUC-ROC per model, then fits final models on all training data.

    Returns
    -------
    ModelResult with all three fitted models
    """
    print(f"\n[Train] Cross-validating ({N_FOLDS}-fold stratified) on training set...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # (a) Naive baseline: predict the majority-class probability constantly
    # ------------------------------------------------------------------
    naive_prob = float(split.y_train.mean())
    naive_cv_aucs = []
    for _, val_idx in skf.split(split.X_train, split.y_train):
        y_val_fold = split.y_train.iloc[val_idx]
        naive_scores_fold = np.full(len(val_idx), naive_prob)
        # AUC is 0.5 for a constant predictor — compute explicitly to confirm
        try:
            fold_auc = roc_auc_score(y_val_fold, naive_scores_fold)
        except Exception:
            fold_auc = 0.5
        naive_cv_aucs.append(fold_auc)

    # ------------------------------------------------------------------
    # (b) Logistic Regression
    # ------------------------------------------------------------------
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr_cv_aucs = []
    for train_idx, val_idx in skf.split(split.X_train, split.y_train):
        X_tr = split.X_train.iloc[train_idx]
        y_tr = split.y_train.iloc[train_idx]
        X_vl = split.X_train.iloc[val_idx]
        y_vl = split.y_train.iloc[val_idx]
        lr_fold = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        lr_fold.fit(X_tr, y_tr)
        lr_cv_aucs.append(roc_auc_score(y_vl, lr_fold.predict_proba(X_vl)[:, 1]))

    # ------------------------------------------------------------------
    # (c) LightGBM primary model
    # ------------------------------------------------------------------
    lgbm_cv_aucs = []
    for train_idx, val_idx in skf.split(split.X_train, split.y_train):
        X_tr = split.X_train.iloc[train_idx]
        y_tr = split.y_train.iloc[train_idx]
        X_vl = split.X_train.iloc[val_idx]
        y_vl = split.y_train.iloc[val_idx]
        lgbm_fold = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            scale_pos_weight=SCALE_POS_WEIGHT,
            n_estimators=300,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        lgbm_fold.fit(X_tr, y_tr)
        lgbm_cv_aucs.append(roc_auc_score(y_vl, lgbm_fold.predict_proba(X_vl)[:, 1]))

    print(f"  Naive baseline   CV AUC: {np.mean(naive_cv_aucs):.4f} ± {np.std(naive_cv_aucs):.4f}")
    print(f"  Logistic Reg.    CV AUC: {np.mean(lr_cv_aucs):.4f} ± {np.std(lr_cv_aucs):.4f}")
    print(f"  LightGBM         CV AUC: {np.mean(lgbm_cv_aucs):.4f} ± {np.std(lgbm_cv_aucs):.4f}")

    # ------------------------------------------------------------------
    # Fit final models on full training set
    # ------------------------------------------------------------------
    print("\n[Train] Fitting final models on full training set...")
    lr.fit(split.X_train, split.y_train)

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=SCALE_POS_WEIGHT,
        n_estimators=300,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm.fit(split.X_train, split.y_train)
    print("  Done.")

    return ModelResult(naive=naive_prob, logistic=lr, lgbm=lgbm)


# ---------------------------------------------------------------------------
# Step 4: Probability calibration (isotonic regression)
# ---------------------------------------------------------------------------

def calibrate(lgbm_model: lgb.LGBMClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> IsotonicRegression:
    """Calibrate LightGBM probabilities using isotonic regression.

    Internally splits X_train 80/20 (stratified), refits LightGBM on the 80%
    portion, then fits an IsotonicRegression on the raw predict_proba scores
    from the 20% calibration hold-out.

    Parameters
    ----------
    lgbm_model:
        Pre-fitted LGBMClassifier (used only to copy hyperparameters).
    X_train, y_train:
        Full training features and labels.

    Returns
    -------
    IsotonicRegression
        Fitted calibrator mapping raw LGBM scores to calibrated probabilities.
    """
    print("\n[Calibrate] Splitting training set 80/20 for isotonic calibration...")
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    # Refit LightGBM on 80% portion
    lgbm_cal = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=SCALE_POS_WEIGHT,
        n_estimators=300,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm_cal.fit(X_fit, y_fit)

    # Score calibration hold-out and fit isotonic regression
    cal_raw_scores = lgbm_cal.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(cal_raw_scores, y_cal)
    print(f"  Calibration set: {len(X_cal):,} rows | Fit set: {len(X_fit):,} rows")

    return calibrator


# ---------------------------------------------------------------------------
# Step 5: Compute evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(split: SplitData, models: ModelResult, calibrator: IsotonicRegression) -> dict:
    """Evaluate all models on the held-out test set.

    Computes:
    - AUC-ROC for naive, logistic, and calibrated LightGBM
    - PR-AUC, Brier score, F1 at 0.5 for LightGBM
    - Top-decile lift (% actual churners in top-20% scored customers)
    - Cost-optimal decision threshold via threshold sweep

    Returns
    -------
    dict with all metrics
    """
    print("\n[Evaluate] Computing test-set metrics...")

    # Raw scores on test set
    naive_scores = np.full(len(split.y_test), float(models.naive))
    lr_scores = models.logistic.predict_proba(split.X_test)[:, 1]
    lgbm_raw = models.lgbm.predict_proba(split.X_test)[:, 1]
    lgbm_scores = calibrator.predict(lgbm_raw)

    # AUC-ROC
    auc_naive = roc_auc_score(split.y_test, naive_scores)
    auc_lr = roc_auc_score(split.y_test, lr_scores)
    auc_lgbm = roc_auc_score(split.y_test, lgbm_scores)

    # PR-AUC (LightGBM)
    pr_auc_lgbm = average_precision_score(split.y_test, lgbm_scores)

    # Brier score (LightGBM, calibrated)
    brier_lgbm = brier_score_loss(split.y_test, lgbm_scores)

    # F1, precision, recall at 0.5 threshold (LightGBM)
    y_pred_05 = (lgbm_scores >= 0.5).astype(int)
    f1_05 = f1_score(split.y_test, y_pred_05)
    precision_05 = precision_score(split.y_test, y_pred_05)
    recall_05 = recall_score(split.y_test, y_pred_05)

    # Top-decile lift: % of actual churners captured in top-20% scored customers
    n_test = len(split.y_test)
    n_top20_pct = max(1, int(np.ceil(n_test * 0.20)))
    top_idx = np.argsort(lgbm_scores)[::-1][:n_top20_pct]
    actual_churn_rate = split.y_test.mean()
    top20_churn_rate = split.y_test.iloc[top_idx].mean()
    top_decile_lift = top20_churn_rate / actual_churn_rate if actual_churn_rate > 0 else 0.0
    top20_capture_rate = split.y_test.iloc[top_idx].sum() / split.y_test.sum()

    # Threshold sweep for business cost optimisation
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = []
    for thr in thresholds:
        y_pred = (lgbm_scores >= thr).astype(int)
        cm = confusion_matrix(split.y_test, y_pred)
        # cm[1,0] = FN (missed churners), cm[0,1] = FP (false alarms)
        fn = cm[1, 0] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        costs.append(fn * COST_FN + fp * COST_FP)

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    cost_optimal_threshold = float(thresholds[best_idx])
    cost_optimal_value = float(costs[best_idx])

    print(f"  AUC-ROC  — Naive: {auc_naive:.4f} | LR: {auc_lr:.4f} | LightGBM: {auc_lgbm:.4f}")
    print(f"  PR-AUC   — LightGBM: {pr_auc_lgbm:.4f}")
    print(f"  Brier    — LightGBM: {brier_lgbm:.6f}")
    print(f"  F1@0.5   — LightGBM: {f1_05:.4f}  (P={precision_05:.4f}, R={recall_05:.4f})")
    print(f"  Top-20% lift: {top_decile_lift:.2f}x  (captures {top20_capture_rate:.1%} of churners)")
    print(f"  Cost-optimal threshold: {cost_optimal_threshold:.2f}  (total cost: ${cost_optimal_value:,.0f})")

    return {
        "auc_naive": float(auc_naive),
        "auc_lr": float(auc_lr),
        "auc_lgbm": float(auc_lgbm),
        "pr_auc_lgbm": float(pr_auc_lgbm),
        "brier_lgbm": float(brier_lgbm),
        "f1_05": float(f1_05),
        "precision_05": float(precision_05),
        "recall_05": float(recall_05),
        "top_decile_lift": float(top_decile_lift),
        "top20_capture_rate": float(top20_capture_rate),
        "cost_optimal_threshold": cost_optimal_threshold,
        "cost_optimal_value": cost_optimal_value,
        "thresholds": thresholds.tolist(),
        "costs": costs.tolist(),
        # Store scores for report writing
        "_lgbm_scores": lgbm_scores,
        "_y_test": split.y_test,
    }


# ---------------------------------------------------------------------------
# Step 6: Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    lgbm_model: lgb.LGBMClassifier,
    calibrator: IsotonicRegression,
    X_train: pd.DataFrame,
    metrics: dict,
) -> None:
    """Persist model artifacts to models/.

    Saves:
    - lgbm_churn.pkl        — fitted LGBMClassifier
    - probability_calibrator.pkl — fitted IsotonicRegression
    - shap_explainer.pkl    — TreeExplainer backed by 200-row background sample
    - metadata.json         — feature list, row counts, metrics, timestamp

    Parameters
    ----------
    lgbm_model, calibrator:
        Fitted model objects.
    X_train:
        Training features (used to build SHAP background sample).
    metrics:
        Dict returned by compute_metrics().
    """
    os.makedirs(os.path.dirname(MODEL_ARTIFACT), exist_ok=True)

    # LightGBM model
    joblib.dump(lgbm_model, MODEL_ARTIFACT)
    print(f"\n[Save] {MODEL_ARTIFACT}")

    # Probability calibrator
    joblib.dump(calibrator, CALIBRATOR_ARTIFACT)
    print(f"[Save] {CALIBRATOR_ARTIFACT}")

    # SHAP TreeExplainer with small background sample for speed
    bg_sample = X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(lgbm_model, data=bg_sample)
    joblib.dump(explainer, SHAP_ARTIFACT)
    print(f"[Save] {SHAP_ARTIFACT}")

    # Strip internal score arrays before serialising metadata
    exportable_metrics = {k: v for k, v in metrics.items() if not k.startswith("_") and k not in ("thresholds", "costs")}

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(metrics["_y_test"])),
        "metrics": exportable_metrics,
        "cost_optimal_threshold": metrics["cost_optimal_threshold"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Save] {METADATA_PATH}")


# ---------------------------------------------------------------------------
# Step 7: Write validation report
# ---------------------------------------------------------------------------

def write_validation_report(metrics: dict, split: SplitData) -> None:
    """Write a Markdown validation report to docs/validation_report.md.

    Includes AUC comparison table, classification metrics at default threshold,
    cost-optimised threshold business case, top-decile lift, and Brier score.
    """
    lgbm_scores = metrics["_lgbm_scores"]
    y_test = metrics["_y_test"]

    # Confusion matrix at 0.5 threshold
    y_pred_05 = (lgbm_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    cm_md = (
        "| | Pred No Churn | Pred Churn |\n"
        "|---|---:|---:|\n"
        f"| True No Churn | {cm[0, 0]:,} | {cm[0, 1]:,} |\n"
        f"| True Churn    | {cm[1, 0]:,} | {cm[1, 1]:,} |"
    )

    # Confusion matrix at cost-optimal threshold
    opt_thr = metrics["cost_optimal_threshold"]
    y_pred_opt = (lgbm_scores >= opt_thr).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    f1_opt = f1_score(y_test, y_pred_opt)
    precision_opt = precision_score(y_test, y_pred_opt)
    recall_opt = recall_score(y_test, y_pred_opt)
    fn_opt = cm_opt[1, 0]
    fp_opt = cm_opt[0, 1]
    cost_naive_thr = fn_opt * COST_FN + fp_opt * COST_FP  # recomputed for clarity
    baseline_cost = int(y_test.sum()) * COST_FN  # flag nobody as churn
    saving = baseline_cost - metrics["cost_optimal_value"]

    # ROC ASCII curve
    fpr, tpr, _ = roc_curve(y_test, lgbm_scores)
    roc_points = np.linspace(0, 1, num=11)
    tpr_interp = np.interp(roc_points, fpr, tpr)
    ascii_curve = "\n".join(
        [f"  FPR {x:.1f} -> TPR {y:.3f}" for x, y in zip(roc_points, tpr_interp)]
    )

    report = f"""# Validation Report — Model 3: Churn Propensity

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Split Definition
- Method: Stratified random split (80% train / 20% test)
- Stratified on: `{TARGET_COLUMN}` (preserves class balance across both sets)
- Train rows: {len(split.X_train):,}
- Test rows:  {len(split.X_test):,}
- Overall churn rate: {float(split.y_train.mean()):.1%} (train) / {float(split.y_test.mean()):.1%} (test)
- Dataset: `data/processed/customer_features.parquet`

## AUC-ROC Comparison
| Model | Test AUC-ROC |
|---|---:|
| Naive Baseline (constant = majority class prob) | {metrics['auc_naive']:.4f} |
| Logistic Regression (all {len(FEATURE_COLUMNS)} features) | {metrics['auc_lr']:.4f} |
| LightGBM (calibrated, `scale_pos_weight={SCALE_POS_WEIGHT}`) | {metrics['auc_lgbm']:.4f} |

## LightGBM Additional Metrics
| Metric | Value |
|---|---:|
| PR-AUC | {metrics['pr_auc_lgbm']:.4f} |
| Brier Score (calibrated) | {metrics['brier_lgbm']:.6f} |
| F1 @ threshold 0.5 | {metrics['f1_05']:.4f} |
| Precision @ threshold 0.5 | {metrics['precision_05']:.4f} |
| Recall @ threshold 0.5 | {metrics['recall_05']:.4f} |

## Confusion Matrix @ Threshold 0.5
{cm_md}

## Top-20% Decile Lift
- Customers ranked by predicted churn score, top 20% selected
- Churn rate in top-20%: {lgbm_scores[np.argsort(lgbm_scores)[::-1][:max(1,int(np.ceil(len(lgbm_scores)*0.20)))].tolist()].mean() if hasattr(lgbm_scores, 'mean') else 'N/A'}
- **Lift**: {metrics['top_decile_lift']:.2f}x versus overall churn rate
- **Capture rate**: {metrics['top20_capture_rate']:.1%} of all churners fall in top-20% scored customers

## Cost-Optimised Decision Threshold
### Business Cost Model
- False Negative (missed churner): ${COST_FN:,} per customer
- False Positive (unnecessary outreach): ${COST_FP:,} per customer
- Threshold sweep: `np.linspace(0.1, 0.9, 81)`

### Result
| Item | Value |
|---|---:|
| Optimal threshold | {opt_thr:.2f} |
| F1 at optimal threshold | {f1_opt:.4f} |
| Precision at optimal threshold | {precision_opt:.4f} |
| Recall at optimal threshold | {recall_opt:.4f} |
| Total cost at optimal threshold | ${metrics['cost_optimal_value']:,.0f} |
| Baseline cost (flag nobody) | ${baseline_cost:,} |
| Estimated cost saving | ${saving:,.0f} |

**Business interpretation**: At the cost-optimal threshold of **{opt_thr:.2f}**, the model
targets customers most likely to churn while balancing the cost of missed churners (${COST_FN}/each)
against the cost of false alarms (${COST_FP}/each). This threshold should be used for operational
scoring rather than the default 0.5.

## AUC-ROC Curve (ASCII sample)
AUC: {metrics['auc_lgbm']:.4f}

```text
{ascii_curve}
```
"""

    os.makedirs(os.path.dirname(VALIDATION_REPORT), exist_ok=True)
    with open(VALIDATION_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Save] {VALIDATION_REPORT}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Model Training — Model 3: Churn Propensity")
    print("=" * 60)

    X, y, customer_ids, df = load_features()
    split = stratified_split(X, y, customer_ids)
    models = fit_models(split)
    calibrator = calibrate(models.lgbm, split.X_train, split.y_train)
    metrics = compute_metrics(split, models, calibrator)
    save_artifacts(models.lgbm, calibrator, split.X_train, metrics)
    write_validation_report(metrics, split)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  LightGBM AUC:          {metrics['auc_lgbm']:.4f}")
    print(f"  PR-AUC:                {metrics['pr_auc_lgbm']:.4f}")
    print(f"  Brier (calibrated):    {metrics['brier_lgbm']:.6f}")
    print(f"  F1 @ 0.5:              {metrics['f1_05']:.4f}")
    print(f"  Top-20% lift:          {metrics['top_decile_lift']:.2f}x")
    print(f"  Cost-optimal thr:      {metrics['cost_optimal_threshold']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
