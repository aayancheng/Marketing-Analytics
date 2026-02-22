"""
Model Training: Phase 4 (Leakage-safe)
- Trains/evaluates on event_features.csv (event-level rows with send_datetime)
- Time-based split: first 18 months train, final 6 months test
- Compares Naive, Logistic Regression, and LightGBM
- Calibrates LightGBM probabilities with isotonic regression on validation slice
- Saves model artifacts and validation report

Usage:
    python src/train.py
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

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
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    from src.feature_engineering import FEATURE_COLUMNS
except ModuleNotFoundError:
    from feature_engineering import FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "event_features.csv")
CUSTOMER_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
MODEL_ARTIFACT = os.path.join(BASE_DIR, "models", "lgbm_time_to_engage.pkl")
SHAP_ARTIFACT = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
CALIBRATOR_ARTIFACT = os.path.join(BASE_DIR, "models", "probability_calibrator.pkl")
VALIDATION_REPORT = os.path.join(BASE_DIR, "docs", "validation_report.md")


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_df: pd.DataFrame


def time_split(event_df: pd.DataFrame) -> SplitData:
    start = event_df["send_datetime"].min()
    cutoff_train = start + pd.DateOffset(months=18)
    cutoff_test = start + pd.DateOffset(months=24)

    train_raw = event_df[event_df["send_datetime"] < cutoff_train].copy()
    tail_raw = event_df[(event_df["send_datetime"] >= cutoff_train) & (event_df["send_datetime"] < cutoff_test)].copy()

    # Validation slice for probability calibration: first half of tail, second half test.
    mid = tail_raw["send_datetime"].min() + (tail_raw["send_datetime"].max() - tail_raw["send_datetime"].min()) / 2
    val_raw = tail_raw[tail_raw["send_datetime"] <= mid].copy()
    test_raw = tail_raw[tail_raw["send_datetime"] > mid].copy()

    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        raise RuntimeError("Time split produced empty partition(s). Check send_datetime coverage.")

    return SplitData(
        X_train=train_raw[FEATURE_COLUMNS],
        y_train=train_raw["opened"].astype(int),
        X_val=val_raw[FEATURE_COLUMNS],
        y_val=val_raw["opened"].astype(int),
        X_test=test_raw[FEATURE_COLUMNS],
        y_test=test_raw["opened"].astype(int),
        test_df=test_raw,
    )


def fit_models(split: SplitData):
    # Naive random baseline
    rng = np.random.default_rng(42)
    naive_scores = rng.random(len(split.y_test))

    # Logistic baseline (send_hour + send_dow)
    lr = LogisticRegression(max_iter=200, solver="liblinear", random_state=42)
    lr.fit(split.X_train[["send_hour", "send_dow"]], split.y_train)
    lr_scores = lr.predict_proba(split.X_test[["send_hour", "send_dow"]])[:, 1]

    # LightGBM primary model
    lgbm = LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=3.0,
        verbose=-1,
        random_state=42,
        n_estimators=300,
    )
    lgbm.fit(split.X_train, split.y_train)

    # Calibrate LightGBM probabilities using isotonic regression on validation slice
    val_raw_scores = lgbm.predict_proba(split.X_val)[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(val_raw_scores, split.y_val)

    test_raw_scores = lgbm.predict_proba(split.X_test)[:, 1]
    lgbm_scores = calibrator.predict(test_raw_scores)

    return {
        "naive_scores": naive_scores,
        "lr": lr,
        "lr_scores": lr_scores,
        "lgbm": lgbm,
        "calibrator": calibrator,
        "lgbm_scores": lgbm_scores,
        "test_raw_scores": test_raw_scores,
    }


def compute_top3_hit_rate(model, calibrator, test_df: pd.DataFrame, customer_features: pd.DataFrame):
    # Ground truth optimal slot on test events (customers with >= 3 test events)
    counts = test_df.groupby("customer_id").size()
    eligible_customers = counts[counts >= 3].index
    test_eval = test_df[test_df["customer_id"].isin(eligible_customers)].copy()

    truth = (
        test_eval.groupby(["customer_id", "send_hour", "send_dow"])["opened"]
        .mean()
        .reset_index()
        .sort_values(["customer_id", "opened"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .rename(columns={"send_hour": "truth_hour", "send_dow": "truth_dow"})
    )

    # Score all 168 slots for each test customer using customer profile features.
    slots = pd.DataFrame([(h, d) for d in range(7) for h in range(24)], columns=["send_hour", "send_dow"])
    cust = customer_features[customer_features["customer_id"].isin(eligible_customers)].copy()
    cust["_key"] = 1
    slots["_key"] = 1
    grid = cust.merge(slots, on="_key").drop(columns="_key")

    # Recompute slot/interactions exactly as training features.
    grid["is_weekend"] = (grid["send_dow"] >= 5).astype(np.int8)
    grid["is_business_hours"] = ((grid["send_hour"] >= 9) & (grid["send_hour"] <= 17)).astype(np.int8)
    delta = (grid["send_hour"] - grid["modal_purchase_hour"]).abs()
    grid["hour_delta_from_modal"] = np.minimum(delta, 24 - delta).astype(np.int16)
    grid["dow_match"] = (grid["send_dow"] == grid["modal_purchase_dow"]).astype(np.int8)
    try:
        from src.feature_engineering import INDUSTRY_OPEN_RATE_BY_HOUR
    except ModuleNotFoundError:
        from feature_engineering import INDUSTRY_OPEN_RATE_BY_HOUR

    grid["industry_open_rate_by_hour"] = grid["send_hour"].map(INDUSTRY_OPEN_RATE_BY_HOUR).astype(float)
    grid["hour_x_entropy"] = grid["send_hour"] * grid["purchase_hour_entropy"]
    grid["recency_x_frequency"] = grid["recency_days"] * grid["frequency"]

    grid["score"] = calibrator.predict(model.predict_proba(grid[FEATURE_COLUMNS])[:, 1])

    top3 = (
        grid.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id")
        .head(3)
        .groupby("customer_id")
        .apply(lambda g: set(zip(g["send_hour"], g["send_dow"])), include_groups=False)
        .rename("pred_top3")
        .reset_index()
    )

    eval_df = truth.merge(top3, on="customer_id", how="inner")
    eval_df["hit"] = eval_df.apply(lambda r: (int(r["truth_hour"]), int(r["truth_dow"])) in r["pred_top3"], axis=1)

    seg = customer_features[["customer_id", "rfm_segment"]]
    eval_df = eval_df.merge(seg, on="customer_id", how="left")

    overall = float(eval_df["hit"].mean()) if len(eval_df) else 0.0
    by_segment = eval_df.groupby("rfm_segment")["hit"].mean().sort_values(ascending=False)
    return overall, by_segment


def write_validation_report(metrics: dict, split: SplitData, by_segment: pd.Series, top3_overall: float) -> None:
    fpr, tpr, _ = roc_curve(split.y_test, metrics["lgbm_scores"])
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(split.y_test, (metrics["lgbm_scores"] >= 0.5).astype(int))
    cm_md = f"| | Pred 0 | Pred 1 |\n|---|---:|---:|\n| True 0 | {cm[0,0]} | {cm[0,1]} |\n| True 1 | {cm[1,0]} | {cm[1,1]} |"

    segment_rows = "\n".join([f"| {idx} | {val:.4f} |" for idx, val in by_segment.items()])
    if not segment_rows:
        segment_rows = "| N/A | 0.0000 |"

    roc_points = np.linspace(0, 1, num=11)
    tpr_interp = np.interp(roc_points, fpr, tpr)
    ascii_curve = "\n".join(
        [f"FPR {x:.1f} -> TPR {y:.3f}" for x, y in zip(roc_points, tpr_interp, strict=True)]
    )

    report = f"""# Validation Report (Temporal, Leakage-Safe)

## Split Definition
- Training window: first 18 months from first campaign timestamp
- Validation window: first half of final 6 months (for calibration)
- Test window: second half of final 6 months
- Dataset: `data/processed/event_features.csv`

## AUC Comparison
| Model | Test AUC |
|---|---:|
| Naive Random | {metrics['auc_naive']:.4f} |
| Logistic Regression (`send_hour`,`send_dow`) | {metrics['auc_lr']:.4f} |
| LightGBM (20 features, calibrated) | {metrics['auc_lgbm']:.4f} |

## Calibration
- Brier score (calibrated LightGBM): {metrics['brier_lgbm']:.6f}

## Confusion Matrix @ 0.5
{cm_md}

## Top-3 Hit Rate
- Overall: {top3_overall:.4f}

| RFM Segment | Top-3 Hit Rate |
|---|---:|
{segment_rows}

## AUC-ROC (ASCII sample points)
- AUC: {roc_auc:.4f}

```text
{ascii_curve}
```
"""

    os.makedirs(os.path.dirname(VALIDATION_REPORT), exist_ok=True)
    with open(VALIDATION_REPORT, "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    print("=" * 60)
    print("Model Training — Phase 4")
    print("=" * 60)

    event_df = pd.read_csv(EVENT_FEATURES_CSV, parse_dates=["send_datetime"])
    customer_features = pd.read_csv(CUSTOMER_FEATURES_CSV)
    customer_features["customer_id"] = customer_features["customer_id"].astype(np.int64)

    split = time_split(event_df)
    models = fit_models(split)

    auc_naive = roc_auc_score(split.y_test, models["naive_scores"])
    auc_lr = roc_auc_score(split.y_test, models["lr_scores"])
    auc_lgbm = roc_auc_score(split.y_test, models["lgbm_scores"])
    brier_lgbm = brier_score_loss(split.y_test, models["lgbm_scores"])

    # SHAP explainer (small background sample for speed)
    bg = split.X_train.sample(min(500, len(split.X_train)), random_state=42)
    explainer = shap.TreeExplainer(models["lgbm"], data=bg)

    top3_overall, by_segment = compute_top3_hit_rate(models["lgbm"], models["calibrator"], split.test_df, customer_features)

    print("\nAUC Comparison")
    print("-" * 60)
    print(f"Naive Random:          {auc_naive:.4f}")
    print(f"Logistic (2 features): {auc_lr:.4f}")
    print(f"LightGBM (20 feats):   {auc_lgbm:.4f}")
    print(f"Brier (calibrated):    {brier_lgbm:.6f}")
    print(f"Top-3 hit rate:        {top3_overall:.4f}")

    os.makedirs(os.path.dirname(MODEL_ARTIFACT), exist_ok=True)
    joblib.dump(models["lgbm"], MODEL_ARTIFACT)
    joblib.dump(explainer, SHAP_ARTIFACT)
    joblib.dump(models["calibrator"], CALIBRATOR_ARTIFACT)

    write_validation_report(
        metrics={
            "auc_naive": auc_naive,
            "auc_lr": auc_lr,
            "auc_lgbm": auc_lgbm,
            "brier_lgbm": brier_lgbm,
            "lgbm_scores": models["lgbm_scores"],
        },
        split=split,
        by_segment=by_segment,
        top3_overall=top3_overall,
    )

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "split_rows": {
            "train": int(len(split.X_train)),
            "val": int(len(split.X_val)),
            "test": int(len(split.X_test)),
        },
        "metrics": {
            "auc_naive": auc_naive,
            "auc_lr": auc_lr,
            "auc_lgbm": auc_lgbm,
            "brier_lgbm": brier_lgbm,
            "top3_hit_rate": top3_overall,
        },
    }
    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved: {MODEL_ARTIFACT}")
    print(f"Saved: {SHAP_ARTIFACT}")
    print(f"Saved: {CALIBRATOR_ARTIFACT}")
    print(f"Saved: {VALIDATION_REPORT}")


if __name__ == "__main__":
    main()
