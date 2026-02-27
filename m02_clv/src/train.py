"""
Model Training: m02_clv (Customer Lifetime Value)
- Trains LightGBM Regressor on log1p(clv_12m) target
- Compares against BG/NBD + Gamma-Gamma baseline and naive mean baseline
- Computes SHAP explanations
- Saves model artifacts and validation report

Usage:
    cd m02_clv
    /Users/aayan/MarketingAnalytics/.venv/bin/python src/train.py
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

warnings.filterwarnings("ignore", message=".*Pyarrow.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from src.feature_engineering import FEATURE_COLUMNS
except ModuleNotFoundError:
    from feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "training_features.csv")
OBSERVATION_CSV = os.path.join(BASE_DIR, "data", "processed", "observation_transactions.csv")
CLV_LABELS_CSV = os.path.join(BASE_DIR, "data", "processed", "clv_labels.csv")
CUSTOMER_FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")

MODEL_ARTIFACT = os.path.join(BASE_DIR, "models", "lgbm_clv.pkl")
SHAP_ARTIFACT = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
METADATA_ARTIFACT = os.path.join(BASE_DIR, "models", "metadata.json")
VALIDATION_REPORT = os.path.join(BASE_DIR, "docs", "validation_report.md")

# CLV segment thresholds (percentiles on predicted CLV)
SEGMENT_THRESHOLDS = {
    "Champions": 90,
    "High Value": 75,
    "Growing": 40,
    "Occasional": 10,
    "Dormant": 0,
}


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y_train_raw: pd.Series  # original scale (not log-transformed)
    y_test_raw: pd.Series


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def holdout_split(df: pd.DataFrame, test_size: float = 0.2) -> SplitData:
    """Stratified holdout split. Since we already have a temporal observation/prediction
    design, we split the training set into train/test for model evaluation."""
    print("[Split] Creating train/test split...")

    X = df[FEATURE_COLUMNS]
    y_raw = df["clv_12m"]

    # Stratify by CLV quintile to ensure balanced evaluation
    y_bins = pd.qcut(y_raw.rank(method="first"), q=5, labels=False)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=test_size, random_state=42, stratify=y_bins
    )

    # Log-transform target for training
    y_train = np.log1p(y_train_raw)
    y_test = np.log1p(y_test_raw)

    train_n = len(X_train)
    test_n = len(X_test)
    print(f"  Train: {train_n:,} customers")
    print(f"  Test:  {test_n:,} customers")

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_train_raw=y_train_raw.reset_index(drop=True),
        y_test_raw=y_test_raw.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def naive_baseline(split: SplitData) -> np.ndarray:
    """Predict mean CLV for every customer."""
    mean_clv = split.y_train_raw.mean()
    return np.full(len(split.y_test_raw), mean_clv)


def bgnbd_baseline(split: SplitData, X_train_full: pd.DataFrame, X_test_full: pd.DataFrame) -> np.ndarray:
    """BG/NBD + Gamma-Gamma baseline using lifetimes library."""
    print("\n[Baseline] Fitting BG/NBD + Gamma-Gamma...")
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter

        # lifetimes expects: frequency (repeat purchases), recency (days since first
        # to last purchase = tenure_days), T (days since first purchase to reference date)
        # Note: lifetimes "recency" = age at last purchase = tenure_days in our features
        # lifetimes "T" = total age = tenure_days + recency_days (where our recency_days
        # is days from last purchase to reference date)
        train_rfm = X_train_full[["frequency", "recency_days", "tenure_days", "monetary_avg"]].copy()
        train_rfm["T"] = train_rfm["tenure_days"] + train_rfm["recency_days"]
        train_rfm = train_rfm.rename(columns={
            "tenure_days": "recency",  # lifetimes recency = time between first and last purchase
            "monetary_avg": "monetary_value",
        })
        train_rfm = train_rfm.drop(columns=["recency_days"])
        # lifetimes frequency = number of REPEAT purchases (total - 1)
        train_rfm["frequency"] = (train_rfm["frequency"] - 1).clip(lower=0)
        # Only customers with frequency > 0 for Gamma-Gamma
        train_rfm_gg = train_rfm[train_rfm["frequency"] > 0].copy()

        # Fit BG/NBD
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(train_rfm["frequency"], train_rfm["recency"], train_rfm["T"])

        # Fit Gamma-Gamma
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(train_rfm_gg["frequency"], train_rfm_gg["monetary_value"])

        # Predict on test set
        test_rfm = X_test_full[["frequency", "recency_days", "tenure_days", "monetary_avg"]].copy()
        test_rfm["T"] = test_rfm["tenure_days"] + test_rfm["recency_days"]
        test_rfm = test_rfm.rename(columns={
            "tenure_days": "recency",
            "monetary_avg": "monetary_value",
        })
        test_rfm = test_rfm.drop(columns=["recency_days"])
        test_rfm["frequency"] = (test_rfm["frequency"] - 1).clip(lower=0)

        # Expected purchases in next 12 months (365 days)
        expected_purchases = bgf.predict(365, test_rfm["frequency"], test_rfm["recency"], test_rfm["T"])

        # Expected CLV = expected_purchases * expected_monetary
        expected_monetary = ggf.conditional_expected_average_profit(
            test_rfm["frequency"], test_rfm["monetary_value"]
        )
        # For customers with frequency=0, use overall mean monetary
        expected_monetary = expected_monetary.fillna(train_rfm_gg["monetary_value"].mean())

        pred_clv = (expected_purchases * expected_monetary).clip(lower=0)
        pred_clv = pred_clv.fillna(split.y_train_raw.mean())

        print(f"  BG/NBD fitted. Predicted CLV range: {pred_clv.min():.0f} - {pred_clv.max():.0f}")
        return pred_clv.values

    except ImportError:
        print("  lifetimes not installed. Using median baseline as BG/NBD proxy.")
        median_clv = split.y_train_raw.median()
        return np.full(len(split.y_test_raw), median_clv)
    except Exception as e:
        print(f"  BG/NBD fitting failed: {e}. Using mean baseline.")
        return naive_baseline(split)


# ---------------------------------------------------------------------------
# Primary model
# ---------------------------------------------------------------------------

def train_lgbm(split: SplitData) -> LGBMRegressor:
    """Train LightGBM Regressor on log1p(clv_12m)."""
    print("\n[LightGBM] Training regressor...")

    lgbm = LGBMRegressor(
        objective="regression",
        metric="mae",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42,
        n_estimators=300,
    )
    lgbm.fit(split.X_train, split.y_train)

    # Predictions in original scale
    y_pred_log = lgbm.predict(split.X_test)
    y_pred = np.expm1(y_pred_log).clip(min=0)

    # Metrics
    mae = mean_absolute_error(split.y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(split.y_test_raw, y_pred))

    # MAPE (exclude zero-CLV customers to avoid division by zero)
    nonzero_mask = split.y_test_raw > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(split.y_test_raw[nonzero_mask] - y_pred[nonzero_mask]) / split.y_test_raw[nonzero_mask]) * 100
    else:
        mape = 0.0

    spearman_r, spearman_p = spearmanr(split.y_test_raw, y_pred)

    print(f"  MAE:       {mae:,.2f}")
    print(f"  RMSE:      {rmse:,.2f}")
    print(f"  MAPE:      {mape:.1f}%")
    print(f"  Spearman:  {spearman_r:.4f} (p={spearman_p:.2e})")

    return lgbm


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]) * 100)
    else:
        mape = 0.0

    spearman_r_val, _ = spearmanr(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "spearman_r": float(spearman_r_val),
    }


def compute_decile_lift(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Compute top-decile lift and decile analysis table."""
    df = pd.DataFrame({"actual": y_true.values, "predicted": y_pred})
    df["decile"] = pd.qcut(df["predicted"].rank(method="first"), q=10, labels=range(1, 11))

    decile_table = df.groupby("decile").agg(
        count=("actual", "size"),
        mean_actual=("actual", "mean"),
        mean_predicted=("predicted", "mean"),
        total_actual=("actual", "sum"),
    ).reset_index()

    total_clv = df["actual"].sum()
    top_decile = decile_table[decile_table["decile"] == 10]
    top_decile_share = float(top_decile["total_actual"].values[0] / total_clv) if total_clv > 0 else 0
    top_decile_lift = top_decile_share / 0.1

    return top_decile_lift, decile_table


def assign_clv_segment(predicted_clv: np.ndarray) -> list[str]:
    """Assign CLV segment based on percentile thresholds."""
    percentiles = {
        "p90": np.percentile(predicted_clv, 90),
        "p75": np.percentile(predicted_clv, 75),
        "p40": np.percentile(predicted_clv, 40),
        "p10": np.percentile(predicted_clv, 10),
    }

    segments = []
    for val in predicted_clv:
        if val >= percentiles["p90"]:
            segments.append("Champions")
        elif val >= percentiles["p75"]:
            segments.append("High Value")
        elif val >= percentiles["p40"]:
            segments.append("Growing")
        elif val >= percentiles["p10"]:
            segments.append("Occasional")
        else:
            segments.append("Dormant")

    return segments


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def write_validation_report(
    metrics_lgbm: dict,
    metrics_naive: dict,
    metrics_bgnbd: dict,
    split: SplitData,
    top_decile_lift: float,
    decile_table: pd.DataFrame,
    feature_importance: list[tuple[str, float]],
) -> None:
    """Write validation report markdown."""
    print("\n[Report] Writing validation report...")

    # Decile table markdown
    decile_rows = []
    for _, row in decile_table.iterrows():
        decile_rows.append(
            f"| {int(row['decile'])} | {int(row['count'])} "
            f"| {row['mean_actual']:,.0f} | {row['mean_predicted']:,.0f} "
            f"| {row['total_actual']:,.0f} |"
        )
    decile_md = "\n".join(decile_rows)

    # Feature importance markdown
    feat_rows = []
    for fname, fval in feature_importance[:10]:
        feat_rows.append(f"| {fname} | {fval:.4f} |")
    feat_md = "\n".join(feat_rows)

    train_n = len(split.X_train)
    test_n = len(split.X_test)

    report = f"""# Validation Report — m02_clv (Customer Lifetime Value)

## Split Definition
- Training set: {train_n:,} customers (80% stratified by CLV quintile)
- Test set: {test_n:,} customers (20% holdout)
- Target: clv_12m (log1p-transformed for training, expm1 for evaluation)
- Temporal design: features from Dec 2009-Nov 2010, target from Dec 2010-Dec 2011

## Model Performance Comparison

| Model | MAE | RMSE | MAPE | Spearman r |
|-------|----:|-----:|-----:|-----------:|
| Naive Mean Baseline | {metrics_naive['mae']:,.0f} | {metrics_naive['rmse']:,.0f} | {metrics_naive['mape']:.1f}% | {metrics_naive['spearman_r']:.4f} |
| BG/NBD + Gamma-Gamma | {metrics_bgnbd['mae']:,.0f} | {metrics_bgnbd['rmse']:,.0f} | {metrics_bgnbd['mape']:.1f}% | {metrics_bgnbd['spearman_r']:.4f} |
| **LightGBM (22 features)** | **{metrics_lgbm['mae']:,.0f}** | **{metrics_lgbm['rmse']:,.0f}** | **{metrics_lgbm['mape']:.1f}%** | **{metrics_lgbm['spearman_r']:.4f}** |

## Top-Decile Lift
- Top decile captures {top_decile_lift:.1f}x its proportional share of total CLV

## Decile Analysis

| Decile | Count | Mean Actual | Mean Predicted | Total Actual |
|-------:|------:|------------:|---------------:|-------------:|
{decile_md}

## Top 10 Feature Importance (SHAP)

| Feature | Mean |SHAP| |
|---------|-----:|
{feat_md}

## Cold-Start Strategy
- Customers with < 2 purchases in observation window: use median CLV from training set
- Median CLV (training): {split.y_train_raw.median():,.0f}
"""

    os.makedirs(os.path.dirname(VALIDATION_REPORT), exist_ok=True)
    with open(VALIDATION_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {VALIDATION_REPORT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Model Training — m02_clv (Customer Lifetime Value)")
    print("=" * 60)

    # Load training data
    print("\n[Load] Reading training_features.csv...")
    train_df = pd.read_csv(TRAINING_FEATURES_CSV)
    n_rows = len(train_df)
    n_feats = len(FEATURE_COLUMNS)
    print(f"  Loaded {n_rows:,} customers, {n_feats} features + target")

    # Split
    split = holdout_split(train_df)

    # --- Naive baseline ---
    print("\n[Naive] Mean baseline...")
    naive_pred = naive_baseline(split)
    metrics_naive = compute_metrics(split.y_test_raw, naive_pred)
    print(f"  MAE: {metrics_naive['mae']:,.0f}, RMSE: {metrics_naive['rmse']:,.0f}")

    # --- BG/NBD baseline ---
    bgnbd_pred = bgnbd_baseline(split, split.X_train, split.X_test)
    metrics_bgnbd = compute_metrics(split.y_test_raw, bgnbd_pred)
    print(f"  MAE: {metrics_bgnbd['mae']:,.0f}, RMSE: {metrics_bgnbd['rmse']:,.0f}")

    # --- LightGBM primary ---
    lgbm = train_lgbm(split)

    y_pred_log = lgbm.predict(split.X_test)
    y_pred = np.expm1(y_pred_log).clip(min=0)
    metrics_lgbm = compute_metrics(split.y_test_raw, y_pred)

    # Decile analysis
    top_decile_lift, decile_table = compute_decile_lift(split.y_test_raw, y_pred)
    print(f"\n  Top-decile lift: {top_decile_lift:.2f}x")

    # --- SHAP ---
    print("\n[SHAP] Computing explanations...")
    bg = split.X_train.sample(min(500, len(split.X_train)), random_state=42)
    explainer = shap.TreeExplainer(lgbm, data=bg)

    # Feature importance from SHAP
    shap_values = explainer.shap_values(split.X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(FEATURE_COLUMNS, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )
    print("  Top 5 features:")
    for fname, fval in feature_importance[:5]:
        print(f"    {fname}: {fval:.4f}")

    # --- Save artifacts ---
    print("\n[Save] Saving model artifacts...")
    os.makedirs(os.path.dirname(MODEL_ARTIFACT), exist_ok=True)
    joblib.dump(lgbm, MODEL_ARTIFACT)
    print(f"  {MODEL_ARTIFACT}")

    joblib.dump(explainer, SHAP_ARTIFACT)
    print(f"  {SHAP_ARTIFACT}")

    # Metadata
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "split_rows": {
            "train": int(len(split.X_train)),
            "test": int(len(split.X_test)),
        },
        "metrics": {
            "lgbm": metrics_lgbm,
            "naive": metrics_naive,
            "bgnbd": metrics_bgnbd,
        },
        "top_decile_lift": top_decile_lift,
        "median_clv_train": float(split.y_train_raw.median()),
        "mean_clv_train": float(split.y_train_raw.mean()),
        "segment_thresholds": SEGMENT_THRESHOLDS,
    }
    with open(METADATA_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  {METADATA_ARTIFACT}")

    # Validation report
    write_validation_report(
        metrics_lgbm=metrics_lgbm,
        metrics_naive=metrics_naive,
        metrics_bgnbd=metrics_bgnbd,
        split=split,
        top_decile_lift=top_decile_lift,
        decile_table=decile_table,
        feature_importance=feature_importance,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Model: LightGBM Regressor (log1p target)")
    print(f"  Features: {n_feats}")
    print(f"  Training customers: {len(split.X_train):,}")
    print(f"  Test customers: {len(split.X_test):,}")
    print(f"  LightGBM MAE: {metrics_lgbm['mae']:,.0f}")
    print(f"  LightGBM RMSE: {metrics_lgbm['rmse']:,.0f}")
    print(f"  LightGBM MAPE: {metrics_lgbm['mape']:.1f}%")
    print(f"  Spearman r: {metrics_lgbm['spearman_r']:.4f}")
    print(f"  Top-decile lift: {top_decile_lift:.2f}x")
    print("\nDone.")


if __name__ == "__main__":
    main()
