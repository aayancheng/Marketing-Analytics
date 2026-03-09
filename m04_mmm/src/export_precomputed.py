"""
Export precomputed results from a fitted PyMC-Marketing MMM model.

Called from train.py after the model is fitted.  Each function extracts a
specific aspect of the model and saves it as a JSON file that the FastAPI
layer can serve without needing to load the model itself.

Compatible with pymc-marketing 0.18.2.
"""

import json
import os
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to Python-native types."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()[:10]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _save_json(data: Any, path: str) -> None:
    """Serialise *data* to JSON, converting numpy types along the way."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_json_serializable(data), f, indent=2)
    print(f"Saved {path}")


def _channel_short_name(col: str) -> str:
    """Strip '_spend' suffix to get a short channel name (e.g. 'tv')."""
    return col.replace("_spend", "")


# ---------------------------------------------------------------------------
# 1. Decomposition (weekly contributions)
# ---------------------------------------------------------------------------

def export_decomposition(mmm, X_train: pd.DataFrame, y_train: pd.Series,
                         output_path: str) -> None:
    """Extract weekly channel contributions and save to JSON.

    Output schema:
    {
        "weeks": [{"date_week": "YYYY-MM-DD", "revenue_actual": float,
                    "base": float, "<channel>": float, ...}],
        "totals": {"<channel>": float, ...},
        "pct": {"<channel>": float, ...}
    }
    """
    channel_cols = list(mmm.channel_columns)
    short_names = [_channel_short_name(c) for c in channel_cols]
    n_weeks = len(y_train)

    # Channel contributions: xarray DataArray (chain, draw, date, channel)
    contributions_xa = mmm.compute_channel_contribution_original_scale()
    # Average over posterior samples -> shape (date, channel)
    contrib_mean = contributions_xa.mean(dim=["chain", "draw"]).values

    # Actual revenue
    y_vals = np.asarray(y_train, dtype=np.float64)

    # Dates -- try several common sources
    try:
        date_col = mmm.date_column
        dates = pd.to_datetime(X_train[date_col]).tolist()
    except Exception:
        dates = pd.date_range(
            "2015-11-23", periods=n_weeks, freq="W-MON"
        ).tolist()

    # Base = actual revenue minus sum of channel contributions
    total_channel = contrib_mean.sum(axis=1)
    base = y_vals - total_channel

    # Build weekly records
    weeks: List[Dict[str, Any]] = []
    for i in range(n_weeks):
        date_str = dates[i]
        if hasattr(date_str, "isoformat"):
            date_str = date_str.isoformat()[:10]
        else:
            date_str = str(date_str)[:10]

        row: Dict[str, Any] = {
            "date_week": date_str,
            "revenue_actual": float(y_vals[i]),
            "base": float(base[i]),
        }
        for j, name in enumerate(short_names):
            row[name] = float(contrib_mean[i, j])
        weeks.append(row)

    # Totals and percentages
    total_revenue = float(y_vals.sum())
    totals: Dict[str, float] = {}
    pct: Dict[str, float] = {}
    for j, name in enumerate(short_names):
        ch_total = float(contrib_mean[:, j].sum())
        totals[name] = ch_total
        pct[name] = float(ch_total / total_revenue) if total_revenue != 0 else 0.0

    totals["base"] = float(base.sum())
    pct["base"] = float(base.sum() / total_revenue) if total_revenue != 0 else 0.0

    _save_json({"weeks": weeks, "totals": totals, "pct": pct}, output_path)


# ---------------------------------------------------------------------------
# 2. ROAS with HDI
# ---------------------------------------------------------------------------

def export_roas(mmm, X_train: pd.DataFrame, output_path: str) -> None:
    """Compute ROAS = total_contribution / total_spend per channel with HDI.

    Output schema:
    {
        "channels": [{"channel": str, "roas_mean": float,
                       "roas_hdi_3": float, "roas_hdi_97": float,
                       "total_spend": float, "total_contribution": float}]
    }
    """
    channel_cols = list(mmm.channel_columns)
    short_names = [_channel_short_name(c) for c in channel_cols]

    # contributions: (chain, draw, date, channel)
    contributions_xa = mmm.compute_channel_contribution_original_scale()
    contrib_vals = contributions_xa.values

    n_chains, n_draws, n_dates, n_channels = contrib_vals.shape

    # Flatten chain x draw -> (samples, date, channel)
    contrib_flat = contrib_vals.reshape(n_chains * n_draws, n_dates, n_channels)

    results: List[Dict[str, Any]] = []
    for j, (col, name) in enumerate(zip(channel_cols, short_names)):
        total_spend = float(X_train[col].sum())

        if total_spend <= 0:
            results.append({
                "channel": name,
                "roas_mean": 0.0,
                "roas_hdi_3": 0.0,
                "roas_hdi_97": 0.0,
                "total_spend": total_spend,
                "total_contribution": 0.0,
            })
            continue

        # Per-sample total contribution for this channel
        sample_total_contrib = contrib_flat[:, :, j].sum(axis=1)  # (samples,)
        sample_roas = sample_total_contrib / total_spend

        roas_mean = float(np.mean(sample_roas))
        # Use ArviZ hdi on the 1-D array
        hdi = az.hdi(np.asarray(sample_roas), hdi_prob=0.94)
        roas_hdi_3 = float(hdi[0])
        roas_hdi_97 = float(hdi[1])
        total_contribution = float(np.mean(sample_total_contrib))

        results.append({
            "channel": name,
            "roas_mean": roas_mean,
            "roas_hdi_3": roas_hdi_3,
            "roas_hdi_97": roas_hdi_97,
            "total_spend": total_spend,
            "total_contribution": total_contribution,
        })

    _save_json({"channels": results}, output_path)


# ---------------------------------------------------------------------------
# 3. Response curves
# ---------------------------------------------------------------------------

def export_response_curves(mmm, X_train: pd.DataFrame,
                           output_path: str) -> None:
    """Generate spend-vs-contribution curves for each channel.

    Uses a simplified approach: manually apply adstock and saturation
    transforms with posterior-mean parameters, then calibrate the beta
    coefficient from the model's actual channel contributions.

    Output schema:
    {
        "channels": [{"channel": str,
                       "curve": [{"spend": float, "contribution": float}],
                       "current_avg_spend": float}]
    }
    """
    from src.data_generator import geometric_adstock, logistic_saturation

    channel_cols = list(mmm.channel_columns)
    short_names = [_channel_short_name(c) for c in channel_cols]
    posterior = mmm.fit_result.posterior

    # Extract posterior mean parameters per channel
    try:
        adstock_alphas = posterior["adstock_alpha"].mean(
            dim=["chain", "draw"]
        ).values
    except KeyError:
        adstock_alphas = np.full(len(channel_cols), 0.5)

    try:
        sat_lams = posterior["saturation_lam"].mean(
            dim=["chain", "draw"]
        ).values
    except KeyError:
        sat_lams = np.full(len(channel_cols), 0.5)

    # Channel contributions for beta calibration
    contributions_xa = mmm.compute_channel_contribution_original_scale()
    contrib_mean = contributions_xa.mean(dim=["chain", "draw"]).values

    n_points = 50
    results: List[Dict[str, Any]] = []

    for j, (col, name) in enumerate(zip(channel_cols, short_names)):
        spend_series = X_train[col].values.astype(np.float64)
        max_spend = float(spend_series.max())
        avg_spend = float(spend_series.mean())
        alpha = float(adstock_alphas[j])
        lam = float(sat_lams[j])

        # Determine the channel scaler (MaxAbsScaler value) used during fit
        scaler_max = max_spend if max_spend > 0 else 1.0
        try:
            channel_transformer = mmm.channel_transformer
            if channel_transformer is not None:
                scaler_max = float(channel_transformer.scale_[j])
        except Exception:
            pass

        # Calibrate beta: apply same transforms to training data, then
        # beta = mean(contribution) / mean(saturated_adstocked_spend)
        adstocked = geometric_adstock(spend_series / scaler_max, alpha)
        saturated = logistic_saturation(adstocked, lam)
        mean_saturated = saturated.mean()
        mean_contrib = float(contrib_mean[:, j].mean())
        beta = mean_contrib / mean_saturated if mean_saturated > 0 else 0.0

        # Generate curve: for a single-period steady-state spend level,
        # adstock of constant spend x with normalised weights sums to x,
        # so contribution ~ beta * saturation(spend_scaled)
        spend_range = np.linspace(0, 2.0 * max_spend, n_points)
        curve: List[Dict[str, float]] = []
        for s in spend_range:
            s_scaled = s / scaler_max if scaler_max > 0 else 0.0
            sat_val = logistic_saturation(np.array([s_scaled]), lam)[0]
            contribution = beta * float(sat_val)
            curve.append({"spend": float(s), "contribution": contribution})

        results.append({
            "channel": name,
            "curve": curve,
            "current_avg_spend": avg_spend,
        })

    _save_json({"channels": results}, output_path)


# ---------------------------------------------------------------------------
# 4. Adstock decay vectors
# ---------------------------------------------------------------------------

def export_adstock(mmm, output_path: str) -> None:
    """Extract adstock decay vectors per channel.

    Output schema:
    {
        "channels": [{"channel": str, "decay_vector": [float],
                       "alpha_mean": float, "alpha_hdi_3": float,
                       "alpha_hdi_97": float}]
    }
    """
    channel_cols = list(mmm.channel_columns)
    short_names = [_channel_short_name(c) for c in channel_cols]
    posterior = mmm.fit_result.posterior

    # Determine l_max from the model's adstock configuration
    try:
        l_max = int(mmm.adstock.l_max)
    except Exception:
        l_max = 8

    try:
        alpha_posterior = posterior["adstock_alpha"].values  # (chain, draw, channel)
        alpha_flat = alpha_posterior.reshape(-1, alpha_posterior.shape[-1])
    except KeyError:
        alpha_flat = np.full((1, len(channel_cols)), 0.5)

    results: List[Dict[str, Any]] = []
    for j, name in enumerate(short_names):
        samples = alpha_flat[:, j]
        alpha_mean = float(np.mean(samples))

        hdi = az.hdi(np.asarray(samples), hdi_prob=0.94)
        alpha_hdi_3 = float(hdi[0])
        alpha_hdi_97 = float(hdi[1])

        # Decay vector using posterior mean alpha
        weights = np.array([alpha_mean ** i for i in range(l_max)])
        weights_norm = weights / weights.sum()

        results.append({
            "channel": name,
            "decay_vector": weights_norm.tolist(),
            "alpha_mean": alpha_mean,
            "alpha_hdi_3": alpha_hdi_3,
            "alpha_hdi_97": alpha_hdi_97,
        })

    _save_json({"channels": results}, output_path)


# ---------------------------------------------------------------------------
# 5. Simulator parameters (point estimates for /api/simulate)
# ---------------------------------------------------------------------------

def export_simulator_params(mmm, X_train: pd.DataFrame, y_train: pd.Series,
                            output_path: str) -> None:
    """Save point-estimate parameters needed for the /api/simulate endpoint.

    Output schema:
    {
        "intercept": float,
        "channel_betas": {"<channel>": float, ...},
        "adstock_alphas": {"<channel>": float, ...},
        "saturation_lambdas": {"<channel>": float, ...},
        "mean_revenue": float,
        "channel_columns": [str],
        "channel_scalers": {"<channel>": {"max": float}, ...}
    }
    """
    from src.data_generator import geometric_adstock, logistic_saturation

    channel_cols = list(mmm.channel_columns)
    short_names = [_channel_short_name(c) for c in channel_cols]
    posterior = mmm.fit_result.posterior

    # Intercept
    try:
        intercept = float(posterior["intercept"].mean().values)
    except KeyError:
        intercept = 0.0

    # Adstock alphas
    try:
        alpha_means = posterior["adstock_alpha"].mean(
            dim=["chain", "draw"]
        ).values
    except KeyError:
        alpha_means = np.full(len(channel_cols), 0.5)

    # Saturation lambdas
    try:
        lam_means = posterior["saturation_lam"].mean(
            dim=["chain", "draw"]
        ).values
    except KeyError:
        lam_means = np.full(len(channel_cols), 0.5)

    # Channel betas -- derive from decomposition
    contributions_xa = mmm.compute_channel_contribution_original_scale()
    contrib_mean = contributions_xa.mean(dim=["chain", "draw"]).values

    channel_betas: Dict[str, float] = {}
    adstock_alphas: Dict[str, float] = {}
    saturation_lambdas: Dict[str, float] = {}
    channel_scalers: Dict[str, Dict[str, float]] = {}

    for j, (col, name) in enumerate(zip(channel_cols, short_names)):
        alpha = float(alpha_means[j])
        lam = float(lam_means[j])
        adstock_alphas[name] = alpha
        saturation_lambdas[name] = lam

        spend = X_train[col].values.astype(np.float64)
        scaler_max = float(spend.max()) if spend.max() > 0 else 1.0

        # Try to read the actual scaler from the model
        try:
            channel_transformer = mmm.channel_transformer
            if channel_transformer is not None:
                scaler_max = float(channel_transformer.scale_[j])
        except Exception:
            pass

        channel_scalers[name] = {"max": scaler_max}

        # Derive beta: mean(contribution) / mean(saturated(adstocked(scaled_spend)))
        adstocked = geometric_adstock(spend / scaler_max, alpha)
        saturated = logistic_saturation(adstocked, lam)
        mean_saturated = saturated.mean()
        mean_contrib = float(contrib_mean[:, j].mean())
        beta = mean_contrib / mean_saturated if mean_saturated > 0 else 0.0
        channel_betas[name] = beta

    mean_revenue = float(np.mean(y_train))

    _save_json({
        "intercept": intercept,
        "channel_betas": channel_betas,
        "adstock_alphas": adstock_alphas,
        "saturation_lambdas": saturation_lambdas,
        "mean_revenue": mean_revenue,
        "channel_columns": short_names,
        "channel_scalers": channel_scalers,
    }, output_path)


# ---------------------------------------------------------------------------
# 6. Optimal budget allocation
# ---------------------------------------------------------------------------

def export_optimal_allocation(mmm, X_train: pd.DataFrame,
                              total_budget: float,
                              output_path: str) -> None:
    """Run budget optimizer and save results.

    Output schema:
    {
        "total_budget": float,
        "current": {"<channel>": float, ...},
        "optimal": {"<channel>": float, ...},
        "current_revenue": float,
        "optimal_revenue": float,
        "lift_abs": float,
        "lift_pct": float,
        "recommendations": [{"channel": str, "action": str,
                              "current": float, "optimal": float,
                              "delta": float, "delta_pct": float}]
    }
    """
    channel_cols = list(mmm.channel_columns)
    short_names = [_channel_short_name(c) for c in channel_cols]

    # Current average weekly allocation
    current_alloc: Dict[str, float] = {}
    for col, name in zip(channel_cols, short_names):
        current_alloc[name] = float(X_train[col].mean())

    # Run the optimizer
    try:
        opt_result = mmm.optimize_budget(budget=total_budget, num_periods=1)

        # opt_result may be a dict or a DataFrame depending on version
        if isinstance(opt_result, pd.DataFrame):
            optimal_alloc: Dict[str, float] = {}
            for col, name in zip(channel_cols, short_names):
                if col in opt_result.columns:
                    optimal_alloc[name] = float(opt_result[col].iloc[0])
                elif name in opt_result.columns:
                    optimal_alloc[name] = float(opt_result[name].iloc[0])
                else:
                    optimal_alloc[name] = current_alloc[name]
        elif isinstance(opt_result, dict):
            optimal_alloc = {}
            for col, name in zip(channel_cols, short_names):
                if col in opt_result:
                    optimal_alloc[name] = float(opt_result[col])
                elif name in opt_result:
                    optimal_alloc[name] = float(opt_result[name])
                else:
                    optimal_alloc[name] = current_alloc[name]
        else:
            # Try treating as array-like
            optimal_alloc = {
                name: float(val)
                for name, val in zip(
                    short_names, np.asarray(opt_result).flatten()
                )
            }
    except Exception as e:
        print(f"Budget optimizer failed: {e}")
        print("Using current allocation as optimal (no optimization applied).")
        optimal_alloc = dict(current_alloc)

    # Estimate revenues using channel contributions
    contributions_xa = mmm.compute_channel_contribution_original_scale()
    contrib_mean = contributions_xa.mean(dim=["chain", "draw"]).values

    current_revenue = float(contrib_mean.sum())

    # Estimate optimal revenue proportionally.
    # Use square-root scaling as a saturation-aware heuristic when the
    # optimizer does not directly return a revenue estimate.
    optimal_revenue = 0.0
    for j, name in enumerate(short_names):
        cur = current_alloc[name]
        opt = optimal_alloc[name]
        ch_contrib = float(contrib_mean[:, j].sum())
        if cur > 0:
            ratio = opt / cur
            scaled = ch_contrib * np.sqrt(ratio) if ratio > 0 else 0.0
            optimal_revenue += scaled
        else:
            optimal_revenue += ch_contrib

    lift_abs = optimal_revenue - current_revenue
    lift_pct = (
        float(lift_abs / current_revenue * 100)
        if current_revenue != 0
        else 0.0
    )

    # Build recommendations
    recommendations: List[Dict[str, Any]] = []
    for name in short_names:
        cur = current_alloc[name]
        opt = optimal_alloc[name]
        delta = opt - cur
        delta_pct = float(delta / cur * 100) if cur != 0 else 0.0

        if delta_pct > 5:
            action = "increase"
        elif delta_pct < -5:
            action = "decrease"
        else:
            action = "maintain"

        recommendations.append({
            "channel": name,
            "action": action,
            "current": cur,
            "optimal": opt,
            "delta": delta,
            "delta_pct": delta_pct,
        })

    _save_json({
        "total_budget": total_budget,
        "current": current_alloc,
        "optimal": optimal_alloc,
        "current_revenue": current_revenue,
        "optimal_revenue": optimal_revenue,
        "lift_abs": lift_abs,
        "lift_pct": lift_pct,
        "recommendations": recommendations,
    }, output_path)


# ---------------------------------------------------------------------------
# Master export function
# ---------------------------------------------------------------------------

def export_all(mmm, X_train: pd.DataFrame, y_train: pd.Series,
               output_dir: str) -> None:
    """Run all six export functions and write JSON files to *output_dir*.

    Files created:
        decomposition.json
        roas.json
        response_curves.json
        adstock.json
        simulator_params.json
        optimal_allocation.json
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting precomputed results to {output_dir}")

    export_decomposition(
        mmm, X_train, y_train,
        os.path.join(output_dir, "decomposition.json"),
    )

    export_roas(
        mmm, X_train,
        os.path.join(output_dir, "roas.json"),
    )

    export_response_curves(
        mmm, X_train,
        os.path.join(output_dir, "response_curves.json"),
    )

    export_adstock(
        mmm,
        os.path.join(output_dir, "adstock.json"),
    )

    export_simulator_params(
        mmm, X_train, y_train,
        os.path.join(output_dir, "simulator_params.json"),
    )

    # Total budget = sum of current average weekly allocation
    channel_cols = list(mmm.channel_columns)
    total_budget = float(X_train[channel_cols].mean().sum())

    export_optimal_allocation(
        mmm, X_train, total_budget,
        os.path.join(output_dir, "optimal_allocation.json"),
    )

    print("All exports complete.")
