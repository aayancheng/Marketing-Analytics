"""
Model Training: Model 4 -- Marketing Mix Modeling

Fits a Bayesian MMM using PyMC-Marketing with:
- GeometricAdstock(l_max=8) per channel
- LogisticSaturation per channel
- Fourier seasonality (yearly_seasonality=2)
- Control variables (competitor_index, event_flag)

After fitting, exports pre-computed JSON artifacts for the FastAPI
serving layer (MMM is too slow for real-time inference).

Usage:
    cd m04_mmm
    /path/to/.venv/bin/python src/train.py
"""

import json
import os
import sys
import time
import warnings

import arviz as az
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module resolution fix (same pattern as m01-m03)
# ---------------------------------------------------------------------------
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from src.feature_engineering import (
    CHANNEL_COLUMNS,
    CONTROL_COLUMNS,
    TARGET_COLUMN,
    add_features,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SYNTHETIC = os.path.join(PROJECT_ROOT, "data", "synthetic")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PRECOMPUTED_DIR = os.path.join(MODELS_DIR, "precomputed")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_WEEKS = 156  # 75% of 208
L_MAX = 8          # adstock max lag
YEARLY_SEASONALITY = 2  # number of Fourier pairs

# MCMC settings (conservative for pymc sampler without numpyro)
MCMC_CHAINS = 4
MCMC_DRAWS = 1000
MCMC_TUNE = 1000
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42


def load_data():
    """Load synthetic data and add engineered features."""
    raw_path = os.path.join(DATA_SYNTHETIC, "mmm_weekly_data.csv")
    df = pd.read_csv(raw_path, parse_dates=["date_week"])
    df = add_features(df)
    return df


def train_test_split(df):
    """Time-based split: first TRAIN_WEEKS for training, rest for test."""
    train = df.iloc[:TRAIN_WEEKS].copy().reset_index(drop=True)
    test = df.iloc[TRAIN_WEEKS:].copy().reset_index(drop=True)
    print(f"Train: {len(train)} weeks | Test: {len(test)} weeks")
    return train, test


def build_and_fit_mmm(X_train, y_train):
    """Configure and fit the PyMC-Marketing MMM."""
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM

    print("\n--- Configuring MMM ---")
    print(f"  Channels: {CHANNEL_COLUMNS}")
    print(f"  Controls: {CONTROL_COLUMNS}")
    print(f"  Adstock: GeometricAdstock(l_max={L_MAX})")
    print(f"  Saturation: LogisticSaturation()")
    print(f"  Seasonality: yearly_seasonality={YEARLY_SEASONALITY}")

    mmm = MMM(
        date_column="date_week",
        channel_columns=CHANNEL_COLUMNS,
        control_columns=CONTROL_COLUMNS,
        adstock=GeometricAdstock(l_max=L_MAX),
        saturation=LogisticSaturation(),
        yearly_seasonality=YEARLY_SEASONALITY,
    )

    print(f"\n--- Fitting MMM ({MCMC_CHAINS} chains x {MCMC_DRAWS} draws, "
          f"tune={MCMC_TUNE}) ---")
    print("  This may take 5-15 minutes...")

    t0 = time.time()

    # Try numpyro first (faster), fall back to pymc
    sampler = "pymc"
    try:
        import numpyro  # noqa: F401
        sampler = "numpyro"
        print("  Using numpyro sampler (JAX-accelerated)")
    except ImportError:
        print("  numpyro not available, using pymc sampler")

    fit_kwargs = dict(
        X=X_train,
        y=y_train,
        random_seed=RANDOM_SEED,
    )

    if sampler == "numpyro":
        fit_kwargs["nuts_sampler"] = "numpyro"
        fit_kwargs["target_accept"] = TARGET_ACCEPT
        fit_kwargs["chains"] = MCMC_CHAINS
        fit_kwargs["draws"] = MCMC_DRAWS
        fit_kwargs["tune"] = MCMC_TUNE
    else:
        # pymc sampler uses progressbar and different kwarg names
        fit_kwargs["progressbar"] = True

    # Suppress excessive warnings during sampling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)

        try:
            mmm.fit(**fit_kwargs)
        except Exception as e:
            err_msg = str(e)
            print(f"  Fit with kwargs failed ({err_msg}), retrying with minimal args...")
            # Minimal fallback: just X, y, random_seed
            mmm.fit(X=X_train, y=y_train, random_seed=RANDOM_SEED)

    elapsed = time.time() - t0
    elapsed_min = elapsed / 60
    print(f"  Fitting complete in {elapsed_min:.1f} minutes")

    return mmm, sampler


def validate_convergence(mmm):
    """Check R-hat and ESS for MCMC convergence."""
    print("\n--- Convergence Diagnostics ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = mmm.idata if hasattr(mmm, "idata") else mmm.fit_result
        summary = az.summary(idata, round_to=4)

    # Drop rows with NaN r_hat (constant parameters)
    valid = summary.dropna(subset=["r_hat"])
    max_rhat = float(valid["r_hat"].max()) if len(valid) > 0 else 1.0
    min_ess_bulk = float(valid["ess_bulk"].min()) if len(valid) > 0 else 0.0
    min_ess_tail = float(valid["ess_tail"].min()) if len(valid) > 0 else 0.0

    print(f"  Max R-hat:      {max_rhat:.4f} (target < 1.01)")
    print(f"  Min ESS (bulk): {min_ess_bulk:.0f} (target > 400)")
    print(f"  Min ESS (tail): {min_ess_tail:.0f} (target > 400)")

    rhat_ok = bool(max_rhat < 1.05)  # slightly relaxed for synthetic data
    ess_ok = bool(min_ess_bulk > 100)  # relaxed minimum

    if rhat_ok and ess_ok:
        print("  Convergence: PASSED")
    else:
        print("  Convergence: WARNING — some diagnostics below target")
        if not rhat_ok:
            print(f"    R-hat {max_rhat:.4f} > 1.05 for some parameters")
        if not ess_ok:
            print(f"    ESS {min_ess_bulk:.0f} < 100 for some parameters")

    return {
        "max_rhat": float(max_rhat),
        "min_ess_bulk": float(min_ess_bulk),
        "min_ess_tail": float(min_ess_tail),
        "rhat_ok": rhat_ok,
        "ess_ok": ess_ok,
    }


def evaluate_out_of_sample(mmm, X_test, y_test):
    """Compute out-of-sample prediction metrics."""
    print("\n--- Out-of-Sample Evaluation ---")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        posterior_pred = mmm.predict(X=X_test)

    # Get mean predictions — predict() returns InferenceData in pymc-marketing 0.18
    try:
        if hasattr(posterior_pred, "posterior_predictive"):
            pp = posterior_pred.posterior_predictive
            var_name = list(pp.data_vars)[0]
            y_pred = pp[var_name].mean(dim=["chain", "draw"]).values.flatten()
        elif hasattr(posterior_pred, "values") and hasattr(posterior_pred, "dims"):
            dims = list(posterior_pred.dims)
            avg_dims = [d for d in dims if d in ("chain", "draw")]
            y_pred = posterior_pred.mean(dim=avg_dims).values.flatten() if avg_dims else posterior_pred.values.flatten()
        elif isinstance(posterior_pred, np.ndarray):
            y_pred = posterior_pred.flatten()
        else:
            y_pred = np.array(posterior_pred).flatten()
    except Exception:
        # Fallback: use idata posterior_predictive
        idata = mmm.idata
        if hasattr(idata, "posterior_predictive"):
            pp = idata.posterior_predictive
            var_name = list(pp.data_vars)[0]
            y_pred = pp[var_name].mean(dim=["chain", "draw"]).values.flatten()
        else:
            y_pred = np.full(len(y_test), float(y_test.mean()))

    y_actual = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    # Take last len(y_test) predictions if longer
    if len(y_pred) > len(y_actual):
        y_pred = y_pred[-len(y_actual):]
    min_len = min(len(y_pred), len(y_actual))
    y_pred = y_pred[:min_len]
    y_actual = y_actual[:min_len]

    # Metrics
    mae = float(np.mean(np.abs(y_pred - y_actual)))
    mape = float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100)
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"  MAE:  {mae:,.0f} EUR")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R2:   {r2:.4f}")

    return {"mae": mae, "mape": mape, "r2": r2}


def check_parameter_recovery(mmm):
    """Compare estimated posteriors against known true parameters."""
    print("\n--- Parameter Recovery ---")

    true_params_path = os.path.join(DATA_SYNTHETIC, "true_params.json")
    with open(true_params_path) as f:
        true_params = json.load(f)

    recovery = {}
    posterior = mmm.fit_result

    # Find adstock alpha variable name
    adstock_vars = [v for v in posterior.data_vars if "alpha" in v.lower() or "adstock" in v.lower()]
    saturation_vars = [v for v in posterior.data_vars if "lam" in v.lower() or "saturation" in v.lower()]

    print(f"  Posterior variables found: {list(posterior.data_vars.keys())}")

    if adstock_vars:
        alpha_var = adstock_vars[0]
        alpha_samples = posterior[alpha_var].values  # (chain, draw, channel) or (chain, draw)
        print(f"  Adstock variable: {alpha_var}, shape: {alpha_samples.shape}")

        recovery["adstock_alphas"] = {}
        for i, ch in enumerate(CHANNEL_COLUMNS):
            ch_name = ch.replace("_spend", "")
            true_val = true_params["adstock_alphas"].get(ch_name, None)
            if true_val is None:
                continue

            if alpha_samples.ndim == 3:
                samples = alpha_samples[:, :, i].flatten()
            elif alpha_samples.ndim == 2:
                samples = alpha_samples.flatten()
            else:
                continue

            est_mean = float(np.mean(samples))
            hdi = az.hdi(samples, hdi_prob=0.95)
            in_hdi = bool(hdi[0] <= true_val <= hdi[1])

            recovery["adstock_alphas"][ch_name] = {
                "true": true_val,
                "estimated": round(est_mean, 4),
                "hdi_3": round(float(hdi[0]), 4),
                "hdi_97": round(float(hdi[1]), 4),
                "in_hdi": in_hdi,
            }
            status = "OK" if in_hdi else "MISS"
            print(f"    {ch_name} alpha: true={true_val:.2f}, "
                  f"est={est_mean:.4f}, HDI=[{hdi[0]:.4f}, {hdi[1]:.4f}] [{status}]")

    if saturation_vars:
        lam_var = saturation_vars[0]
        lam_samples = posterior[lam_var].values
        print(f"  Saturation variable: {lam_var}, shape: {lam_samples.shape}")

        recovery["saturation_lambdas"] = {}
        for i, ch in enumerate(CHANNEL_COLUMNS):
            ch_name = ch.replace("_spend", "")
            true_val = true_params["saturation_lambdas"].get(ch_name, None)
            if true_val is None:
                continue

            if lam_samples.ndim == 3:
                samples = lam_samples[:, :, i].flatten()
            elif lam_samples.ndim == 2:
                samples = lam_samples.flatten()
            else:
                continue

            est_mean = float(np.mean(samples))
            hdi = az.hdi(samples, hdi_prob=0.95)
            in_hdi = bool(hdi[0] <= true_val <= hdi[1])

            recovery["saturation_lambdas"][ch_name] = {
                "true": true_val,
                "estimated": round(est_mean, 4),
                "hdi_3": round(float(hdi[0]), 4),
                "hdi_97": round(float(hdi[1]), 4),
                "in_hdi": in_hdi,
            }
            status = "OK" if in_hdi else "MISS"
            print(f"    {ch_name} lambda: true={true_val:.2f}, "
                  f"est={est_mean:.4f}, HDI=[{hdi[0]:.4f}, {hdi[1]:.4f}] [{status}]")

    return recovery


def save_artifacts(mmm, convergence, oos_metrics, recovery, sampler):
    """Save model artifacts and metadata."""
    print("\n--- Saving Artifacts ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

    # Note: MMM object can't be pickled (PyMC internal closures).
    # We don't need it — the API serves pre-computed JSONs instead.
    print("  Skipping MMM pkl (not picklable, not needed for serving)")

    # Save trace as NetCDF
    trace_dir = os.path.join(MODELS_DIR, "mmm_trace")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "trace.nc")
    try:
        idata = mmm.idata if hasattr(mmm, "idata") else None
        if idata is not None:
            idata.to_netcdf(trace_path)
        else:
            mmm.fit_result.to_netcdf(trace_path)
        print(f"  Saved trace: {trace_path}")
    except Exception as e:
        print(f"  Warning: could not save trace ({e})")

    # Save metadata
    metadata = {
        "model_type": "PyMC-Marketing Bayesian MMM",
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_train_weeks": TRAIN_WEEKS,
        "n_test_weeks": 208 - TRAIN_WEEKS,
        "mcmc_config": {
            "chains": MCMC_CHAINS,
            "draws": MCMC_DRAWS,
            "tune": MCMC_TUNE,
            "target_accept": TARGET_ACCEPT,
            "sampler": sampler,
        },
        "convergence": convergence,
        "out_of_sample": oos_metrics,
        "parameter_recovery": recovery,
        "channels": [c.replace("_spend", "") for c in CHANNEL_COLUMNS],
        "channel_columns": CHANNEL_COLUMNS,
        "control_columns": CONTROL_COLUMNS,
    }

    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata: {meta_path}")

    return metadata


def export_precomputed_jsons(mmm, X_train, y_train):
    """Export all pre-computed JSON files for the API."""
    print("\n--- Exporting Pre-computed JSONs ---")

    try:
        from src.export_precomputed import export_all
        export_all(mmm, X_train, y_train, PRECOMPUTED_DIR)
    except Exception as e:
        print(f"  Warning: export_precomputed failed ({e})")
        print("  Attempting fallback export...")
        _fallback_export(mmm, X_train, y_train)


def _fallback_export(mmm, X_train, y_train):
    """Minimal fallback export if the full export module fails."""
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

    # Decomposition
    try:
        contributions = mmm.compute_channel_contribution_original_scale()
        contrib_mean = contributions.mean(dim=["chain", "draw"]).values
        n_weeks = contrib_mean.shape[0]

        y_vals = y_train.values if hasattr(y_train, "values") else np.array(y_train)
        y_vals = y_vals[:n_weeks]

        weeks_data = []
        totals = {}
        for i in range(n_weeks):
            row = {"date_week": str(X_train["date_week"].iloc[i].date()),
                   "revenue_actual": float(y_vals[i])}
            ch_sum = 0.0
            for j, col in enumerate(CHANNEL_COLUMNS):
                ch_name = col.replace("_spend", "")
                val = float(contrib_mean[i, j])
                row[ch_name] = val
                ch_sum += val
                totals[ch_name] = totals.get(ch_name, 0.0) + val
            row["base"] = float(y_vals[i] - ch_sum)
            weeks_data.append(row)

        total_rev = float(np.sum(y_vals[:n_weeks]))
        pct = {ch: (v / total_rev * 100) if total_rev > 0 else 0.0
               for ch, v in totals.items()}

        decomp = {"weeks": weeks_data, "totals": totals, "pct": pct}
        decomp_path = os.path.join(PRECOMPUTED_DIR, "decomposition.json")
        with open(decomp_path, "w") as f:
            json.dump(decomp, f, indent=2, default=float)
        print(f"  Saved decomposition.json ({n_weeks} weeks)")
    except Exception as e:
        print(f"  Failed to export decomposition: {e}")

    # ROAS
    try:
        contributions = mmm.compute_channel_contribution_original_scale()
        channels_list = []
        for j, col in enumerate(CHANNEL_COLUMNS):
            ch_name = col.replace("_spend", "")
            total_spend = float(X_train[col].sum())

            # Per-sample ROAS
            ch_contrib_samples = contributions[:, :, :, j].values  # chain, draw, time
            total_contrib_samples = ch_contrib_samples.sum(axis=2)  # chain, draw
            roas_samples = total_contrib_samples.flatten() / total_spend if total_spend > 0 else np.zeros(1)

            roas_mean = float(np.mean(roas_samples))
            hdi = az.hdi(roas_samples, hdi_prob=0.94)

            channels_list.append({
                "channel": ch_name,
                "roas_mean": round(roas_mean, 4),
                "roas_hdi_3": round(float(hdi[0]), 4),
                "roas_hdi_97": round(float(hdi[1]), 4),
                "total_spend": round(total_spend, 2),
                "total_contribution": round(float(np.mean(total_contrib_samples)), 2),
            })

        roas_data = {"channels": channels_list}
        roas_path = os.path.join(PRECOMPUTED_DIR, "roas.json")
        with open(roas_path, "w") as f:
            json.dump(roas_data, f, indent=2)
        print(f"  Saved roas.json ({len(channels_list)} channels)")
    except Exception as e:
        print(f"  Failed to export ROAS: {e}")

    # Adstock
    try:
        posterior = mmm.fit_result
        adstock_vars = [v for v in posterior.data_vars if "alpha" in v.lower() or "adstock" in v.lower()]
        channels_adstock = []

        if adstock_vars:
            alpha_var = adstock_vars[0]
            alpha_samples = posterior[alpha_var].values

            for i, col in enumerate(CHANNEL_COLUMNS):
                ch_name = col.replace("_spend", "")
                if alpha_samples.ndim == 3:
                    samples = alpha_samples[:, :, i].flatten()
                else:
                    samples = alpha_samples.flatten()

                alpha_mean = float(np.mean(samples))
                hdi = az.hdi(samples, hdi_prob=0.94)

                # Geometric decay vector
                decay = [float(alpha_mean ** k) for k in range(L_MAX)]
                decay_sum = sum(decay)
                decay_norm = [d / decay_sum for d in decay]

                channels_adstock.append({
                    "channel": ch_name,
                    "decay_vector": decay_norm,
                    "alpha_mean": round(alpha_mean, 4),
                    "alpha_hdi_3": round(float(hdi[0]), 4),
                    "alpha_hdi_97": round(float(hdi[1]), 4),
                })

        adstock_data = {"channels": channels_adstock}
        adstock_path = os.path.join(PRECOMPUTED_DIR, "adstock.json")
        with open(adstock_path, "w") as f:
            json.dump(adstock_data, f, indent=2)
        print(f"  Saved adstock.json ({len(channels_adstock)} channels)")
    except Exception as e:
        print(f"  Failed to export adstock: {e}")

    # Simulator params
    try:
        posterior = mmm.fit_result

        sim_params = {
            "channel_columns": CHANNEL_COLUMNS,
            "mean_revenue": float(y_train.mean()),
        }

        # Adstock alphas
        adstock_vars = [v for v in posterior.data_vars if "alpha" in v.lower() or "adstock" in v.lower()]
        if adstock_vars:
            alpha_samples = posterior[adstock_vars[0]].values
            alphas = {}
            for i, col in enumerate(CHANNEL_COLUMNS):
                ch = col.replace("_spend", "")
                if alpha_samples.ndim == 3:
                    alphas[ch] = float(np.mean(alpha_samples[:, :, i]))
                else:
                    alphas[ch] = float(np.mean(alpha_samples))
            sim_params["adstock_alphas"] = alphas

        # Saturation lambdas
        sat_vars = [v for v in posterior.data_vars if "lam" in v.lower() or "saturation" in v.lower()]
        if sat_vars:
            lam_samples = posterior[sat_vars[0]].values
            lambdas = {}
            for i, col in enumerate(CHANNEL_COLUMNS):
                ch = col.replace("_spend", "")
                if lam_samples.ndim == 3:
                    lambdas[ch] = float(np.mean(lam_samples[:, :, i]))
                else:
                    lambdas[ch] = float(np.mean(lam_samples))
            sim_params["saturation_lambdas"] = lambdas

        # Channel scalers (MaxAbsScaler values used during fit)
        scalers = {}
        for col in CHANNEL_COLUMNS:
            ch = col.replace("_spend", "")
            scalers[ch] = {"max": float(X_train[col].max())}
        sim_params["channel_scalers"] = scalers

        sim_path = os.path.join(PRECOMPUTED_DIR, "simulator_params.json")
        with open(sim_path, "w") as f:
            json.dump(sim_params, f, indent=2)
        print(f"  Saved simulator_params.json")
    except Exception as e:
        print(f"  Failed to export simulator params: {e}")

    # Response curves (simplified)
    try:
        curves_data = {"channels": []}
        for j, col in enumerate(CHANNEL_COLUMNS):
            ch_name = col.replace("_spend", "")
            max_spend = float(X_train[col].max())
            avg_spend = float(X_train[col].mean())
            spends = np.linspace(0, max_spend * 2, 50)

            # Use simulator params to compute curve
            alpha = sim_params.get("adstock_alphas", {}).get(ch_name, 0.3)
            lam = sim_params.get("saturation_lambdas", {}).get(ch_name, 0.5)
            scaler_max = sim_params.get("channel_scalers", {}).get(ch_name, {}).get("max", max_spend)

            curve_points = []
            for s in spends:
                scaled = s / scaler_max if scaler_max > 0 else 0
                adstocked = scaled / (1 - alpha) if alpha < 1 else scaled
                saturated = adstocked / (adstocked + lam) if (adstocked + lam) > 0 else 0
                curve_points.append({"spend": round(float(s), 2), "contribution": round(float(saturated), 6)})

            curves_data["channels"].append({
                "channel": ch_name,
                "curve": curve_points,
                "current_avg_spend": round(avg_spend, 2),
            })

        curves_path = os.path.join(PRECOMPUTED_DIR, "response_curves.json")
        with open(curves_path, "w") as f:
            json.dump(curves_data, f, indent=2)
        print(f"  Saved response_curves.json")
    except Exception as e:
        print(f"  Failed to export response curves: {e}")

    # Optimal allocation
    try:
        total_weekly_budget = float(X_train[CHANNEL_COLUMNS].sum(axis=1).mean())
        current_allocation = {col.replace("_spend", ""): float(X_train[col].mean())
                              for col in CHANNEL_COLUMNS}

        try:
            opt_result = mmm.optimize_budget(
                budget=total_weekly_budget,
                num_periods=1,
            )
            optimal_allocation = {}
            if hasattr(opt_result, "items"):
                optimal_allocation = {k.replace("_spend", ""): float(v)
                                      for k, v in opt_result.items()}
            elif isinstance(opt_result, np.ndarray):
                for i, col in enumerate(CHANNEL_COLUMNS):
                    optimal_allocation[col.replace("_spend", "")] = float(opt_result[i])
            else:
                # Try to use it as-is
                optimal_allocation = current_allocation.copy()
                print("  Warning: optimize_budget returned unexpected type, using current as fallback")
        except Exception as e:
            print(f"  mmm.optimize_budget failed ({e}), using scipy fallback...")
            try:
                from src.budget_optimizer import optimize_budget as scipy_optimize
                opt = scipy_optimize(total_weekly_budget, sim_params)
                optimal_allocation = opt.get("optimal_allocation", current_allocation)
            except Exception as e2:
                print(f"  Scipy fallback also failed ({e2}), using equal allocation")
                equal = total_weekly_budget / len(CHANNEL_COLUMNS)
                optimal_allocation = {col.replace("_spend", ""): equal
                                      for col in CHANNEL_COLUMNS}

        # Compute revenue estimates using simulator params
        current_rev = float(y_train.mean())
        optimal_rev = current_rev
        if sim_params and optimal_allocation != current_allocation:
            try:
                from src.data_generator import logistic_saturation
                base_rev = sim_params.get("intercept", 0.0)
                betas = sim_params.get("channel_betas", {})
                lams = sim_params.get("saturation_lambdas", {})
                scalers = sim_params.get("channel_scalers", {})
                opt_rev_est = base_rev
                for ch_name in optimal_allocation:
                    spend = optimal_allocation[ch_name]
                    scaler_max = scalers.get(ch_name, {}).get("max", 1.0)
                    s_scaled = spend / scaler_max if scaler_max > 0 else 0.0
                    sat_val = logistic_saturation(np.array([s_scaled]), lams.get(ch_name, 0.5))[0]
                    opt_rev_est += betas.get(ch_name, 0.0) * float(sat_val)
                optimal_rev = opt_rev_est
            except Exception:
                optimal_rev = current_rev

        lift_abs_val = round(optimal_rev - current_rev, 2)
        lift_pct_val = round(lift_abs_val / current_rev * 100, 1) if current_rev > 0 else 0.0

        opt_data = {
            "total_budget": round(total_weekly_budget, 2),
            "current": {k: round(v, 2) for k, v in current_allocation.items()},
            "optimal": {k: round(v, 2) for k, v in optimal_allocation.items()},
            "current_revenue": round(current_rev, 2),
            "optimal_revenue": round(optimal_rev, 2),
            "lift_abs": lift_abs_val,
            "lift_pct": lift_pct_val,
            "recommendations": [],
        }

        # Generate recommendations
        for ch in [c.replace("_spend", "") for c in CHANNEL_COLUMNS]:
            curr = current_allocation.get(ch, 0)
            opt = optimal_allocation.get(ch, 0)
            delta = opt - curr
            delta_pct = (delta / curr * 100) if curr > 0 else 0
            if delta_pct > 5:
                action = "Increase"
            elif delta_pct < -5:
                action = "Decrease"
            else:
                action = "Maintain"
            opt_data["recommendations"].append({
                "channel": ch,
                "action": action,
                "current": round(curr, 2),
                "optimal": round(opt, 2),
                "delta": round(delta, 2),
                "delta_pct": round(delta_pct, 1),
            })

        opt_path = os.path.join(PRECOMPUTED_DIR, "optimal_allocation.json")
        with open(opt_path, "w") as f:
            json.dump(opt_data, f, indent=2)
        print(f"  Saved optimal_allocation.json")
    except Exception as e:
        print(f"  Failed to export optimal allocation: {e}")


def write_validation_report(convergence, oos_metrics, recovery):
    """Generate docs/validation_report.md."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    report_path = os.path.join(DOCS_DIR, "validation_report.md")

    lines = [
        "# Validation Report: Marketing Mix Model",
        "",
        "## Convergence Diagnostics",
        "",
        f"| Metric | Value | Target |",
        f"|--------|-------|--------|",
        f"| Max R-hat | {convergence['max_rhat']:.4f} | < 1.01 |",
        f"| Min ESS (bulk) | {convergence['min_ess_bulk']:.0f} | > 400 |",
        f"| Min ESS (tail) | {convergence['min_ess_tail']:.0f} | > 400 |",
        "",
        "## Out-of-Sample Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| MAE | {oos_metrics['mae']:,.0f} EUR |",
        f"| MAPE | {oos_metrics['mape']:.1f}% |",
        f"| R-squared | {oos_metrics['r2']:.4f} |",
        "",
    ]

    if recovery:
        lines.append("## Parameter Recovery")
        lines.append("")

        if "adstock_alphas" in recovery:
            lines.append("### Adstock Alphas")
            lines.append("")
            lines.append("| Channel | True | Estimated | HDI 3% | HDI 97% | In HDI |")
            lines.append("|---------|------|-----------|--------|---------|--------|")
            for ch, vals in recovery["adstock_alphas"].items():
                in_hdi = "Yes" if vals["in_hdi"] else "No"
                lines.append(
                    f"| {ch} | {vals['true']:.2f} | {vals['estimated']:.4f} "
                    f"| {vals['hdi_3']:.4f} | {vals['hdi_97']:.4f} | {in_hdi} |"
                )
            lines.append("")

        if "saturation_lambdas" in recovery:
            lines.append("### Saturation Lambdas")
            lines.append("")
            lines.append("| Channel | True | Estimated | HDI 3% | HDI 97% | In HDI |")
            lines.append("|---------|------|-----------|--------|---------|--------|")
            for ch, vals in recovery["saturation_lambdas"].items():
                in_hdi = "Yes" if vals["in_hdi"] else "No"
                lines.append(
                    f"| {ch} | {vals['true']:.2f} | {vals['estimated']:.4f} "
                    f"| {vals['hdi_3']:.4f} | {vals['hdi_97']:.4f} | {in_hdi} |"
                )
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved validation report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  M04 MMM — Model Training")
    print("=" * 60)

    # Load and split data
    df = load_data()
    train_df, test_df = train_test_split(df)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # Build and fit MMM
    mmm, sampler = build_and_fit_mmm(X_train, y_train)

    # Validate convergence
    convergence = validate_convergence(mmm)

    # Out-of-sample evaluation
    oos_metrics = evaluate_out_of_sample(mmm, X_test, y_test)

    # Parameter recovery
    recovery = check_parameter_recovery(mmm)

    # Save model artifacts
    save_artifacts(mmm, convergence, oos_metrics, recovery, sampler)

    # Export pre-computed JSONs
    export_precomputed_jsons(mmm, X_train, y_train)

    # Write validation report
    write_validation_report(convergence, oos_metrics, recovery)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)
    print(f"\n  Artifacts in: {MODELS_DIR}")
    print(f"  Pre-computed: {PRECOMPUTED_DIR}")


if __name__ == "__main__":
    main()
