import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import SimulateRequest, SimulateResponse

CHANNELS = ["tv", "ooh", "print", "facebook", "search"]
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
PRECOMPUTED_DIR = MODEL_DIR / "precomputed"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all precomputed JSON files at startup
    app.state.decomposition = _load_json(PRECOMPUTED_DIR / "decomposition.json")
    app.state.roas = _load_json(PRECOMPUTED_DIR / "roas.json")
    app.state.response_curves = _load_json(PRECOMPUTED_DIR / "response_curves.json")
    app.state.adstock = _load_json(PRECOMPUTED_DIR / "adstock.json")
    app.state.simulator_params = _load_json(PRECOMPUTED_DIR / "simulator_params.json")
    app.state.optimal_allocation = _load_json(PRECOMPUTED_DIR / "optimal_allocation.json")
    app.state.metadata = _load_json(MODEL_DIR / "metadata.json")

    # Pre-compute simulator calibration values
    _calibrate_simulator(app)

    print("MMM API ready — all precomputed data loaded")
    yield


def _calibrate_simulator(app: FastAPI):
    """Derive effective_beta per channel for the simulator."""
    decomp = app.state.decomposition
    sim_params = app.state.simulator_params
    n_weeks = len(decomp["weeks"])

    alphas = sim_params["adstock_alphas"]
    lambdas = sim_params["saturation_lambdas"]
    scalers = sim_params["channel_scalers"]

    # Average weekly contribution per channel from decomposition totals
    avg_contributions = {}
    for ch in CHANNELS:
        total_contrib = decomp["totals"].get(ch, 0)
        avg_contributions[ch] = total_contrib / n_weeks if n_weeks > 0 else 0

    # Average weekly x_sat at current spend levels
    optimal_alloc = app.state.optimal_allocation
    current_spends = optimal_alloc.get("current", {})

    effective_betas = {}
    for ch in CHANNELS:
        current_spend = current_spends.get(ch, 0)
        x_scaled = current_spend / scalers[ch]["max"] if scalers[ch]["max"] > 0 else 0
        alpha = min(alphas.get(ch, 0), 0.99)
        x_adstock = x_scaled / (1 - alpha)
        lam = lambdas.get(ch, 1)
        x_sat = x_adstock / (x_adstock + lam) if (x_adstock + lam) > 0 else 0

        if x_sat > 0:
            effective_betas[ch] = avg_contributions[ch] / x_sat
        else:
            effective_betas[ch] = 0

    app.state.effective_betas = effective_betas

    # Base revenue = mean of weekly base values
    base_values = [w.get("base", 0) for w in decomp["weeks"]]
    app.state.base_revenue = sum(base_values) / len(base_values) if base_values else 0


app = FastAPI(title="MMM Analytics API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "mmm", "channels": CHANNELS}


# ── Pre-computed data endpoints ─────────────────────────────────────────

@app.get("/api/decomposition")
def get_decomposition():
    return app.state.decomposition


@app.get("/api/roas")
def get_roas():
    return app.state.roas


@app.get("/api/response-curves")
def get_response_curves():
    return app.state.response_curves


@app.get("/api/adstock")
def get_adstock():
    return app.state.adstock


@app.get("/api/optimal-allocation")
def get_optimal_allocation():
    return app.state.optimal_allocation


@app.get("/api/metadata")
def get_metadata():
    return app.state.metadata


# ── Simulator ───────────────────────────────────────────────────────────

@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    sim_params = app.state.simulator_params
    alphas = sim_params["adstock_alphas"]
    lambdas = sim_params["saturation_lambdas"]
    scalers = sim_params["channel_scalers"]
    betas = app.state.effective_betas
    base_revenue = app.state.base_revenue

    spend_map = {
        "tv": req.tv_spend,
        "ooh": req.ooh_spend,
        "print": req.print_spend,
        "facebook": req.facebook_spend,
        "search": req.search_spend,
    }

    channel_contributions = {}
    saturation_warnings = []

    for ch in CHANNELS:
        spend = spend_map[ch]
        max_val = scalers[ch]["max"]
        x_scaled = spend / max_val if max_val > 0 else 0

        alpha = min(alphas.get(ch, 0), 0.99)
        x_adstock = x_scaled / (1 - alpha)

        lam = lambdas.get(ch, 1)
        x_sat = x_adstock / (x_adstock + lam) if (x_adstock + lam) > 0 else 0

        channel_contributions[ch] = betas[ch] * x_sat

        # Saturation warnings
        if x_sat > 0.8:
            saturation_warnings.append({"channel": ch, "level": "high"})
        elif x_sat > 0.6:
            saturation_warnings.append({"channel": ch, "level": "moderate"})

    predicted_revenue = base_revenue + sum(channel_contributions.values())
    current_revenue = app.state.optimal_allocation.get("current_revenue", 0)
    delta = predicted_revenue - current_revenue
    delta_pct = (delta / current_revenue * 100) if current_revenue else 0

    return SimulateResponse(
        predicted_revenue=round(predicted_revenue, 2),
        current_revenue=round(current_revenue, 2),
        delta=round(delta, 2),
        delta_pct=round(delta_pct, 2),
        channel_contributions={ch: round(v, 2) for ch, v in channel_contributions.items()},
        saturation_warnings=saturation_warnings,
    )
