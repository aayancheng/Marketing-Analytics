# m04_mmm — Marketing Mix Model

Bayesian Marketing Mix Model measuring the incremental revenue impact of five paid media channels (TV, OOH, Print, Facebook, Search) using PyMC-Marketing.

## Key Results

| Metric | Value |
|--------|-------|
| Out-of-sample MAPE | 3.9% |
| Out-of-sample R² | 0.91 |
| Parameter recovery | 9/10 within 94% HDI |
| Channels | TV, OOH, Print, Facebook, Search |

## Quick Start

```bash
# Terminal 1: FastAPI (run from this directory)
cd m04_mmm
/Users/aayan/MarketingAnalytics/.venv/bin/python -m uvicorn src.api.main:app --port 8004 --reload

# Terminal 2: Express proxy
cd m04_mmm/app/server && npm start

# Terminal 3: Vite frontend
cd m04_mmm/app/client && npm run dev

# Open http://localhost:5176
```

## Architecture

```
React + Vite (5176) → Express Proxy (3004) → FastAPI (8004)
                                                  ↓
                                         models/precomputed/*.json
```

The MMM model object is not picklable. All results are pre-computed during training and served as static JSON files. The only real-time computation is the `/api/simulate` endpoint.

## Project Structure

```
m04_mmm/
├── data/
│   ├── synthetic/          # Raw generated data + true parameters
│   └── processed/          # Feature-engineered data
├── notebooks/
│   ├── 01_eda_mmm.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_diagnostics.ipynb
├── src/
│   ├── data_generator.py       # Synthetic data generation
│   ├── feature_engineering.py  # Trend + Fourier features
│   ├── train.py                # MCMC fitting + export
│   ├── export_precomputed.py   # JSON export functions
│   ├── budget_optimizer.py     # Scipy-based fallback optimizer
│   └── api/
│       ├── main.py             # FastAPI with 8 endpoints
│       └── schemas.py          # Pydantic v2 models
├── app/
│   ├── server/                 # Express proxy (port 3004)
│   └── client/                 # React + Vite (port 5176)
├── models/
│   ├── metadata.json
│   ├── mmm_trace/              # ArviZ InferenceData (NetCDF)
│   └── precomputed/            # 6 JSON files for API serving
└── docs/
    ├── model_card.md
    ├── model_documentation.md
    ├── executive_deck.md
    ├── research_brief.md
    ├── data_dictionary.md
    └── validation_report.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/api/decomposition` | GET | Weekly channel revenue decomposition |
| `/api/roas` | GET | ROAS with 94% credible intervals |
| `/api/response-curves` | GET | Spend-vs-contribution curves |
| `/api/adstock` | GET | Decay vectors per channel |
| `/api/optimal-allocation` | GET | Current vs. optimal budget split |
| `/api/metadata` | GET | Training metadata and metrics |
| `/api/simulate` | POST | Budget simulator (real-time) |

## Retraining

```bash
cd m04_mmm
/Users/aayan/MarketingAnalytics/.venv/bin/python src/train.py
```

Training takes ~10 minutes (MCMC sampling) and automatically exports all pre-computed JSON files.
