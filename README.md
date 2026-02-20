# Model 1 — Optimal Time-to-Engage Analytics

Email send-time optimization using LightGBM, FastAPI, and React.

## Setup

**Python (requires Python 3.11–3.13):**
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/fetch_data.py  # download UCI data to data/raw/
```

**Node.js:**
```bash
npm install  # installs all workspaces
```

## Running

```bash
# 1. FastAPI (Python ML service)
source .venv/bin/activate
uvicorn src.api.main:app --port 8000 --reload

# 2. Node backend proxy
node app/server/index.js

# 3. React frontend
cd app/client && npm run dev
```

## Data

Uses UCI Online Retail II (id=502). Run `python scripts/fetch_data.py` to download.
Dataset: Chen, D. (2012). Online Retail II. UCI ML Repository. https://doi.org/10.24432/C5CG6D
