#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  if [[ -n "${VITE_PID:-}" ]]; then kill "$VITE_PID" >/dev/null 2>&1 || true; wait "$VITE_PID" 2>/dev/null || true; fi
  if [[ -n "${NODE_PID:-}" ]]; then kill "$NODE_PID" >/dev/null 2>&1 || true; wait "$NODE_PID" 2>/dev/null || true; fi
  if [[ -n "${API_PID:-}" ]]; then kill "$API_PID" >/dev/null 2>&1 || true; wait "$API_PID" 2>/dev/null || true; fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

wait_for_http() {
  local url="$1"
  local retries="${2:-60}"
  local i
  for i in $(seq 1 "$retries"); do
    if curl -sS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "Timeout waiting for $url" >&2
  return 1
}

cd "$ROOT_DIR"

echo "[1/4] Starting FastAPI..."
MPLCONFIGDIR=/tmp .venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000 >"$TMP_DIR/api.log" 2>&1 &
API_PID=$!
wait_for_http "http://127.0.0.1:8000/docs"

echo "[2/4] Starting Node proxy..."
node app/server/index.js >"$TMP_DIR/node.log" 2>&1 &
NODE_PID=$!
wait_for_http "http://127.0.0.1:3001/api/docs/model_card"

echo "[3/4] Starting Vite..."
(
  cd app/client
  npm run dev -- --host 127.0.0.1 --port 5173 >"$TMP_DIR/vite.log" 2>&1
) &
VITE_PID=$!
wait_for_http "http://127.0.0.1:5173/"

echo "[4/4] Running smoke checks..."

http_code() {
  local method="$1"
  local url="$2"
  local out="$3"
  local body="${4:-}"
  if [[ "$method" == "GET" ]]; then
    curl -sS -o "$out" -w '%{http_code}' "$url"
  else
    curl -sS -o "$out" -w '%{http_code}' -X "$method" -H 'Content-Type: application/json' -d "$body" "$url"
  fi
}

DOCS_CODE="$(http_code GET 'http://127.0.0.1:8000/docs' "$TMP_DIR/fdocs.html")"
CUST_CODE="$(http_code GET 'http://127.0.0.1:3001/api/customer/12747' "$TMP_DIR/cust.json")"
PRED_CODE="$(http_code POST 'http://127.0.0.1:3001/api/predict' "$TMP_DIR/pred.json" '{"recency_days":30,"frequency":10,"modal_hour":10,"purchase_hour_entropy":0.3}')"
INV_CODE="$(http_code GET 'http://127.0.0.1:3001/api/customer/not-a-number' "$TMP_DIR/inv.json")"
NF_CODE="$(http_code GET 'http://127.0.0.1:3001/api/customer/999999999' "$TMP_DIR/nf.json")"
LOW_CODE="$(http_code GET 'http://127.0.0.1:3001/api/customer/12346' "$TMP_DIR/low.json")"
CUSTS_CODE="$(http_code GET 'http://127.0.0.1:3001/api/customers?page=1&per_page=10&segment=Champions' "$TMP_DIR/customers.json")"
SEGS_CODE="$(http_code GET 'http://127.0.0.1:3001/api/segments' "$TMP_DIR/segments.json")"
DOC_CODE="$(http_code GET 'http://127.0.0.1:3001/api/docs/model_card' "$TMP_DIR/doc.json")"
VITE_CODE="$(http_code GET 'http://127.0.0.1:5173/' "$TMP_DIR/vite.html")"

export TMP_DIR DOCS_CODE CUST_CODE PRED_CODE INV_CODE NF_CODE LOW_CODE CUSTS_CODE SEGS_CODE DOC_CODE VITE_CODE

.venv/bin/python - <<'PY'
import json
import os
import sys

expected = {
    "DOCS_CODE": "200",
    "CUST_CODE": "200",
    "PRED_CODE": "200",
    "INV_CODE": "422",
    "NF_CODE": "404",
    "LOW_CODE": "400",
    "CUSTS_CODE": "200",
    "SEGS_CODE": "200",
    "DOC_CODE": "200",
    "VITE_CODE": "200",
}

for k, v in expected.items():
    actual = os.environ[k]
    if actual != v:
        print(f"[FAIL] {k}: expected {v}, got {actual}")
        sys.exit(1)

base = os.environ["TMP_DIR"]
with open(f"{base}/cust.json", "r", encoding="utf-8") as f:
    cust = json.load(f)
with open(f"{base}/pred.json", "r", encoding="utf-8") as f:
    pred = json.load(f)
with open(f"{base}/inv.json", "r", encoding="utf-8") as f:
    inv = json.load(f)
with open(f"{base}/nf.json", "r", encoding="utf-8") as f:
    nf = json.load(f)
with open(f"{base}/low.json", "r", encoding="utf-8") as f:
    low = json.load(f)
with open(f"{base}/customers.json", "r", encoding="utf-8") as f:
    customers = json.load(f)
with open(f"{base}/segments.json", "r", encoding="utf-8") as f:
    segments = json.load(f)
with open(f"{base}/doc.json", "r", encoding="utf-8") as f:
    doc = json.load(f)

if len(cust["heatmap"]) != 7 or len(cust["heatmap"][0]) != 24:
    print("[FAIL] customer heatmap shape is not 7x24")
    sys.exit(1)
if len(pred["heatmap"]) != 7 or len(pred["heatmap"][0]) != 24:
    print("[FAIL] predict heatmap shape is not 7x24")
    sys.exit(1)
if len(cust["top_3_slots"]) != 3:
    print("[FAIL] customer top_3_slots length != 3")
    sys.exit(1)
if len(pred["shap_values"]) != 5:
    print("[FAIL] predict shap_values length != 5")
    sys.exit(1)

if inv.get("detail", {}).get("error") != "invalid_customer_id":
    print("[FAIL] invalid_customer_id contract mismatch")
    sys.exit(1)
if nf.get("detail", {}).get("error") != "customer_not_found":
    print("[FAIL] customer_not_found contract mismatch")
    sys.exit(1)
if low.get("detail", {}).get("error") != "insufficient_history":
    print("[FAIL] insufficient_history contract mismatch")
    sys.exit(1)

if not customers.get("customers"):
    print("[FAIL] /api/customers returned empty list")
    sys.exit(1)
if not segments.get("segments"):
    print("[FAIL] /api/segments returned empty list")
    sys.exit(1)
if not doc.get("content"):
    print("[FAIL] /api/docs/model_card returned empty content")
    sys.exit(1)

print("[PASS] Integration smoke test passed.")
PY
