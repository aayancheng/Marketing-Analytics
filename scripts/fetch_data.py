#!/usr/bin/env python3
"""
Fetch UCI Online Retail II dataset (id=502).
Saves combined CSV to data/raw/online_retail_ii.csv

Usage:
    python scripts/fetch_data.py
"""
import os
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_PATH = RAW_DIR / "online_retail_ii.zip"

if not ZIP_PATH.exists():
    print(f"Downloading from {URL}...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Download complete.")
else:
    print("Zip already exists, skipping download.")

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(RAW_DIR)
    print(f"Extracted: {z.namelist()}")

# Try to parse xlsx and save as CSV
try:
    import pandas as pd
    xlsx_files = list(RAW_DIR.glob("*.xlsx"))
    if xlsx_files:
        xlsx_path = xlsx_files[0]
        print(f"Reading {xlsx_path}...")
        df1 = pd.read_excel(xlsx_path, sheet_name=0, dtype={'Customer ID': str})
        df2 = pd.read_excel(xlsx_path, sheet_name=1, dtype={'Customer ID': str})
        df = pd.concat([df1, df2], ignore_index=True)
        csv_path = RAW_DIR / "online_retail_ii.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df):,} rows to {csv_path}")
except Exception as e:
    print(f"Could not parse xlsx: {e}")
    print("xlsx file is available in data/raw/")
