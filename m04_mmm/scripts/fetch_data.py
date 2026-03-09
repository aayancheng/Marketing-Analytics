#!/usr/bin/env python
"""Generate synthetic MMM dataset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import generate_mmm_data, save_data

if __name__ == "__main__":
    df = generate_mmm_data()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic")
    save_data(df, output_dir)
    print(f"Generated {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Revenue range: {df['revenue'].min():.0f} - {df['revenue'].max():.0f}")
    print(f"Saved to {output_dir}")
