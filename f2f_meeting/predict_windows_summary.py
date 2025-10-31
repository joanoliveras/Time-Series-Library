#!/usr/bin/env python3
"""
Run Random Forest inference per window CSV and write a single summary CSV with:
  - source_file: input filename
  - last_input_timestamp: max 'date' timestamp in input file (ISO)
  - prediction_mean: mean(preds) over returned prediction horizon

Assumes the Random Forest checkpoint exists at:
  <checkpoints>/<model_id>/model.joblib

Example:
  python f2f_meeting/predict_windows_summary.py \
    --windows_dir /home/jolivera/Documents/CloudSkin/Time-Series-Library/f2f_meeting/comparison_of_approaches/random_forest/job-0/predictions \
    --output_csv /home/jolivera/Documents/CloudSkin/Time-Series-Library/f2f_meeting/comparison_of_approaches/random_forest/job-0/predictions/results/all_predictions.csv \
    --checkpoints /home/jolivera/Documents/CloudSkin/Time-Series-Library/checkpoints \
    --model_id random_forest_n_estimators_100_max_depth_10 \
    --seq_len 10 \
    --pred_len 10 \
    --target pipelines_status_realtime_pipeline_latency
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import sys
import pathlib

# Ensure project root is on sys.path when running as a script via path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.exp_random_forest import Exp_Random_Forest


@dataclass
class ArgsRF:
    checkpoints: str
    model_id: str
    pred_len: int
    seq_len: int
    root_path: str
    data_path: str


def _list_csvs(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description='Per-file RF inference to combined summary CSV')
    parser.add_argument('--windows_dir', required=True, help='Directory containing input window CSVs')
    parser.add_argument('--output_csv', required=True, help='Path to write combined summary CSV')
    parser.add_argument('--checkpoints', default='./checkpoints', help='Checkpoints root directory')
    parser.add_argument('--model_id', required=True, help='Model ID directory under checkpoints (contains model.joblib)')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length used for feature aggregation')
    parser.add_argument('--pred_len', type=int, default=10, help='Prediction length returned by model')
    parser.add_argument('--target', default='pipelines_status_realtime_pipeline_latency', help='Target column name (for reference only)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Prepare args object for Exp_Random_Forest. We'll update root_path/data_path per file.
    rf_args = ArgsRF(
        checkpoints=args.checkpoints,
        model_id=args.model_id,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        root_path=args.windows_dir,
        data_path='',
    )

    # Load model once
    exp = Exp_Random_Forest(rf_args)

    rows: List[Tuple[str, str, float]] = []

    csv_files = _list_csvs(args.windows_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.windows_dir}")

    for fname in csv_files:
        csv_path = os.path.join(args.windows_dir, fname)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping file due to read error {fname}: {e}")
            continue
        if 'date' not in df.columns:
            print(f"Skipping file missing required column 'date': {fname}")
            continue

        # Latest timestamp in input
        try:
            df['date'] = pd.to_datetime(df['date'])
            last_ts = df['date'].max()
            last_ts_str = last_ts.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            # Fallback: store as raw string from the last row
            last_ts_str = str(df.iloc[-1]['date'])

        # Current cluster name from the last row (if present)
        try:
            cluster_name = str(df['cluster'].iloc[-1]) if 'cluster' in df.columns else ''
        except Exception:
            cluster_name = ''

        # Update dataset pointers and predict
        exp.args.root_path = args.windows_dir
        exp.args.data_path = fname
        preds = exp.predict(setting='')  # shape: (1, pred_len, 1)

        # Mean over the returned horizon
        try:
            pred_mean = float(np.asarray(preds).mean())
        except Exception:
            pred_mean = float('nan')

        rows.append((cluster_name, last_ts_str, pred_mean))

    # Build output dataframe
    out_df = pd.DataFrame(rows, columns=['cluster', 'last_input_timestamp', 'prediction_mean'])
    # Sort by timestamp if possible
    try:
        out_df['last_input_timestamp'] = pd.to_datetime(out_df['last_input_timestamp'])
        out_df = out_df.sort_values('last_input_timestamp')
        out_df['last_input_timestamp'] = out_df['last_input_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    except Exception:
        pass

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved summary CSV: {args.output_csv} ({len(out_df)} rows)")


if __name__ == '__main__':
    main()


