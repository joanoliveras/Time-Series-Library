#!/usr/bin/env python3
"""
Extract non-overlapping 5-minute windows (10 previous steps) for a single
experiment, based on the experiment JSON timestamps.

Directory layout (same as other analysis scripts):
  <root>/<approach>/<job>/preprocessed/*.csv   (metrics with 'date', 'cluster')
  <root>/<approach>/<job>/raw/<experiment>.json (has start_time/end_time as epoch seconds)

Usage example:
  python f2f_meeting/extract_experiment_windows.py \
        --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/f2f_meeting/comparison_of_approaches \
        --approach random_forest \
        --job job-0 \
        --experiment validation-random-forest-job-0_2025-09-18T230054_2025-09-19T070056_created_at_2025-09-19T070058.json \
        --out_dir /home/jolivera/Documents/CloudSkin/Time-Series-Library/f2f_meeting/comparison_of_approaches/random_forest/job-0/predictions
Notes:
  - Windows are created as non-overlapping blocks of 10 rows each (assuming
    ~30s step, that is 5 minutes). The first window covers steps [0..9], which
    corresponds to the period from experiment start up to the 5th minute.
  - If --cluster is provided, the script filters to that cluster before windowing
    (recommended to ensure 10 rows == 5 minutes). If omitted, windows are built
    over the combined stream (which may mix clusters, thus not guaranteeing
    10 rows equals 5 minutes).
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np


TARGET_COLUMN = 'pipelines_status_realtime_pipeline_latency'


@dataclass
class ExperimentWindow:
    start_epoch_seconds: float
    end_epoch_seconds: float
    source_file: str


def _load_preprocessed_metrics(preprocessed_dir: str) -> pd.DataFrame:
    csv_files = [
        os.path.join(preprocessed_dir, f)
        for f in os.listdir(preprocessed_dir)
        if f.endswith('.csv')
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in preprocessed dir: {preprocessed_dir}")

    dataframes: List[pd.DataFrame] = []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        dataframes.append(df)

    df_all = pd.concat(dataframes, ignore_index=True)
    for required in ('date', 'cluster'):
        if required not in df_all.columns:
            raise ValueError(f"Expected '{required}' in preprocessed CSVs under {preprocessed_dir}")
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.sort_values('date').reset_index(drop=True)


def _load_experiment_window(raw_dir: str, experiment_filename: str) -> ExperimentWindow:
    path = os.path.join(raw_dir, experiment_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Experiment JSON not found: {path}")
    with open(path, 'r') as f:
        meta = json.load(f)
    start_str = meta.get('start_time')
    end_str = meta.get('end_time')
    if start_str is None or end_str is None:
        raise ValueError(f"Missing start_time/end_time in {path}")
    try:
        start_epoch = float(start_str)
        end_epoch = float(end_str)
    except Exception as exc:
        raise ValueError(f"Invalid epoch in {path}: {exc}")
    return ExperimentWindow(start_epoch_seconds=start_epoch, end_epoch_seconds=end_epoch, source_file=os.path.basename(path))


def _slice_experiment(df_metrics: pd.DataFrame, window: ExperimentWindow) -> pd.DataFrame:
    start_time = pd.to_datetime(window.start_epoch_seconds, unit='s')
    end_time = pd.to_datetime(window.end_epoch_seconds, unit='s')
    df_exp = df_metrics[(df_metrics['date'] >= start_time) & (df_metrics['date'] <= end_time)].copy()
    df_exp = df_exp.sort_values('date').reset_index(drop=True)
    df_exp['step_index'] = np.arange(len(df_exp), dtype=int)
    return df_exp


def _save_window_csv(df_window: pd.DataFrame, out_dir: str, base_name: str, window_idx: int) -> str:
    if df_window.empty:
        return ''
    start_ts = df_window['date'].min()
    end_ts = df_window['date'].max()
    # Safer filenames: replace ':' with '-' in ISO timestamps
    start_str = start_ts.isoformat().replace(':', '-')
    end_str = end_ts.isoformat().replace(':', '-')
    out_path = os.path.join(
        out_dir,
        f"{base_name}__win{window_idx:04d}__{start_str}__to__{end_str}.csv"
    )
    df_window.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Extract non-overlapping 5-min (10-step) windows for a single experiment')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--approach', required=True, help='Approach subfolder (e.g., reactive, random_forest)')
    parser.add_argument('--job', default='job-0', help='Job subdirectory (default: job-0)')
    parser.add_argument('--experiment', required=True, help='Exact JSON filename in approach raw dir')
    parser.add_argument('--out_dir', required=True, help='Directory to write the window CSV files')
    parser.add_argument('--cluster', default=None, help='Optional cluster ID to filter before windowing')
    args = parser.parse_args()

    approach_path = os.path.join(args.root, args.approach)
    preprocessed_dir = os.path.join(approach_path, args.job, 'preprocessed')
    raw_dir = os.path.join(approach_path, args.job, 'raw')

    df_metrics = _load_preprocessed_metrics(preprocessed_dir)
    exp_window = _load_experiment_window(raw_dir, args.experiment)

    # Slice to experiment time range
    df_exp = _slice_experiment(df_metrics, exp_window)
    if args.cluster is not None:
        df_exp = df_exp[df_exp['cluster'].astype(str) == str(args.cluster)].reset_index(drop=True)
        df_exp['step_index'] = np.arange(len(df_exp), dtype=int)

    if df_exp.empty:
        raise RuntimeError('Selected experiment has no data after slicing (check timestamps/cluster filter).')

    # Create non-overlapping windows of length 10 steps: [0:10], [10:20], ...
    window_size = 10
    num_complete_windows = len(df_exp) // window_size
    if num_complete_windows == 0:
        raise RuntimeError(f"Not enough rows ({len(df_exp)}) to form a single 10-step window.")

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(exp_window.source_file)[0]

    saved_files: List[str] = []
    for w in range(num_complete_windows):
        end_idx = (w + 1) * window_size
        start_idx = end_idx - window_size
        df_w = df_exp.iloc[start_idx:end_idx].copy()
        out_path = _save_window_csv(df_w.drop(columns=['step_index']), args.out_dir, base_name, w + 1)
        if out_path:
            saved_files.append(out_path)

    print(f"Experiment: {exp_window.source_file}")
    print(f"Total rows in experiment slice: {len(df_exp)}")
    print(f"Window size (steps): {window_size}")
    print(f"Created {len(saved_files)} non-overlapping windows.")
    for p in saved_files:
        print(f"Saved: {p}")


if __name__ == '__main__':
    main()


