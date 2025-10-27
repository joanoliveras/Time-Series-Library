#!/usr/bin/env python3
"""
Compute per-experiment metrics for each approach, then average them across
repeated experiments to obtain an approach-level summary.

This differs from aggregating experiments first: here we compute metrics on
each individual experiment (run) and then average those metrics per approach.

Expected input layout (same as aggregate_experiments_by_time_delta.py):
  <root>/<approach>/<job>/preprocessed/*.csv  (metrics with a 'date' column)
  <root>/<approach>/<job>/raw/*.json          (each JSON has start_time/end_time)

Usage:
  python compare_approaches_per_run_then_average.py \
    --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
    --job job-0 \
    [--approach informer] \
    [--sla 0.2] \
    [--per_experiment_out result_analysis/per_experiment_metrics.csv] \
    [--approach_summary_out result_analysis/approach_per_run_averaged_summary.csv]
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExperimentWindow:
    start_epoch_seconds: float
    end_epoch_seconds: float
    source_file: str


def _discover_approaches(root: str) -> List[str]:
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


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
    if 'date' not in df_all.columns:
        raise ValueError("Expected 'date' column in preprocessed CSVs")
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.sort_values('date').reset_index(drop=True)


def _load_experiment_windows(raw_dir: str) -> List[ExperimentWindow]:
    json_files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith('.json')
    ]
    if not json_files:
        raise FileNotFoundError(f"No JSON experiment descriptors found in raw dir: {raw_dir}")

    experiments: List[ExperimentWindow] = []
    for path in sorted(json_files):
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
        experiments.append(ExperimentWindow(start_epoch_seconds=start_epoch, end_epoch_seconds=end_epoch, source_file=os.path.basename(path)))

    experiments.sort(key=lambda e: e.start_epoch_seconds)
    return experiments


def _slice_experiment(df_metrics: pd.DataFrame, window: ExperimentWindow) -> pd.DataFrame:
    start_time = pd.to_datetime(window.start_epoch_seconds, unit='s')
    end_time = pd.to_datetime(window.end_epoch_seconds, unit='s')
    df_exp = df_metrics[(df_metrics['date'] >= start_time) & (df_metrics['date'] <= end_time)].copy()
    df_exp = df_exp.sort_values('date').reset_index(drop=True)
    df_exp['step_index'] = np.arange(len(df_exp), dtype=int)
    df_exp['time_since_start_seconds'] = (df_exp['date'] - start_time).dt.total_seconds()
    return df_exp


def _summarize_single_experiment(df_exp: pd.DataFrame, sla: float) -> Dict[str, float]:
    """Compute metrics for a single experiment's time series of latency.

    Metrics:
      - num_steps: total steps in this experiment
      - overall_mean_latency: mean of per-step latency across entire run
      - peak_latency: maximum per-step latency across entire run
      - num_times_crossed_above_sla: rising-edge crossings of SLA threshold
      - total_seconds_above_sla_edge_cluster: sum of dt where latency >= SLA in edge cluster
      - total_seconds_above_sla_cloud_cluster: sum of dt where latency >= SLA in cloud cluster
      - total_seconds_in_cloud_cluster: sum of dt spent in non-edge cluster
      - percent_time_below_sla: percentage of total dt where latency < SLA
    """
    if df_exp.empty:
        return {
            'num_steps': 0,
            'overall_mean_latency': float('nan'),
            'peak_latency': float('nan'),
            'num_times_crossed_above_sla': 0,
            'total_seconds_above_sla_edge_cluster': 0.0,
            'total_seconds_above_sla_cloud_cluster': 0.0,
            'total_seconds_in_cloud_cluster': 0.0,
            'percent_time_below_sla': float('nan'),
        }

    latency = df_exp['pipelines_status_realtime_pipeline_latency'].astype(float)
    overall_mean_latency = float(latency.mean())
    peak_latency = float(latency.max())

    above = latency >= sla
    crossings = int((above.astype(int).diff().fillna(0) == 1).sum())

    # dt to next step based on time_since_start_seconds
    t = df_exp['time_since_start_seconds'].astype(float)
    dt_next = t.shift(-1) - t
    dt_next = dt_next.fillna(0.0)
    total_duration_seconds = float(dt_next.sum())
    # Only count duration above SLA for the edge cluster rows
    EDGE_CLUSTER_ID = 'eb0e3eaa-b668-4ad6-bc10-2bb0eb7da259'
    CLOUD_CLUSTER_ID = 'fd7816db-7948-4602-af7a-1d51900792a7'
    # Count migrations between clusters
    number_migrations_edge_to_cloud = 0
    number_migrations_cloud_to_edge = 0
    if 'cluster' in df_exp.columns:
        cluster_series = df_exp['cluster'].astype(str)
        cluster_shift = cluster_series.shift(1)
        transitions = cluster_series != cluster_shift

        # Identify direction of transitions
        number_migrations_edge_to_cloud = int(((cluster_shift == EDGE_CLUSTER_ID) & (cluster_series == CLOUD_CLUSTER_ID) & transitions).sum())
        number_migrations_cloud_to_edge = int(((cluster_shift == CLOUD_CLUSTER_ID) & (cluster_series == EDGE_CLUSTER_ID) & transitions).sum())

    if 'cluster' in df_exp.columns:
        in_edge_cluster = (df_exp['cluster'] == EDGE_CLUSTER_ID)
        in_cloud_cluster = (df_exp['cluster'] == CLOUD_CLUSTER_ID)
    else:
        # Fallback to previous behavior if no cluster info is present
        in_edge_cluster = pd.Series([True] * len(df_exp), index=df_exp.index)

    total_seconds_above_sla_edge_cluster = float((((above) & in_edge_cluster).astype(float) * dt_next).sum())
    total_seconds_above_sla_cloud_cluster = float((((above) & in_cloud_cluster).astype(float) * dt_next).sum())

    # Time attribution by cluster
    seconds_in_edge_cluster = float((in_edge_cluster.astype(float) * dt_next).sum())
    total_seconds_in_cloud_cluster = float(max(0.0, total_duration_seconds - seconds_in_edge_cluster))

    # Percentage of time below SLA over the entire experiment duration
    below = (latency < sla)
    seconds_below_sla = float((below.astype(float) * dt_next).sum())
    percent_time_below_sla = float(100.0 * seconds_below_sla / total_duration_seconds) if total_duration_seconds > 0.0 else float('nan')

    return {
        'num_steps': int(df_exp.shape[0]),
        'overall_mean_latency': overall_mean_latency,
        'peak_latency': peak_latency,
        'num_times_crossed_above_sla': crossings,
        'total_seconds_above_sla_edge_cluster': total_seconds_above_sla_edge_cluster,
        'total_seconds_above_sla_cloud_cluster': total_seconds_above_sla_cloud_cluster,
        'number_migrations_edge_to_cloud': number_migrations_edge_to_cloud,
        'number_migrations_cloud_to_edge': number_migrations_cloud_to_edge,
        'total_seconds_in_cloud_cluster': total_seconds_in_cloud_cluster,
        'total_seconds_in_edge_cluster': seconds_in_edge_cluster,
        'percent_time_below_sla': percent_time_below_sla,
    }


def process_approach_per_experiment(
    approach_path: str,
    job: str,
    sla: float,
) -> pd.DataFrame:
    """Return a DataFrame with one row per experiment for this approach."""
    preprocessed_dir = os.path.join(approach_path, job, 'preprocessed')
    raw_dir = os.path.join(approach_path, job, 'raw')

    df_metrics = _load_preprocessed_metrics(preprocessed_dir)
    windows = _load_experiment_windows(raw_dir)

    rows: List[Dict[str, float]] = []
    for window in windows:
        df_exp = _slice_experiment(df_metrics, window)
        if df_exp.empty:
            continue

        summary = _summarize_single_experiment(df_exp, sla=sla)
        summary['experiment_id'] = window.source_file
        rows.append(summary)

    if not rows:
        raise RuntimeError(f"No experiments with data found for {approach_path}")

    df = pd.DataFrame(rows)
    return df[['experiment_id', 'num_steps', 'overall_mean_latency', 'peak_latency', 'num_times_crossed_above_sla', 'total_seconds_above_sla_edge_cluster', 'total_seconds_above_sla_cloud_cluster', 'number_migrations_edge_to_cloud', 'number_migrations_cloud_to_edge', 'total_seconds_in_cloud_cluster', 'total_seconds_in_edge_cluster', 'percent_time_below_sla']]


def main():
    parser = argparse.ArgumentParser(description='Per-experiment metrics then averaged per approach')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--job', default='job-0', help='Job subdirectory to process (default: job-0)')
    parser.add_argument('--approach', default=None, help='If provided, only process this approach (subfolder name)')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold for latency; default 0.2')
    parser.add_argument('--per_experiment_out', default=None, help='Optional CSV path for all per-experiment metrics')
    parser.add_argument('--approach_summary_out', default=None, help='Optional CSV path for averaged approach summary')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    approaches = [args.approach] if args.approach else _discover_approaches(root)

    # Collect per-experiment metrics across approaches
    per_exp_records: List[pd.DataFrame] = []
    approach_summaries: List[pd.DataFrame] = []

    for approach in sorted(approaches):
        approach_path = os.path.join(root, approach)
        job_path = os.path.join(approach_path, args.job)
        if not os.path.isdir(job_path):
            continue
        try:
            df_exp = process_approach_per_experiment(approach_path, args.job, sla=args.sla)
            df_exp.insert(0, 'approach', approach)
            per_exp_records.append(df_exp)

            # Averaged summary for this approach
            df_summary = pd.DataFrame({
                'approach': [approach],
                'num_experiments': [int(df_exp.shape[0])],
                'avg_num_steps': [float(df_exp['num_steps'].mean())],
                'avg_overall_mean_latency': [float(df_exp['overall_mean_latency'].mean())],
                'avg_peak_latency': [float(df_exp['peak_latency'].mean())],
                'avg_num_times_crossed_above_sla': [float(df_exp['num_times_crossed_above_sla'].mean())],
                'avg_total_seconds_above_sla_edge_cluster': [float(df_exp['total_seconds_above_sla_edge_cluster'].mean())],
                'avg_total_seconds_above_sla_cloud_cluster': [float(df_exp['total_seconds_above_sla_cloud_cluster'].mean())],
                'avg_number_migrations_edge_to_cloud': [float(df_exp['number_migrations_edge_to_cloud'].mean())],
                'avg_number_migrations_cloud_to_edge': [float(df_exp['number_migrations_cloud_to_edge'].mean())],
                'avg_total_seconds_in_cloud_cluster': [float(df_exp['total_seconds_in_cloud_cluster'].mean())],
                'avg_total_seconds_in_edge_cluster': [float(df_exp['total_seconds_in_edge_cluster'].mean())],
                'avg_percent_time_below_sla': [float(df_exp['percent_time_below_sla'].mean())],
            })
            approach_summaries.append(df_summary)
            print(f"Processed {approach}: {df_exp.shape[0]} experiments")
        except Exception as exc:
            print(f"Failed to process {approach}: {exc}")

    if not per_exp_records:
        raise RuntimeError('No approaches were processed. Check paths and inputs.')

    df_per_experiment = pd.concat(per_exp_records, ignore_index=True)
    df_approach_summary = pd.concat(approach_summaries, ignore_index=True)
    df_approach_summary = df_approach_summary.sort_values('avg_overall_mean_latency').reset_index(drop=True)

    # Write outputs if requested
    if args.per_experiment_out:
        os.makedirs(os.path.dirname(args.per_experiment_out), exist_ok=True)
        df_per_experiment.to_csv(args.per_experiment_out, index=False)
        print(f"Per-experiment metrics written to {args.per_experiment_out}")
    else:
        print(df_per_experiment.head())

    if args.approach_summary_out:
        os.makedirs(os.path.dirname(args.approach_summary_out), exist_ok=True)
        df_approach_summary.to_csv(args.approach_summary_out, index=False, sep=';')
        print(f"Approach summary written to {args.approach_summary_out}")
    else:
        print(df_approach_summary)


if __name__ == '__main__':
    main()


