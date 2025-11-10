#!/usr/bin/env python3
"""
Plot a pairwise comparison between one experiment from Approach 1 and one from
Approach 2, aligned by step index (not timestamps). Shade segments where
Approach 2 avoids SLA violations while Approach 1 is above SLA on Edge.

Input layout (same as other analysis scripts):
  <root>/<approach>/<job>/preprocessed/*.csv  (metrics with 'date', 'cluster')
  <root>/<approach>/<job>/raw/*.json          (each JSON has start_time/end_time)

Usage example:
  python f2f_meeting/plot_pairwise_approaches.py \
    --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
    --job job-0 \
    --approach_1 reactive \
    --approach_2 random_forest \
    --experiment_1 <approach_1_json> \
    --experiment_2 <approach_2_json> \
    --sla 0.2 \
    --out_dir f2f_meeting/result_analysis/pair_plots \
    --animate True \
    --animation_format mp4 \
    --plot_mlflow_1 /home/jolivera/Documents/CloudSkin/Time-Series-Library/f2f_meeting/comparison_of_approaches/reactive/job-0/raw/validation-reactive-updated-job-0_2025-09-14T160419_2025-09-15T000421_created_at_2025-09-15T000423_mlflow_results_cloudedge-migration-experiment-ci.csv \
    --plot_mlflow_2 f2f_meeting/comparison_of_approaches/random_forest/job-0/predictions/results/all_predictions.csv


Example prompt for f2f plot:

python f2f_meeting/plot_pairwise_approaches.py --root f2f_meeting/comparison_of_approaches/ --approach_1 reactive --approach_2 random_forest --job job-0 --experiment_1 validation-reactive-updated-job-0_2025-09-14T160419_2025-09-15T000421_created_at_2025-09-15T000423.json --experiment_2 validation-random-forest-job-0_2025-09-18T230054_2025-09-19T070056_created_at_2025-09-19T070058.json --sla 0.2 --out_dir f2f_meeting/result_analysis/pair_plots/ --animate True --animation_format mp4 --plot_mlflow_1 /home/jolivera/Documents/CloudSkin/Time-Series-Library/f2f_meeting/comparison_of_approaches/reactive/job-0/raw/validation-reactive-updated-job-0_2025-09-14T160419_2025-09-15T000421_created_at_2025-09-15T000423_mlflow_results_cloudedge-migration-experiment-ci.csv --plot_mlflow_2 f2f_meeting/comparison_of_approaches/random_forest/job-0/predictions/results/all_predictions.csv

"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter, FFMpegWriter
from datetime import datetime

TARGET_COLUMN = 'pipelines_status_realtime_pipeline_latency'
CLOUD_CLUSTER_ID = 'fd7816db-7948-4602-af7a-1d51900792a7'
EDGE_CLUSTER_ID = 'eb0e3eaa-b668-4ad6-bc10-2bb0eb7da259'


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
    for required in ('date', TARGET_COLUMN, 'cluster'):
        if required not in df_all.columns:
            raise ValueError(f"Expected '{required}' in preprocessed CSVs under {preprocessed_dir}")
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


def _find_window_by_name(windows: List[ExperimentWindow], name: str) -> ExperimentWindow:
    for w in windows:
        if w.source_file == name:
            return w
    raise FileNotFoundError(f"Experiment JSON '{name}' not found among: {[w.source_file for w in windows]}")


def _contiguous_runs(mask: pd.Series) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if mask.empty:
        return runs
    in_run = False
    start = 0
    for i, val in enumerate(mask.astype(bool).values.tolist()):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def main():
    parser = argparse.ArgumentParser(description='Pairwise plot by step index: shade indices where Approach 2 avoids SLA vs Approach 1')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--job', default='job-0', help='Job subdirectory to process (default: job-0)')
    parser.add_argument('--approach_1', default='reactive', help='First approach subfolder')
    parser.add_argument('--approach_2', default='random_forest', help='Second approach subfolder')
    parser.add_argument('--experiment_1', required=True, help='Exact JSON filename in approach_1 raw dir')
    parser.add_argument('--experiment_2', required=True, help='Exact JSON filename in approach_2 raw dir')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold for latency; default 0.2')
    parser.add_argument('--out_dir', default='result_analysis/pair_plots', help='Directory to write the plot image')
    parser.add_argument('--animate', type=bool, default=False, help='Create an animated plot')
    parser.add_argument('--animation_format', type=str, choices=['gif', 'mp4'], default='gif', help='Set animation format')
    parser.add_argument('--plot_mlflow_1', default=None, help='Path to MLflow predictions CSV for experiment 1 (or directory)')
    parser.add_argument('--plot_mlflow_2', default=None, help='Path to MLflow predictions CSV for experiment 2 (or directory)')
    args = parser.parse_args()

    a1_path = os.path.join(args.root, args.approach_1)
    a2_path = os.path.join(args.root, args.approach_2)

    # Load metrics and experiment windows
    a1_metrics = _load_preprocessed_metrics(os.path.join(a1_path, args.job, 'preprocessed'))
    a1_windows = _load_experiment_windows(os.path.join(a1_path, args.job, 'raw'))
    a2_metrics = _load_preprocessed_metrics(os.path.join(a2_path, args.job, 'preprocessed'))
    a2_windows = _load_experiment_windows(os.path.join(a2_path, args.job, 'raw'))

    w1 = _find_window_by_name(a1_windows, args.experiment_1)
    w2 = _find_window_by_name(a2_windows, args.experiment_2)

    df1 = _slice_experiment(a1_metrics, w1)
    df2 = _slice_experiment(a2_metrics, w2)
    if df1.empty or df2.empty:
        raise RuntimeError('Selected experiments have no data after slicing their windows')

    # Align by step index (for comparisons), but we'll plot against timestamps
    min_len = int(min(len(df1), len(df2)))
    df1 = df1.iloc[:min_len].reset_index(drop=True)
    df2 = df2.iloc[:min_len].reset_index(drop=True)

    # Build plot data (use timestamps on x-axis); align x to experiment_2 timestamps
    x1_time = pd.to_datetime(df1['date'])
    x2_time = pd.to_datetime(df2['date'])
    x_time = x2_time
    y1 = df1[TARGET_COLUMN].astype(float).values
    y2 = df2[TARGET_COLUMN].astype(float).values
    c1 = df1['cluster'].astype(str)
    c2 = df2['cluster'].astype(str)

    # Optional MLflow predictions overlays (CSV with 'qos_0') for exp1 and exp2
    y_pred1 = None
    y_pred2 = None
    x_pred1_time = None
    x_pred2_time = None
    if args.plot_mlflow_1:
        pred_path = args.plot_mlflow_1
        if os.path.isdir(pred_path):
            csv_files = [
                os.path.join(pred_path, f)
                for f in os.listdir(pred_path)
                if f.endswith('.csv')
            ]
            if csv_files:
                pred_path = sorted(csv_files)[0]
        if os.path.isfile(pred_path):
            try:
                df_pred = pd.read_csv(pred_path, sep=';')
            except Exception:
                df_pred = pd.read_csv(pred_path)
            if 'qos_0' in df_pred.columns:
                y_all = pd.to_numeric(df_pred['qos_0'], errors='coerce').values.astype(float)
                if len(y_all) > 0:
                    y_pred1 = y_all
                    if 'start_time' in df_pred.columns:
                        try:
                            x_pred1_time = pd.to_datetime(pd.to_numeric(df_pred['start_time'], errors='coerce'), unit='s')
                        except Exception:
                            x_pred1_time = None
    if args.plot_mlflow_2:
        pred_path = args.plot_mlflow_2
        if os.path.isdir(pred_path):
            csv_files = [
                os.path.join(pred_path, f)
                for f in os.listdir(pred_path)
                if f.endswith('.csv')
            ]
            if csv_files:
                pred_path = sorted(csv_files)[0]
        if os.path.isfile(pred_path):
            try:
                df_pred = pd.read_csv(pred_path, sep=';')
            except Exception:
                df_pred = pd.read_csv(pred_path)
            if 'qos_0' in df_pred.columns:
                y_all = pd.to_numeric(df_pred['qos_0'], errors='coerce').values.astype(float)
                if len(y_all) > 0:
                    y_pred2 = y_all
                    if 'start_time' in df_pred.columns:
                        try:
                            x_pred2_time = pd.to_datetime(pd.to_numeric(df_pred['start_time'], errors='coerce'), unit='s')
                        except Exception:
                            x_pred2_time = None

    # Align predictions to each experiment's timestamp range
    y_pred1_aligned = None
    y_pred2_aligned = None
    if y_pred1 is not None:
        if x_pred1_time is not None:
            try:
                s1 = pd.Series(y_pred1, index=x_pred1_time)
                s1_aligned = s1.reindex(x1_time, method='nearest')
                y_pred1_aligned = s1_aligned.values.astype(float)
            except Exception:
                y_pred1_aligned = y_pred1[:len(x1_time)]
        else:
            y_pred1_aligned = y_pred1[:len(x1_time)]
    if y_pred2 is not None:
        if x_pred2_time is not None:
            try:
                s2 = pd.Series(y_pred2, index=x_pred2_time)
                s2_aligned = s2.reindex(x_time, method='nearest')
                y_pred2_aligned = s2_aligned.values.astype(float)
            except Exception:
                y_pred2_aligned = y_pred2[:len(x_time)]
        else:
            y_pred2_aligned = y_pred2[:len(x_time)]

    # Avoided condition mask (Approach 2 avoids while Approach 1 violates on Edge)
    a1_above = y1 >= args.sla
    a1_edge = (c1 == EDGE_CLUSTER_ID).values
    a2_cloud = (c2 == CLOUD_CLUSTER_ID).values
    a2_below = y2 < args.sla
    avoided_mask = a1_above & a1_edge & a2_cloud & a2_below

    # Two subplots (shared x/y scales), shade avoided ranges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    def _plot_by_cluster(ax, x_arr, y_arr, clusters):
        cloud_color = 'C0'
        edge_color = 'C1'
        run_start = 0
        for i in range(1, len(x_arr) + 1):
            end_run = (i == len(x_arr)) or (clusters.iloc[i] != clusters.iloc[i - 1])
            if end_run:
                seg_cluster = clusters.iloc[run_start]
                color = cloud_color if seg_cluster == CLOUD_CLUSTER_ID else edge_color
                ax.plot(x_arr[run_start:i], y_arr[run_start:i], color=color, linestyle='-', linewidth=2)
                run_start = i

    # Calculate SLA violations
    sla_violations = _contiguous_runs(pd.Series(avoided_mask))

    # Approach 1 subplot
    _plot_by_cluster(ax1, x_time, y1, c1)
    ax1.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    for start, end in sla_violations:
        ax1.axvspan(x_time.iloc[start], x_time.iloc[end], color='gold', alpha=0.25)
    if y_pred1_aligned is not None:
        _n = int(min(len(y_pred1_aligned), len(x_time)))
        if _n > 0:
            ax1.plot(x_time[:_n], y_pred1_aligned[:_n], color='black', linestyle=':', linewidth=2)

    ax1.set_title(f"Reactive Approach")
    ax1.set_ylabel('Realtime latency')
    ax1.grid(True, alpha=0.3)
    legend_elements_ax1 = [
        Line2D([0], [0], color='C0', lw=2, label='Cloud'),
        Line2D([0], [0], color='C1', lw=2, label='Edge'),
        Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='SLA'),
        Line2D([0], [0], color='gold', lw=6, alpha=0.25, label='SLA breach'),
    ]
    if y_pred1_aligned is not None:
        legend_elements_ax1.append(Line2D([0], [0], color='black', lw=2, linestyle=':', label='Past 5-min QoS Mean'))
    ax1.legend(handles=legend_elements_ax1, loc='best')

    # Approach 2 subplot
    _plot_by_cluster(ax2, x_time, y2, c2)
    ax2.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    for start, end in sla_violations:
        ax2.axvspan(x_time.iloc[start], x_time.iloc[end], color='gold', alpha=0.25)
    if y_pred2_aligned is not None:
        _n = int(min(len(y_pred2_aligned), len(x_time)))
        if _n > 0:
            ax2.plot(x_time[:_n], y_pred2_aligned[:_n], color='black', linestyle=':', linewidth=2)

    ax2.set_title(f"Proactive Approach")
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Realtime latency')
    ax2.grid(True, alpha=0.3)
    legend_elements_ax2 = [
        Line2D([0], [0], color='C0', lw=2, label='Cloud'),
        Line2D([0], [0], color='C1', lw=2, label='Edge'),
        Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='SLA'),
        Line2D([0], [0], color='gold', lw=6, alpha=0.25, label='Avoided SLA breach'),
    ]
    if y_pred2_aligned is not None:
        legend_elements_ax2.append(Line2D([0], [0], color='black', lw=2, linestyle=':', label='Predicted QoS Proactive'))
    ax2.legend(handles=legend_elements_ax2, loc='best')

    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    safe_1 = os.path.splitext(w1.source_file)[0]
    safe_2 = os.path.splitext(w2.source_file)[0]
    out_path = os.path.join(args.out_dir, f"pairplot_two_panels__{safe_1}__VS__{safe_2}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

    # =============
    # Animated plot
    # =============
    if args.animate:

        def _reset_plots():
            """
            Reset plot canvas
            """
            # Clear data for both subplots
            ax1.cla()
            ax2.cla()

            # Update axes limits
            # ax1.set_xlim(
            #     df1["step_index"].min(),
            #     df1["step_index"].max()
            # )
            # ax2.set_xlim(
            #     df2["step_index"].min(),
            #     df2["step_index"].max()
            # )

            ax1.set_ylim(
                df1[TARGET_COLUMN].min(),
                df1[TARGET_COLUMN].max()
            )
            ax2.set_ylim(
                df2[TARGET_COLUMN].min(),
                df2[TARGET_COLUMN].max()
            )

            # Set up additional lines in top plot:
            # - Horizontal red line for SLA
            ax1.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            # Set static labels
            ax1.set_ylabel('Realtime latency')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Realtime latency')

            # Set grid
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # Set legend
            ax1.legend(handles=legend_elements_ax1, loc='best')
            ax2.legend(handles=legend_elements_ax2, loc='best')

            return [ax1, ax2]


        # Create an animation function to update the plot at each step
        def update(frame: int):
            # DEBUG: print current frame in terminal
            _start_time = datetime.now()
            # Reset plots
            ax1, ax2 = _reset_plots()

            # Update plot with latency values for the current step
            ax1.set_title(f"Reactive Approach")
            ax2.set_title(f"Proactive Approach")

            # Set data for each subplot using experiment_2 timestamps
            _plot_by_cluster(ax1, x_time[:frame + 1], y1[:frame + 1], c1)
            _plot_by_cluster(ax2, x_time[:frame + 1], y2[:frame + 1], c2)

            # Overlay predictions up to current frame (aligned to each experiment then mapped to shared x_time)
            if y_pred1_aligned is not None:
                end_idx = min(frame + 1, len(y_pred1_aligned), len(x_time))
                ax1.plot(x_time[:end_idx], y_pred1_aligned[:end_idx], color='black', linestyle=':', linewidth=2)
            if y_pred2_aligned is not None:
                end_idx = min(frame + 1, len(y_pred2_aligned), len(x_time))
                ax2.plot(x_time[:end_idx], y_pred2_aligned[:end_idx], color='black', linestyle=':', linewidth=2)

            # Plot Time windows for avoided SLA violations
            # - Disable it if this is already enabled in the init function
            for start, end in sla_violations:
                # Only draw segments prior or equal to the current frame
                if frame >= start:
                    # Render the full segment if frame > end; else, only up to the current frame
                    last_idx = min(frame, end)
                    ax1.axvspan(x_time.iloc[start], x_time.iloc[last_idx], color='gold', alpha=0.25)
                    ax2.axvspan(x_time.iloc[start], x_time.iloc[last_idx], color='gold', alpha=0.25)
                    
            _end_time = datetime.now()
            print(f'\rLast processed frame: {frame}. Process time: {_end_time - _start_time}. Total animation time: {_end_time - begin_anim_time}', end='', flush=True)

        # Create an animation object
        print(f"Frames length: {len(df1)}")
        print("Saving animated plot...")
        begin_anim_time = datetime.now()
        ani = FuncAnimation(
            fig, func=update, frames=len(df1), repeat=False,
            init_func=_reset_plots, blit=False, interval=200
            )
        print("")
        fig.tight_layout()
        # Save the animated plot (compatible with Python <3.10)
        animated_out_path = os.path.join(args.out_dir, f"pairplot_two_panels__{safe_1}__VS__{safe_2}_animated.{args.animation_format}")
        if args.animation_format == 'gif':
            ani.save(animated_out_path, writer=PillowWriter(fps=15))
        elif args.animation_format == 'mp4':
            ani.save(animated_out_path, writer=FFMpegWriter(fps=15))
        print(f"Saved animated plot to {animated_out_path}")


if __name__ == '__main__':
    main()


