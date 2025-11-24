from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class Exp_ETS:
    """
    ETS inference that:
    - Determines cluster from latest production row
    - Picks matching training dataset (edge/cloud)
    - Shifts the entire training timeline so its last calendar day equals the
      day before production's last calendar day (preserving time-of-day)
    - Concatenates training + production, fills 30s gaps with zeros
    - Fits ExponentialSmoothing and forecasts 20 steps, returning last 10
    """

    TARGET_COL = "pipelines_status_realtime_pipeline_latency"
    DATE_COL = "date"
    CLUSTER_COL = "cluster"
    FREQ = "30S"
    FORECAST_STEPS = 20
    OUTPUT_LAST_STEPS = 10  # return steps 10..19

    TRAIN_EDGE_PATH = "./TSLib/dataset/training_dataset/new_queues_concurrency_4_after_migration/preprocessed_data_ets_edge.csv"
    TRAIN_CLOUD_PATH = "./TSLib/dataset/training_dataset/new_queues_concurrency_4_after_migration/preprocessed_data_ets_cloud.csv"

    def __init__(self, args):
        self.args = args

    def _read_production(self) -> pd.DataFrame:
        prod_path = os.path.join(self.args.root_path, self.args.data_path)
        if not os.path.exists(prod_path):
            raise FileNotFoundError(f"Production dataset not found at: {prod_path}")
        df = pd.read_csv(prod_path)
        if self.DATE_COL not in df.columns:
            raise ValueError(f"Production dataset must contain '{self.DATE_COL}' column")
        # Keep only required columns (date, target, cluster if exists)
        cols = [self.DATE_COL]
        if self.TARGET_COL in df.columns:
            cols.append(self.TARGET_COL)
        else:
            raise ValueError(
                f"Production dataset must contain '{self.TARGET_COL}' column"
            )
        if self.CLUSTER_COL in df.columns:
            cols.append(self.CLUSTER_COL)
        df = df[cols].copy()
        df[self.DATE_COL] = pd.to_datetime(df[self.DATE_COL])
        df.sort_values(self.DATE_COL, inplace=True)
        df.drop_duplicates(subset=[self.DATE_COL], keep="last", inplace=True)
        # Coerce target to numeric
        df[self.TARGET_COL] = pd.to_numeric(df[self.TARGET_COL], errors="coerce")
        return df

    def _select_training_path_from_cluster(self, cluster_value: str) -> str:
        cluster_str = str(cluster_value).strip().lower()
        # default to cloud unless 'edge' is clearly present
        if "eb0e3eaa-b668-4ad6-bc10-2bb0eb7da259" in cluster_str:
            return self.TRAIN_EDGE_PATH
        if "fd7816db-7948-4602-af7a-1d51900792a7" in cluster_str:
            return self.TRAIN_CLOUD_PATH
        else:
            raise ValueError(f"Invalid cluster: {cluster_str}") 

    def _read_training(self, training_path: str) -> pd.DataFrame:
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Training dataset not found at: {training_path}")
        df = pd.read_csv(training_path)
        if self.DATE_COL not in df.columns or self.TARGET_COL not in df.columns:
            raise ValueError(
                f"Training dataset must contain '{self.DATE_COL}' and '{self.TARGET_COL}'"
            )
        df = df[[self.DATE_COL, self.TARGET_COL]].copy()
        df[self.DATE_COL] = pd.to_datetime(df[self.DATE_COL])
        df.sort_values(self.DATE_COL, inplace=True)
        df.drop_duplicates(subset=[self.DATE_COL], keep="last", inplace=True)
        # Coerce target to numeric
        df[self.TARGET_COL] = pd.to_numeric(df[self.TARGET_COL], errors="coerce")
        return df

    def _shift_training_to_day_before(self, df_train: pd.DataFrame, last_prod_ts: pd.Timestamp) -> pd.DataFrame:
        """
        Shift entire training dates so that its last calendar day equals
        (prod_last_day - 1 day), preserving time-of-day patterns.
        """
        last_train_ts = df_train[self.DATE_COL].max()
        # Align midnights difference to preserve time-of-day
        new_last_day_midnight = last_prod_ts.normalize() - pd.Timedelta(days=1)
        delta = new_last_day_midnight - last_train_ts.normalize()
        df_train_shifted = df_train.copy()
        df_train_shifted[self.DATE_COL] = df_train_shifted[self.DATE_COL] + delta
        return df_train_shifted

    def _merge_fill_30s(self, df_train: pd.DataFrame, df_prod: pd.DataFrame) -> pd.Series:
        df = pd.concat(
            [
                df_train[[self.DATE_COL, self.TARGET_COL]],
                df_prod[[self.DATE_COL, self.TARGET_COL]],
            ],
            ignore_index=True,
        )
        df[self.DATE_COL] = pd.to_datetime(df[self.DATE_COL])
        df.sort_values(self.DATE_COL, inplace=True)
        full_range = pd.date_range(
            start=df[self.DATE_COL].min(), end=df[self.DATE_COL].max(), freq=self.FREQ
        )
        df_full = (
            df.set_index([self.DATE_COL])
            .reindex(full_range)
            .rename_axis(self.DATE_COL)
            .reset_index()
        )
        df_full[self.TARGET_COL] = df_full[self.TARGET_COL].fillna(0.0)
        # Return a Series indexed by date for ETS fitting
        series = df_full.set_index(self.DATE_COL)[self.TARGET_COL]
        return series

    def _fit_ets(self, series: pd.Series):
        model = ExponentialSmoothing(
            series,
            seasonal_periods=2880,  # 1 day at 30-second frequency
            trend="add",
            seasonal="add",
            damped_trend=False,
            initialization_method="heuristic",
        )
        fitted = model.fit()
        return fitted

    def predict(self, setting: str, load: bool = False):
        # Load production and decide cluster
        prod_df_all = self._read_production()
        if self.CLUSTER_COL not in prod_df_all.columns or prod_df_all[self.CLUSTER_COL].dropna().empty:
            raise ValueError(
                f"Production dataset must contain a non-empty '{self.CLUSTER_COL}' column to select training dataset"
            )
        # Consider only latest production day; drop previous-day rows if present
        last_prod_ts = prod_df_all[self.DATE_COL].max()
        day_start = last_prod_ts.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        prod_df = prod_df_all[
            (prod_df_all[self.DATE_COL] >= day_start) & (prod_df_all[self.DATE_COL] < day_end)
        ].copy()
        if prod_df.empty:
            raise ValueError("Filtered production dataset for latest day is empty")
        latest_cluster = prod_df[self.CLUSTER_COL].dropna().iloc[-1]
        training_path = self._select_training_path_from_cluster(latest_cluster)

        # Load and shift training, then merge and fill 30s gaps
        train_df = self._read_training(training_path)
        train_shifted = self._shift_training_to_day_before(train_df, last_prod_ts)

        series = self._merge_fill_30s(train_shifted, prod_df)

        # Fit ETS and forecast
        fitted = self._fit_ets(series)
        fcst = fitted.forecast(self.FORECAST_STEPS)
        # Keep the last 10 steps (indices 10..19)
        last_10 = np.asarray(fcst)[-self.OUTPUT_LAST_STEPS:].astype(float)
        # Return as (1, steps, 1)
        pred = last_10.reshape(1, self.OUTPUT_LAST_STEPS, 1)
        return pred


