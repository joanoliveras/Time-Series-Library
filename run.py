from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch.backends
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_ets import Exp_ETS
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_linear_regression import Exp_Linear_Regression
from exp.exp_random_forest import Exp_Random_Forest
from utils.print_args import print_args

if __name__ == "__main__":
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, linear_regression, ets]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument('--data_iterate', type=bool, default=True, help='Whether to iterate through the root_path directory instead of using data_path directory. Used for inference with multiple files.')
    parser.add_argument(
        "--data_path", type=str, default="ETTh1.csv", help="data file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="OT",
        help="target feature in S or MS task",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument(
        "--seq_len", type=int, default=96, help="input sequence length"
    )
    parser.add_argument(
        "--label_len", type=int, default=48, help="start token length"
    )
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns",
        type=str,
        default="Monthly",
        help="subset for M4",
    )

    # inputation task
    parser.add_argument(
        "--mask_rate", type=float, default=0.25, help="mask ratio"
    )

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio",
        type=float,
        default=0.25,
        help="prior anomaly ratio (%)",
    )

    # model define
    parser.add_argument(
        "--expand", type=int, default=2, help="expansion factor for Mamba"
    )
    parser.add_argument(
        "--d_conv", type=int, default=4, help="conv kernel size for Mamba"
    )
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument(
        "--num_kernels", type=int, default=6, help="for Inception"
    )
    parser.add_argument(
        "--enc_in", type=int, default=7, help="encoder input size"
    )
    parser.add_argument(
        "--dec_in", type=int, default=7, help="decoder input size"
    )
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument(
        "--d_model", type=int, default=512, help="dimension of model"
    )
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument(
        "--e_layers", type=int, default=2, help="num of encoder layers"
    )
    parser.add_argument(
        "--d_layers", type=int, default=1, help="num of decoder layers"
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048, help="dimension of fcn"
    )
    parser.add_argument(
        "--moving_avg",
        type=int,
        default=25,
        help="window size of moving average",
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--activation", type=str, default="gelu", help="activation"
    )
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=0,
        help="num of down sampling layers",
    )
    parser.add_argument(
        "--down_sampling_window",
        type=int,
        default=1,
        help="down sampling window size",
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=48,
        help="the length of segmen-wise iteration of SegRNN",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument(
        "--train_epochs", type=int, default=10, help="train epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size of train input data",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="optimizer learning rate",
    )
    parser.add_argument(
        "--des", type=str, default="test", help="exp description"
    )
    parser.add_argument(
        "--loss", type=str, default="MSE", help="loss function"
    )
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=False, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--gpu_type", type=str, default="cuda", help="gpu type"
    )  # cuda or mps
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="use multiple gpus",
        default=False,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
        help="device ids of multile gpus",
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    # metrics (dtw)
    parser.add_argument(
        "--use_dtw",
        type=bool,
        default=False,
        help="the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)",
    )

    # Augmentation
    parser.add_argument(
        "--augmentation_ratio",
        type=int,
        default=0,
        help="How many times to augment",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Randomization seed"
    )
    parser.add_argument(
        "--jitter",
        default=False,
        action="store_true",
        help="Jitter preset augmentation",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
        help="Scaling preset augmentation",
    )
    parser.add_argument(
        "--permutation",
        default=False,
        action="store_true",
        help="Equal Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--randompermutation",
        default=False,
        action="store_true",
        help="Random Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--magwarp",
        default=False,
        action="store_true",
        help="Magnitude warp preset augmentation",
    )
    parser.add_argument(
        "--timewarp",
        default=False,
        action="store_true",
        help="Time warp preset augmentation",
    )
    parser.add_argument(
        "--windowslice",
        default=False,
        action="store_true",
        help="Window slice preset augmentation",
    )
    parser.add_argument(
        "--windowwarp",
        default=False,
        action="store_true",
        help="Window warp preset augmentation",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
        help="Rotation preset augmentation",
    )
    parser.add_argument(
        "--spawner",
        default=False,
        action="store_true",
        help="SPAWNER preset augmentation",
    )
    parser.add_argument(
        "--dtwwarp",
        default=False,
        action="store_true",
        help="DTW warp preset augmentation",
    )
    parser.add_argument(
        "--shapedtwwarp",
        default=False,
        action="store_true",
        help="Shape DTW warp preset augmentation",
    )
    parser.add_argument(
        "--wdba",
        default=False,
        action="store_true",
        help="Weighted DBA preset augmentation",
    )
    parser.add_argument(
        "--discdtw",
        default=False,
        action="store_true",
        help="Discrimitive DTW warp preset augmentation",
    )
    parser.add_argument(
        "--discsdtw",
        default=False,
        action="store_true",
        help="Discrimitive shapeDTW warp preset augmentation",
    )
    parser.add_argument(
        "--extra_tag", type=str, default="", help="Anything extra"
    )

    # TimeXer
    parser.add_argument(
        "--patch_len", type=int, default=16, help="patch length"
    )

    ## New arguments
    parser.add_argument(
        "--categorical_cols",
        type=str,
        default=None,
        help="Comma-separated list of categorical columns",
    )
    parser.add_argument(
        "--inverse", type=bool, help="inverse output data", default=False
    )
    parser.add_argument(
    "--output_path",
    type=str,
    default="./workflow/results/",
    help="path to save prediction outputs",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=None,
        help="If set (< pred_len), only keep the last output_len prediction steps",
    )

    args = parser.parse_args()
    if args.categorical_cols:
        args.categorical_cols = args.categorical_cols.split(",")
    print(torch.cuda.is_available())
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
        print("Using GPU")
    else:
        if hasattr(torch.backends, "mps"):
            args.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            args.device = torch.device("cpu")
        print("Using cpu or mps")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    elif args.task_name == "linear_regression":
        Exp = Exp_Linear_Regression
    elif args.task_name == "random_forest": 
        Exp = Exp_Random_Forest
    elif args.task_name == "ets":
        Exp = Exp_ETS
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(
                    setting
                )
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                    setting
                )
            )
            exp.test(setting)
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        all_predictions = {}

        if args.data_iterate:
            # Get list of all CSV files in root_path
            csv_files = [f for f in os.listdir(args.root_path) if f.endswith('.csv')]

            # Create output directory if it doesn't exist
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

            # Directory to store merged inputs
            merged_inputs_dir = os.path.join(args.output_path, 'merged_inputs')
            if not os.path.exists(merged_inputs_dir):
                os.makedirs(merged_inputs_dir)

            # Read and concatenate all files, then sort and drop duplicate timestamps
            import pandas as pd
            frames = []
            for file in csv_files:
                input_file_path = os.path.join(args.root_path, file)
                try:
                    df = pd.read_csv(input_file_path)
                except Exception as e:
                    print(f"Skipping file due to read error {file}: {e}")
                    continue
                if 'date' not in df.columns:
                    print(f"Skipping file missing required column 'date': {file}")
                    continue
                frames.append(df)

            if not frames:
                print("No valid CSV files found to merge for inference.")
            else:
                merged_df = pd.concat(frames, ignore_index=True)
                merged_df['date'] = pd.to_datetime(merged_df['date'])
                merged_df = merged_df.sort_values('date')
                merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')

                # Write merged file
                merged_filename = "merged_all.csv"
                merged_path = os.path.join(merged_inputs_dir, merged_filename)
                merged_df.to_csv(merged_path, index=False)

                # Predict using the merged file (Dataset will take last seq_len rows)
                saved_root = args.root_path
                args.root_path = merged_inputs_dir
                args.data_path = merged_filename
                exp = Exp(args)  # set experiments
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                preds = exp.predict(setting)
                # If output_len is set, keep only the last output_len steps (time dimension)
                output_len = getattr(args, 'output_len', None)
                try:
                    if (
                        output_len is not None
                        and isinstance(output_len, int)
                        and hasattr(preds, 'shape')
                        and preds.ndim == 3
                        and 0 < output_len < preds.shape[1]
                    ):
                        preds = preds[:, -output_len:, :]
                except Exception as e:
                    print(f"Warning: could not apply output_len slicing: {e}")

                # Future timestamps based on last time in merged data (30-second steps)
                last_timestamp = merged_df['date'].iloc[-1]
                # Number of steps equals the predictions returned (after any slicing)
                n_steps = int(preds.shape[1]) if hasattr(preds, 'shape') else int(args.pred_len)
                future_timestamps = pd.date_range(
                    start=last_timestamp + pd.Timedelta(seconds=30),
                    periods=n_steps,
                    freq='30S'
                )

                # Prepare predictions for CSV format
                print(preds.shape)
                print(preds)
                predictions_flat = preds.reshape(-1)
                print(predictions_flat.shape)

                # Use the last row's cluster for output tagging if present
                cluster_name = str(merged_df['cluster'].iloc[-1]) if 'cluster' in merged_df.columns else ''

                all_predictions_data = []
                for i, pred_value in enumerate(predictions_flat):
                    row = {
                        'date': future_timestamps[i].strftime('%Y-%m-%dT%H:%M:%S'),
                        args.target: float(pred_value)
                    }
                    if cluster_name:
                        row['cluster'] = cluster_name
                    all_predictions_data.append(row)

                # Store predictions with key 'merged_all' (for console output)
                all_predictions['merged_all'] = preds
                torch.cuda.empty_cache()
                # Restore original root_path
                args.root_path = saved_root

                # Save all predictions to a single CSV file
                predictions_df = pd.DataFrame(all_predictions_data)
                output_filename = "all_predictions.csv"
                output_path = os.path.join(args.output_path, output_filename)
                predictions_df.to_csv(output_path, index=False)
                print(f"\nSaved all predictions to {output_path}")
                print(f"Total predictions: {len(predictions_df)} rows")