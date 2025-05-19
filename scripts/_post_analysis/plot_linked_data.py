#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 20:08:25 (ywatanabe)"
# File: ./torchPAC/scripts/link_data.py

"""
1. Functionality:
   - Links performance measurements with statistical data and visualizes relationships
2. Input:
   - Statistical data from multiple files
   - Processor usage data (CPU, RAM, GPU, VRAM)
3. Output:
   - Barplots showing relationships between resource usage and calculation time
4. Prerequisites:
   - mngs package
   - pandas, matplotlib, seaborn
"""

"""Imports"""
import sys

import mngs
import pandas as pd

CONFIG = mngs.io.load_configs()

"""Functions & Classes"""
def load_stats() -> pd.DataFrame:
    """Loads and preprocesses statistical data from multiple files."""
    PATHS_STATS = mngs.io.glob(CONFIG.PATH.RESULTS.STATS)
    df = pd.concat([mngs.io.load(path) for path in PATHS_STATS])
    df = df.drop(columns=["package"])
    df = df.rename(columns={"package.1": "package"})
    df = df.set_index("time")
    return df

def load_processor_usages() -> pd.DataFrame:
    """Loads processor usage data from CSV file."""
    return mngs.io.load(CONFIG.PATH.RESULTS.PROCESSOR_USAGE)

def match_nearest_timestamps(df_p: pd.DataFrame, df_s: pd.DataFrame) -> pd.DataFrame:
    """Matches performance data with nearest statistical timestamps.

    Parameters
    ----------
    df_p : pd.DataFrame
        Performance data with Timestamp column
    df_s : pd.DataFrame
        Statistical data with datetime index

    Returns
    -------
    pd.DataFrame
        Performance data resampled to statistical timestamps
    """
    df_p_times = pd.to_datetime(df_p["Timestamp"])
    df_s_times = pd.to_datetime(df_s.index)
    indices = [abs(df_p_times - t).argmin() for t in df_s_times]
    return df_p.iloc[indices].set_index(df_s.index)


def link(df_p: pd.DataFrame, df_s: pd.DataFrame) -> pd.DataFrame:
    """Links performance data with statistical data.

    Parameters
    ----------
    df_p : pd.DataFrame
        Performance data with Timestamp column
    df_s : pd.DataFrame
        Statistical data with datetime index

    Returns
    -------
    pd.DataFrame
        Combined dataframe with matched timestamps
    """
    df = match_nearest_timestamps(df_p, df_s)
    df = pd.concat([df, df_s], axis=1)
    return df

# def visualize_performance_metrics(df: pd.DataFrame, save_dir: str = "./jpg") -> None:
#     """Creates performance comparison plots between packages."""
#     param_groups = {
#         "Signal Size": {
#             "batch_size": [2, 4, 8, 16, 32, 64],
#             "n_chs": [2, 4, 8, 16, 32, 64],
#             "n_segments": [2, 4, 8, 16, 32],
#             "t_sec": [2, 4, 8],
#             "fs": [512, 1024],
#         },
#         "PAC Resolution": {
#             "pha_n_bands": [10, 30, 50, 70, 100],
#             "amp_n_bands": [10, 30, 50, 70, 100],
#         },
#         "Computation": {
#             "chunk_size": [2, 4, 8],
#             "n_perm": [None, 1, 2, 4, 8],
#             "fp16": [False, True],
#         },
#         "Package Specific": {
#             "no_grad": [False, True],
#             "in_place": [False, True],
#             "trainable": [False, True],
#             "device": ["cpu", "cuda"],
#             "use_threads": [False, True],
#         }
#     }


#     for group_name, params in param_groups.items():
#         n_cols = min(3, len(params))
#         n_rows = (len(params) + n_cols - 1) // n_cols

#         fig, axes = mngs.plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows))
#         axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

#         for idx, ((param, values), ax) in enumerate(zip(params.items(), axes)):
#             sns_plot = ax.sns_barplot(
#                 data=df,
#                 x=param,
#                 y="calc_time_mean_sec",
#                 hue="package",
#                 hue_order=["tensorpac", "mngs"],
#                 hue_colors={"mngs": CC["blue"], "tensorpac": CC["red"]},
#             )
#             ax.set_title(f"{param} ({len(values)} values)")

#             if len(values) > 10:
#                 ax.tick_params(axis='x', rotation=45)

#             if idx % n_cols != 0:
#                 ax.set_ylabel("")

#         # Hide empty subplots
#         for ax in axes[len(params):]:
#             ax.set_visible(False)

#         fig.suptitle(f"{group_name} Parameters vs Calculation Time")
#         fig.tight_layout()
#         mngs.io.save(fig, f"{save_dir}/{group_name.lower().replace(' ', '_')}.jpg")
#         plt.close()

def process_var(df, col_var):
    # Other columns except for the target variable at the hand
    base_params_left = CONFIG.PARAMS.BASELINE.copy()
    base_params_left.pop(col_var)

    _df = mngs.pd.slice(df, base_params_left)

    y_vars = ['CPU [%]', 'RAM [GiB]', 'GPU [%]', 'VRAM [GiB]', 'init_time_mean_sec', 'calc_time_mean_sec']
    for y_var in y_vars:
        fig, ax = mngs.plt.subplots()
        ax.sns_barplot(
            data=_df,
            x=col_var,
            y=y_var,
            hue="package",
            hue_order=["tensorpac", "mngs"],
            hue_colors={"mngs": CC["blue"], "tensorpac": CC["red"]},
        )
        mngs.io.save(fig, f"./jpg/{y_var}/{col_var}.jpg")

def main():

    # Loading
    df_s = load_stats()
    df_p = load_processor_usages()

    # Linking calculation time and resource usage
    df = link(df_p, df_s)

    # Plotting
    params_baseline = CONFIG.PARAMS.BASELINE
    params_variations = CONFIG.PARAMS.VARIATIONS
    for col_var in CONFIG.PARAMS.VARIATIONS.keys():
        process_var(df, col_var)





if __name__ == "__main__":
    # -----------------------------------
    # Initiatialization of mngs format
    # -----------------------------------
    import sys

    import matplotlib.pyplot as plt

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        # sdir_suffix="",
    )

    # # Argument parser
    # script_mode = mngs.gen.is_script()
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, choices=None, default=1, help='(default: %%(default)s)')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='(default: %%(default)s)')
    # args = parser.parse_args()
    # mngs.gen.print_block(args, c='yellow')

    # -----------------------------------
    # Main
    # -----------------------------------
    exit_status = main()

    # -----------------------------------
    # Cleanup mngs format
    # -----------------------------------
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

# EOF

# # df.columns
# columns = [
#     "package",
#     "init_time_mean_sec",
#     "init_time_std_sec",
#     "init_time_nn",
#     "calc_time_mean_sec",
#     "calc_time_std_sec",
#     "calc_time_nn",
#     "batch_size",
#     "n_chs",
#     "n_segments",
#     "t_sec",
#     "fs",
#     "pha_n_bands",
#     "amp_n_bands",
#     "chunk_size",
#     "n_perm",
#     "fp16",
#     "n_calc",
#     "no_grad",
#     "in_place",
#     "trainable",
#     "device",
#     "use_threads",
#     "package",
#     "seq_len",
# ]

# EOF
