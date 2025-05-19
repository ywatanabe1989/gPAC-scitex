#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 10:48:40 (ywatanabe)"
# File: ./torchPAC/scripts/utils/save_results.py

from typing import Any, Dict

import mngs
import pandas as pd
import torch


def dict_to_df(params: Dict[str, Any]) -> pd.DataFrame:
    """Convert parameters dictionary to DataFrame with proper orientation."""
    return pd.DataFrame.from_dict(params, orient='index', columns=['value'])

def dict_to_series(params: Dict[str, Any]) -> pd.Series:
    """Convert parameters dictionary to Series with proper values."""
    return pd.Series(params.values(), index=params.keys())

def save_results(model, xpac, params, CONFIG):
    mngs.str.printc(
        f"{params['package']}\n"
        f"{model.stats['calc_time_mean_sec'].iloc[0]} +/- {model.stats['calc_time_std_sec'].iloc[0]} sec "
        f"(n = {model.stats['calc_time_nn'].iloc[0]:,})",
        c="green" if params["package"] == "mngs" else "magenta",
    )

    # Convert CUDA tensor to numpy
    if torch.is_tensor(xpac):
        if xpac.is_cuda:
            xpac = xpac.detach().cpu()
        xpac = xpac.numpy()
    mngs.io.save(xpac, CONFIG["SDIR"] + "xpac.npy")

    # Time
    df_time = model.stats
    df_time = df_time.reset_index()
    df_time = df_time.rename(columns={"index": "package"})
    # mngs.io.save(df_time, CONFIG["SDIR"] + "stats.csv")

    # Parameters
    params_copy = params.copy()
    del params_copy["ts"]
    df_params = pd.DataFrame.from_dict(params_copy, orient='index').T
    # mngs.io.save(df_params, CONFIG["SDIR"] + "params.csv")

    # Time + Parameters
    df_combined = pd.concat([df_time, df_params], axis=1)
    df_combined = df_combined.set_index("time")

    # Saving
    mngs.io.save(df_combined, CONFIG["SDIR"] + "stats.csv")

# EOF
