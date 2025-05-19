#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 12:30:54 (ywatanabe)"
# File: ./torchPAC/scripts/main.py

"""
Functionality:
    * Performs PAC (Phase-Amplitude Coupling) calculation using different packages.
Input:
    * Parameters for signal generation and PAC calculation.
Output:
    * Calculation statistics and parameters saved as CSV files.
Prerequisites:
    * mngs, numpy, pandas, torch, matplotlib packages
    * Custom Handlers (MNGSHandler, TensorpacHandler)
"""

import os
from typing import Any, Dict

import mngs
from scripts.utils.define_parameter_space import define_parameter_space
from scripts.utils.init_model import init_model
from scripts.utils.perform_pac_calculation import perform_pac_calculation
from scripts.utils.prepare_signal import prepare_signal
from scripts.utils.save_results import save_results

CONFIG = mngs.io.load_configs()


def run_condition(
    CONFIG, params: Dict[str, Any], condition_count: int
) -> None:
    """Executes PAC calculation for given parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing calculation parameters

    Returns
    -------
    None
    """

    # Time stamper
    params["ts"] = mngs.gen.TimeStamper()

    # Sequence length
    params["seq_len"] = int(params["t_sec"] * params["fs"])

    # Update SDIR
    CONFIG = CONFIG.copy()
    CONFIG.SDIR = os.path.join(
        CONFIG.SDIR, f"condition_{condition_count:04d}/"
    )

    # Model
    model = init_model(params)
    if model is None:
        return

    # Signal
    signal = prepare_signal(params)
    if signal is None:
        return

    # Main
    xpac = perform_pac_calculation(model, signal, params)

    # Saving
    save_results(model, xpac, params, CONFIG)

    # Cleanup
    del model


def main(CONFIG) -> None:
    """Main function to iterate through parameter spaces and run PAC calculations."""

    PARAM_NAMES = CONFIG.PARAMS.VARIATIONS.keys()
    condition_count = 0
    for param_name in PARAM_NAMES:
        params_list = define_parameter_space(param_name)
        for params in params_list:
            for package in CONFIG.PACKAGES:
                try:
                    params["package"] = package
                    run_condition(CONFIG, params, condition_count)
                    condition_count += 1
                except Exception as e:
                    print(e)

    mngs.sh(f"cp ./tmp/processor_usages.csv {CONFIG.SDIR}")
    return 0

# # Grids;
# def main(CONFIG) -> None:
#     """Main function to iterate through parameter spaces and run PAC calculations."""

#     condition_count = 0 # fake
#     PARAMS_SPACE = CONFIG.PARAMS.ALL
#     for params in mngs.utils.yield_grids(PARAMS_SPACE):
#         try:
#             run_condition(CONFIG, params, condition_count)
#             condition_count += 1
#         except Exception as e:
#             print(e)

#     mngs.sh(f"cp ./tmp/processor_usages.csv {COFIG.SDIR}")
#     return 0



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
    # -----------------------------------
    # Main
    # -----------------------------------

    exit_status = main(CONFIG)

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
