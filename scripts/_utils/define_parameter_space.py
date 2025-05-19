#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 23:53:56 (ywatanabe)"
# File: ./torchPAC/scripts/utils/define_parameter_space.py

"""
Functionality:
    * Generates parameter space based on the base space, with only one variable has varied values for experiments
Input:
    * parameter name to vary
Output:
    * List of parameter configurations
Prerequisites:
    * mngs package
    * matplotlib
"""

import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mngs
import pandas as pd

CONFIG = mngs.io.load_configs()


def define_parameter_space(param_name: str) -> List[Dict[str, Any]]:
    """Generates parameter configurations by varying one parameter while keeping others at baseline.

    Parameters
    ----------
    param_name : str
        Name of the parameter to vary

    Returns
    -------
    List[Dict[str, Any]]
        List of parameter configurations
    """

    PARAMS_BASELINE = CONFIG.PARAMS.BASELINE
    PARAMS_VARIATIONS = CONFIG.PARAMS.VARIATIONS

    if param_name not in PARAMS_BASELINE:
        raise ValueError(
            f"Parameter {param_name} not found in baseline configuration"
        )

    configs = []
    for value in PARAMS_VARIATIONS[param_name]:
        config = PARAMS_BASELINE.copy()
        config[param_name] = value
        configs.append(config)

    return configs

def main():
    PARAMS_BASELINE = CONFIG.PARAMS.BASELINE
    PARAMS_VARIATIONS = CONFIG.PARAMS.VARIATIONS
    PACKAGES = CONFIG.PACKAGES

    print("\nParameter Space Configurations:")
    print("-" * 80)

    rows = []
    for param in PARAMS_VARIATIONS.keys():
        configs = define_parameter_space(param)
        rows.append({
            'Parameter': param,
            'N Configs': len(configs),
            'Values': PARAMS_VARIATIONS[param],
            'Baseline': PARAMS_BASELINE[param],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Parameter')

    # Format Values column to be more readable
    df['Values'] = df['Values'].apply(lambda x: str(x).replace('[', '').replace(']', ''))

    print(df.to_string(index=False))
    print("-" * 80)

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )

    params = define_parameter_space("fs")
    main()

    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
