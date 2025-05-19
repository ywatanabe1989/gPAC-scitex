#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 01:18:37 (ywatanabe)"
# File: ./torchPAC/scripts/utils/init_model.py

from typing import Union

from scripts.PackageHandlers import MNGSHandler, TensorpacHandler

def init_model(params: dict) -> Union[MNGSHandler, TensorpacHandler]:
    params_h = params.copy()
    params_h.update({
        "pha_min_hz": 2,
        "pha_max_hz": 20,
        "amp_min_hz": 80,
        "amp_max_hz": 160
    })
    package = params_h["package"]

    for key in ["batch_size", "n_chs", "n_segments", "t_sec", "package", "no_grad", "n_calc"]:
        params_h.pop(key, None)

    return {
        "mngs": MNGSHandler(**params_h),
        "tensorpac": TensorpacHandler(**params_h)
    }[package]


# EOF
