#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:07:30 (ywatanabe)"
# File: ./torchPAC/scripts/utils/perform_pac_calculation.py

import torch


def perform_pac_calculation(model, signal, params):
    try:
        for _ in range(params["n_calc"]):
            model.ts(model.calc_start_str)
            if params["no_grad"]:
                with torch.no_grad():
                    xpac = model.calc_pac(signal)
            else:
                xpac = model.calc_pac(signal)
            model.ts(model.calc_end_str)
        return xpac
    except Exception as exception:
        print(f"Error in PAC calculation: {exception}")


# EOF
