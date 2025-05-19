#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:13:20 (ywatanabe)"
# File: ./torchPAC/scripts/utils/resource_info.py

"""
Functionality:
    * Records and saves the current machine's resource information.
Input:
    * None (gathers information automatically)
Output:
    * YAML file containing resource information
Prerequisites:
    * mngs package
    * matplotlib
"""

"""Imports"""
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import mngs

"""Config"""
CONFIG = mngs.io.load_configs()

"""Functions & Classes"""
def main() -> None:
    """
    Gathers and saves the current machine's resource information.

    Returns
    -------
    None
    """
    resource_info: Dict[str, Any] = mngs.resource.gather_info()
    mngs.io.save(resource_info, "resource_info.yaml")

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
    main()
    mngs.gen.close(CONFIG)

# EOF
