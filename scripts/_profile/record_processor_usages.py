#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 16:45:31 (ywatanabe)"
# File: ./torchPAC/scripts/record_processor_usages.py

"""
Functionality:
    * Records system processes at specified intervals.
Input:
    * Interval in seconds and reset flag.
Output:
    * Process records (output format not specified in the given code).
Prerequisites:
    * mngs package
    * matplotlib
"""

"""Imports"""
import os
import sys

import matplotlib.pyplot as plt
import mngs

"""Functions & Classes"""
def main(interval_s, reset):
    """
    Records system processes at specified intervals.

    Parameters
    ----------
    interval_s : Union[int, float]
        Interval in seconds between recordings.
    reset : bool
        Flag to reset previous recordings.

    Returns
    -------
    None
    """
    mngs.resource.log_processor_usages(
        path=CONFIG.PATH.PROCESSOR_USAGE,
        limit_min=60 * 24,
        interval_s=interval_s,
        init=args.init,
        verbose=False,
        background=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Record system processes at specified intervals."
    )
    parser.add_argument(
        "--interval_s",
        type=float,
        default=0.5,
        help="Interval in seconds between recordings.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help="Reset flag to clear previous recordings.",
    )
    args = parser.parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main(args.interval_s, args.init)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
