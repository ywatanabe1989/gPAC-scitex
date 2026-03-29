#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NOTE: This is a one-off utility script used during development.
# It was used to add n_trials parameter to compare_pac_values.py.
# Kept for historical reference only.
"""Update the compare_pac_values.py script to add n_trials parameter."""

import re

file_path = "./scripts/exp_02_tensorpac_comparison/compare_pac_values.py"

with open(file_path, 'r') as f:
    content = f.read()

# Add n_trials parameter to parse_args function
pattern = r'(parser\.add_argument\(\s*"--device",\s*type=str,\s*default="cpu",\s*choices=\["cpu", "cuda"\],\s*help="Computation device"\s*\)\s*)'
replacement = r'\1\n    # Trial parameters\n    parser.add_argument(\n        "--n_trials", type=int, default=3,\n        help="Number of trials for each test"\n    )\n'

updated_content = re.sub(pattern, replacement, content)

with open(file_path, 'w') as f:
    f.write(updated_content)

print(f"Updated {file_path} to add n_trials parameter")
