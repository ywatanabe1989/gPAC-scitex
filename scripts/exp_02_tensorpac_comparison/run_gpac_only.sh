#!/bin/bash
# Run gPAC analysis with surrogate distributions (skipping Tensorpac comparison)

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "========== Running gPAC Analysis with Surrogate Distributions =========="

# Generate output directory
mkdir -p "$SCRIPT_DIR/results/exp_02/pac_values"

# Run gPAC analysis with return_dist mode
echo "Computing PAC values with gPAC (with surrogate distributions)..."
cd "$ROOT_DIR"
python "$SCRIPT_DIR/compare_pac_values_gpac_only.py" --n_trials 3 --return_dist

echo "========== Analysis completed successfully =========="
echo "Results available in: $SCRIPT_DIR/results/exp_02/pac_values"

# Make the script executable
chmod +x "$0"