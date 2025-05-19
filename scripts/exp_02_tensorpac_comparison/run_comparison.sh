#!/bin/bash
# Run the complete Tensorpac comparison experiment

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "========== Starting gPAC vs Tensorpac Comparison =========="

# Step 1: Compare PAC values between gPAC and Tensorpac (standard mode)
echo "[1/4] Comparing PAC values between gPAC and Tensorpac (standard mode)..."
python "$SCRIPT_DIR/compare_pac_values.py" --n_trials 5

# Step 1b: Compare PAC values between gPAC and Tensorpac (with surrogate distributions)
echo "[2/4] Comparing PAC values between gPAC and Tensorpac (with surrogate distributions)..."
python "$SCRIPT_DIR/compare_pac_values.py" --n_trials 5 --return_dist

# Step 2: Run performance benchmarks
echo "[3/4] Running performance benchmarks..."
python "$SCRIPT_DIR/benchmark_vs_tensorpac.py" --param_set BASELINE --n_trials 3

# Step 3: Generate comparison figures
echo "[4/4] Generating comparison figures..."
python "$SCRIPT_DIR/generate_comparison_figures.py"

echo "========== Experiment completed successfully =========="
echo "Results available in: $SCRIPT_DIR/results"

# Make the script executable
chmod +x "$0"