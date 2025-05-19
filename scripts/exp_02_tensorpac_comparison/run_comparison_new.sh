#!/bin/bash
# Run the complete Tensorpac comparison experiment

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "========== Starting gPAC vs Tensorpac Comparison =========="

# Step 1: Compare PAC values between gPAC and Tensorpac
echo "[1/3] Comparing PAC values between gPAC and Tensorpac..."
python "$SCRIPT_DIR/compare_pac_values.py" --n_trials 2

# Step 2: Run performance benchmarks
echo "[2/3] Running performance benchmarks..."
python "$SCRIPT_DIR/benchmark_vs_tensorpac.py" --param_set BASELINE --n_trials 2

# Step 3: Generate comparison figures
echo "[3/3] Generating comparison figures..."
python "$SCRIPT_DIR/generate_comparison_figures.py"

echo "========== Experiment completed successfully =========="
echo "Results available in: $SCRIPT_DIR/results"