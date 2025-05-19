#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 02:25:04 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/run_binary_pac_analysis.sh
# ----------------------------------------

# Run the complete binary PAC analysis pipeline
# This script:
# 1. Generates synthetic binary PAC signals
# 2. Extracts and visualizes PAC features
# 3. Creates detailed visualizations of discriminative PAC patterns
# 4. Visualizes the "learnability" aspect - how frequency bands evolve during training

# Set the Python interpreter path
PYTHON_CMD="python3"

# Set working directory to project root
cd "$(dirname "$0")/../.." || exit 1
echo "Working directory: $(pwd)"

# First create a backup of previous results
echo "Backing up previous results..."
mkdir -p scripts/learnability/.old
mv -f scripts/learnability/results scripts/learnability/.old/ 2>/dev/null || true

# Create necessary directories
mkdir -p scripts/learnability/data
mkdir -p scripts/learnability/results/visualizations
mkdir -p scripts/learnability/results/learnability

# Step 1: Generate binary PAC signals
echo "==============================================="
echo "Generating synthetic binary PAC signals..."
echo "==============================================="
$PYTHON_CMD -m scripts.learnability.generate_binary_pac_data --n_samples 100

# Check if data generation was successful
if [ ! -f "scripts/learnability/data/binary_pac_signals.pt" ]; then
    echo "Error: Failed to generate binary PAC signals."
    exit 1
fi

echo "Data generation completed successfully."
echo

# Step 2: Run the PAC feature visualization
echo "==============================================="
echo "Visualizing PAC features and discriminative patterns..."
echo "==============================================="
$PYTHON_CMD -m scripts.learnability.visualize_pac_features \
    --data_path "./scripts/learnability/data/binary_pac_signals.pt" \
    --results_dir "./scripts/learnability/results/visualizations" \
    --pha_n_bands 10 \
    --amp_n_bands 10 \
    --save_figs

# Check if visualization was successful
if [ ! -f "scripts/learnability/results/visualizations/discriminative_features.png" ]; then
    echo "Warning: Some visualizations may not have been created."
else
    echo "Visualization completed successfully."
fi

# Step 3: Run the PAC learnability visualization
echo "==============================================="
echo "Visualizing PAC learnability - band evolution during training..."
echo "==============================================="
$PYTHON_CMD -m scripts.learnability.visualize_pac_learnability \
    --data_path "./scripts/learnability/data/binary_pac_signals.pt" \
    --results_dir "./scripts/learnability/results/learnability" \
    --pha_n_bands 5 \
    --amp_n_bands 5 \
    --n_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.01 \
    --track_interval 5 \
    --save_figs

# Check if learnability visualization was successful
if [ ! -f "scripts/learnability/results/learnability/band_evolution.png" ]; then
    echo "Warning: Some learnability visualizations may not have been created."
else
    echo "Learnability visualization completed successfully."
fi

echo
echo "==============================================="
echo "Binary PAC analysis completed!"
echo "Results can be found in: scripts/learnability/results/"
echo "==============================================="

# Provide a list of generated figures
echo "Feature visualizations:"
ls -la scripts/learnability/results/visualizations/

echo -e "\nLearnability visualizations:"
ls -la scripts/learnability/results/learnability/

exit 0