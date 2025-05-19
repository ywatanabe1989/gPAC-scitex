#!/bin/bash
# Timestamp: "2025-05-14 12:25:23 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/learnability_multitask_loss/run_experiment.sh

# Ensure we're in the project root directory
cd "$(dirname "$0")/../../.." || exit 1

# Check if data file exists
DATA_PATH="./scripts/learnability/data/synthetic_pac_signals.pt"
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️ Data file not found at $DATA_PATH"
    echo "Generating synthetic data first..."
    
    # Check if generation script exists
    if [ -f "./scripts/learnability/generate_binary_pac_data.py" ]; then
        python -m scripts.learnability.generate_binary_pac_data --n_samples 100
    else
        echo "❌ Error: Could not find data generation script."
        exit 1
    fi
fi

# Create output directory
mkdir -p "./scripts/learnability/learnability_multitask_loss/results"

# Run the multitask experiment
echo "🚀 Running MultiTaskLoss experiment..."
# Use CUDA for faster training on GPU
CUDA_VISIBLE_DEVICES=0 python -m scripts.learnability.learnability_multitask_loss.classify_binary_pac_multitask \
    --n_epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --pha_n_bands 10 \
    --amp_n_bands 10 \
    --test_ratio 0.2 \
    --track_interval 5

echo "✅ Experiment completed!"
echo "Results are available in: ./scripts/learnability/learnability_multitask_loss/results/"