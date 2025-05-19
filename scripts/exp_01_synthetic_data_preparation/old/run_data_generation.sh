#!/bin/bash
# Script to run the synthetic PAC data generation and tests

# Navigate to project root
cd "$(dirname "$0")/../.." || exit 1

# Check if CUDA is available and set device accordingly
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, will use GPU if available in PyTorch"
    DEVICE="cuda"
else
    echo "CUDA not detected, using CPU"
    DEVICE="cpu"
fi

# Parse command line arguments
SAMPLES=${1:-200}  # Default to 200 samples
CHANNELS=${2:-4}   # Default to 4 channels
DURATION=${3:-2.0} # Default to 2.0 seconds

echo "=== Starting Synthetic PAC Data Generation ==="
echo "Samples per class: $SAMPLES"
echo "Channels: $CHANNELS"
echo "Duration: $DURATION seconds"

# Run data generation script
python ./scripts/exp_01_synthetic_data_preparation/generate_synthetic_data.py \
    --n_samples "$SAMPLES" \
    --n_channels "$CHANNELS" \
    --duration "$DURATION"

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed"
    exit 1
fi

echo -e "\n=== Running tests to verify data generation ==="
# Run test script
python ./scripts/exp_01_synthetic_data_preparation/test_data_generation.py

# Check if tests were successful
if [ $? -ne 0 ]; then
    echo "ERROR: Tests failed"
    exit 1
fi

echo -e "\n=== All tasks completed successfully ==="
echo "Generated data is available in ./scripts/exp_01_synthetic_data_preparation/data/"
echo "You can use this data for subsequent experiments"