#!/bin/bash
# Script to run PAC learnability experiments

# Set working directory to project root
cd "$(dirname "$0")/../.." || exit 1

# Create results directory if it doesn't exist
mkdir -p "scripts/learnability/results"

# Set up Python environment (uncomment and modify if needed)
# source venv/bin/activate

# Run the experiment
echo "Running PAC learnability experiment..."
python scripts/learnability/demonstrate_learnability.py "$@"

# Check if experiment ran successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results are available in: scripts/learnability/results/"
    
    # List generated files
    echo -e "\nGenerated files:"
    ls -la scripts/learnability/results/
else
    echo "Experiment failed. Please check the error messages above."
    exit 1
fi