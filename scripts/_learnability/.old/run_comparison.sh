#!/bin/bash
# Script to run PAC package comparison

# Set working directory to project root
cd "$(dirname "$0")/../.." || exit 1

# Create results directory if it doesn't exist
mkdir -p "scripts/learnability/results"

# Run the comparison
echo "Running PAC package comparison..."
python scripts/learnability/compare_packages.py "$@"

# Check if experiment ran successfully
if [ $? -eq 0 ]; then
    echo "Comparison completed successfully!"
    echo "Results are available in: scripts/learnability/results/"
    
    # List generated files
    echo -e "\nGenerated files:"
    ls -la scripts/learnability/results/
else
    echo "Comparison failed. Please check the error messages above."
    exit 1
fi