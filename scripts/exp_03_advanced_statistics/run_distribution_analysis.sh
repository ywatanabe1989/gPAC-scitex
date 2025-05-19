#!/bin/bash
# Run the PAC distribution analysis experiment

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default output directory
OUTPUT_DIR="$SCRIPT_DIR/results"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "========== Starting PAC Distribution Analysis =========="
echo "Output directory: $OUTPUT_DIR"

# Run the analysis
python "$SCRIPT_DIR/return_pac_distributions.py" --output_dir "$OUTPUT_DIR"

echo "========== Analysis completed successfully =========="
echo "Results available in: $OUTPUT_DIR"

# Make the script executable
chmod +x "$0"