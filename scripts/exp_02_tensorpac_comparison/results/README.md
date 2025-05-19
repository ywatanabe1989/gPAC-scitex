# Tensorpac Comparison Results

This directory contains the results from comparing gPAC with Tensorpac. The comparison includes calculation accuracy, performance benchmarks, and visualizations.

## Directory Structure

- `benchmark/`: Performance benchmark results
  - Contains CSV files with timing and resource usage measurements
  - Includes parameter sweep results for various configurations
  
- `pac_values/`: PAC calculation comparison results
  - Contains CSV files with numerical comparison metrics
  - Includes correlation coefficients, RMSE, and other accuracy metrics
  
- `figures/`: Generated figures for publication
  - `figure1_pac_workflow.{png,pdf,svg}`: Illustrates the PAC calculation workflow
  - `figure2_accuracy_comparison.{png,pdf,svg}`: Compares calculation accuracy
  - `figure3_performance_benchmark.{png,pdf,svg}`: Visualizes performance differences

## How to Regenerate Results

Run the complete experiment:
```bash
./run_comparison.sh
```

Or run individual components:
```bash
# Compare PAC values
python compare_pac_values.py --output_dir results/pac_values --n_trials 10

# Run benchmarks
python benchmark_vs_tensorpac.py --output_dir results/benchmark --param_grid param_grid.json

# Generate figures
python generate_comparison_figures.py \
  --pac_values_dir results/pac_values \
  --benchmark_dir results/benchmark \
  --output_dir results/figures
```