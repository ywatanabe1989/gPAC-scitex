<!-- ---
!-- Timestamp: 2025-05-14 11:34:52
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/exp_02_tensorpac_comparison/README.md
!-- --- -->

# Experiment 02: Comparison with Tensorpac

This directory contains code for comparing the gPAC implementation with Tensorpac, a widely-used Python package for Phase-Amplitude Coupling analysis.

## Overview

The comparison focuses on three key aspects:

1. **Calculation Accuracy**: Verification that gPAC produces results consistent with Tensorpac
2. **Performance Benchmarking**: Speed comparison across various parameter configurations  
3. **Scalability Analysis**: Evaluation of how both packages scale with increasing data complexity

## Scripts

- `benchmark_vs_tensorpac.py`: Main benchmarking script comparing performance
- `compare_pac_values.py`: Script to verify calculation accuracy between packages
- `generate_comparison_figures.py`: Creates publication-ready figures from benchmarking results
- `run_comparison.sh`: Shell script to run the complete comparison pipeline
- `param_grid.json`: Configuration file defining parameter combinations to test

## Figures Generated

The experiment generates several figures for the manuscript:

1. **Figure 1: PAC Calculation Workflow**
   - Visualization of how parallel execution in gPAC accelerates computation

2. **Figure 2: Calculation Accuracy**
   - Comparison of PAC values between gPAC and Tensorpac
   - Correlation and agreement metrics

3. **Figure 3: Performance Benchmarking**
   - Speed comparison across various parameter configurations
   - CPU and GPU (when applicable) performance metrics

## Parameter Testing

The benchmark tests various parameter combinations defined in `param_grid.json`:

- Signal dimensions (batch size, channels, segments, duration)
- PAC resolution (phase bands, amplitude bands)
- Computation settings (chunk size, permutations, precision)
- Framework-specific options (gradients, in-place operations, trainability)

Each parameter set is tested multiple times to ensure statistical reliability.

## Usage

### Run Complete Comparison

```bash
# Navigate to the experiment directory
scripts/exp_02_tensorpac_comparison/run_comparison.sh
```

### Run Individual Components

```bash
# Navigate to the experiment directory
cd scripts/exp_02_tensorpac_comparison

# Benchmark performance
python benchmark_vs_tensorpac.py --output_dir results/benchmark --param_grid param_grid.json

# Compare calculation values
python compare_pac_values.py --output_dir results/pac_values --n_trials 10

# Generate figures
python generate_comparison_figures.py \
  --pac_values_dir results/pac_values \
  --benchmark_dir results/benchmark \
  --output_dir results/figures
```

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- Tensorpac
- NumPy
- Matplotlib
- MNGS (internal package)
- Synthetic data from Experiment 01

## Results

The experiment results are organized as follows:

```
results/
├── benchmark/    # Performance benchmark results
├── pac_values/   # PAC calculation comparison results
└── figures/      # Publication-ready figures
    ├── figure1_pac_workflow.png
    ├── figure2_accuracy_comparison.png
    └── figure3_performance_benchmark.png
```

See the [results README](./results/README.md) for more details on the output formats and how to interpret them.

<!-- EOF -->