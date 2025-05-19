# Experiment 03: Advanced Statistical Analysis with PAC Distributions

This directory contains code for advanced statistical analysis of Phase-Amplitude Coupling (PAC) data, with a focus on permutation distributions.

## Overview

The experiment extends gPAC functionality to return and analyze full permutation distributions, enabling:

1. **Advanced Statistical Testing**: Beyond default Z-scores
2. **Custom Significance Thresholds**: Multiple threshold calculation methods  
3. **Distribution Analysis**: Statistical properties of null distributions
4. **Visualization Tools**: For inspecting PAC significance

## Scripts

- `return_pac_distributions.py`: Main script implementing distribution analysis functionality
- `run_distribution_analysis.sh`: Shell script to run the complete analysis pipeline

## Features

1. **Return Distribution Option**
   - Extends PAC calculation to return full surrogate distributions
   - Maintains backward compatibility with existing code
   - Supports both single and multi-channel data

2. **Statistical Analysis**
   - Multiple threshold calculation methods:
     - Percentile-based thresholds
     - Z-score based thresholds
     - False Discovery Rate (FDR) correction
   - Normality testing of null distributions
   - Effect size calculations

3. **Visualization Tools**
   - Histogram plots of null distributions
   - QQ plots for distribution normality assessment
   - Threshold comparison visualizations
   - PAC matrix with significance marking

## Statistical Methods

The experiment implements various statistical approaches:

1. **Percentile Method**: Classical non-parametric threshold based on distribution percentiles
2. **Z-Score Method**: Parametric method assuming normal distribution
3. **FDR Correction**: Controls false discovery rate in multiple comparisons
4. **Effect Size**: Cohen's d calculation for PAC effect magnitude estimation

## Usage

### Run Complete Analysis

```bash
# Navigate to the experiment directory
cd scripts/exp_03_advanced_statistics

# Run analysis with default settings
./run_distribution_analysis.sh

# Run with custom output location
./run_distribution_analysis.sh --output_dir /path/to/results
```

### Use in Custom Code

```python
from scripts.exp_03_advanced_statistics.return_pac_distributions import (
    calculate_enhanced_pac, PACDistributionAnalyzer
)

# Calculate PAC with distributions
pac_values, surrogate_dist, pha_freqs, amp_freqs = calculate_enhanced_pac(
    signal=my_signal,
    fs=1000,
    n_perm=200,
    return_dist=True
)

# Analyze distributions
analyzer = PACDistributionAnalyzer(n_perm=200, alpha=0.05)
thresholds = analyzer.calculate_thresholds(surrogate_dist)

# Visualize
analyzer.visualize_distributions(
    pac_values, surrogate_dist, pha_freqs, amp_freqs,
    save_path='./my_distributions.png'
)
```

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- Matplotlib
- MNGS (internal package)
- Core gPAC functionality

## Results

The analysis produces several outputs:

- Statistical threshold comparisons for PAC significance testing
- Visualizations of PAC null distributions
- QQ plots for assessing distribution normality
- Effect size measurements for significant PAC connections

These results enable more detailed and rigorous statistical assessment of PAC measurements, particularly for neuroscience research requiring robust significance testing.