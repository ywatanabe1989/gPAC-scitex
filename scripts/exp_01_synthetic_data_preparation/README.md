# Experiment 01: Synthetic PAC Data Preparation

This directory contains code for generating synthetic Phase-Amplitude Coupling (PAC) data with different coupling patterns for use in further experiments.

## Overview

The synthetic data generation creates 5 distinct classes of PAC signals with the following characteristics:

1. **Class 1**: Low phase frequency (4-8 Hz, theta) coupled with low amplitude frequency (50-70 Hz, low gamma)
2. **Class 2**: Low phase frequency (4-8 Hz, theta) coupled with high amplitude frequency (150-170 Hz, high gamma)
3. **Class 3**: Medium phase frequency (9-14 Hz, alpha) coupled with medium amplitude frequency (100-120 Hz, mid gamma)
4. **Class 4**: High phase frequency (15-20 Hz, beta) coupled with low amplitude frequency (50-70 Hz, low gamma)
5. **Class 5**: High phase frequency (15-20 Hz, beta) coupled with high amplitude frequency (150-170 Hz, high gamma)

The data is generated with varying levels of noise and coupling strengths to provide robust training and testing sets.

## Scripts

- `generate_synthetic_data.py`: Main script for generating the synthetic PAC data
- `test_data_generation.py`: Test script to verify data generation correctness

## Dataset Structure

The generation process produces several outputs:

1. **Raw Signals**:
   - `synthetic_pac_signals.npz`: NumPy-based storage
   - `synthetic_pac_signals.pt`: PyTorch-based storage

2. **PyTorch Dataset**:
   - `synthetic_pac_dataset.pt`: Full dataset
   - `train_dataset.pt`: Training split (70%)
   - `val_dataset.pt`: Validation split (15%)
   - `test_dataset.pt`: Test split (15%)

3. **Visualizations**:
   - `synthetic_pac_examples.png`: Example signals from each class
   - `frequency_space.png`: Visualization of the phase-amplitude frequency space

## Dataset Format

Each dataset contains:

- **Signals**: Tensor of shape `[n_samples, n_channels, n_segments, seq_len]`
- **Labels**: Tensor of shape `[n_samples]` with class IDs (0-4)
- **Metadata**:
  - `pha_freqs`: Phase modulation frequencies
  - `amp_freqs`: Amplitude carrier frequencies
  - `noise_levels`: Applied noise levels
  - `coupling_strengths`: Coupling strength values
  - `sample_ids`: Sample identifiers

## Usage

### Generate Data

```bash
# Generate with default parameters
python ./scripts/exp_01_synthetic_data_preparation/generate_synthetic_data.py

# Generate with custom parameters
python ./scripts/exp_01_synthetic_data_preparation/generate_synthetic_data.py --n_samples 300 --n_channels 8 --duration 3.0
```

### Test Data Generation

```bash
# Run tests on generated data
python ./scripts/exp_01_synthetic_data_preparation/test_data_generation.py

# Specify custom data directory
python ./scripts/exp_01_synthetic_data_preparation/test_data_generation.py --data_dir /path/to/data
```

### Load Dataset in PyTorch

```python
import torch
from torch.utils.data import DataLoader

# Load dataset
train_dataset = torch.load('./scripts/exp_01_synthetic_data_preparation/data/train_dataset.pt')
val_dataset = torch.load('./scripts/exp_01_synthetic_data_preparation/data/val_dataset.pt')
test_dataset = torch.load('./scripts/exp_01_synthetic_data_preparation/data/test_dataset.pt')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Iterate through batches
for signals, labels in train_loader:
    # signals shape: [batch_size, n_channels, n_segments, seq_len]
    # labels shape: [batch_size]
    pass
```

## Parameters

The dataset generation accepts the following parameters:

- `n_samples`: Number of samples per class (default: 200)
- `n_channels`: Number of channels per sample (default: 4)
- `n_segments`: Number of segments per channel (default: 1)
- `duration`: Duration of each signal in seconds (default: 2.0)
- `fs`: Sampling frequency in Hz (default: 1000.0)

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- MNGS (internal package)