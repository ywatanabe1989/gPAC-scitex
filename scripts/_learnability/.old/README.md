# gPAC Learnability Demonstrations

This directory contains scripts to demonstrate the learnability advantages of the gPAC module with trainable parameters.

## Overview

The gPAC module is designed to measure Phase-Amplitude Coupling (PAC) in neural signals, with the unique capability of having trainable frequency bands. This allows the module to adapt its parameters to better detect PAC in specific signals, which can improve performance in real-world neural signal processing applications.

## Scripts

The learnability demonstration consists of three main scripts:

1. **`generate_synthetic_data.py`**: Generates synthetic PAC signals with known coupling frequencies for training and testing.
2. **`classify_demo_signals_using_gPAC_module.py`**: Trains classifiers using both trainable and fixed gPAC modules to identify phase and amplitude frequencies in the signals, comparing performance across different noise levels.
3. **`visualize_learnability.py`**: Visualizes the parameter adaptation process of the trainable gPAC module, showing how it learns to better detect specific coupling frequencies.

## Usage

Each script can be run independently with various command-line parameters:

### Generating Synthetic Data

```bash
python generate_synthetic_data.py [--pha_freqs PHA_FREQS [PHA_FREQS ...]] [--amp_freqs AMP_FREQS [AMP_FREQS ...]] [--n_samples N_SAMPLES]
```

### Classifying Signals

```bash
python classify_demo_signals_using_gPAC_module.py [--data_path DATA_PATH] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--pha_n_bands PHA_N_BANDS] [--amp_n_bands AMP_N_BANDS] [--test_ratio TEST_RATIO]
```

### Visualizing Learnability

```bash
python visualize_learnability.py [--target_pha_freq TARGET_PHA_FREQ] [--target_amp_freq TARGET_AMP_FREQ] [--n_epochs N_EPOCHS] [--learning_rate LEARNING_RATE] [--pha_n_bands PHA_N_BANDS] [--amp_n_bands AMP_N_BANDS] [--noise_level NOISE_LEVEL]
```

## Default Parameters

The scripts use the following default parameters:

### Synthetic Data Generation
- Phase frequencies: 4.0 Hz, 8.0 Hz, 12.0 Hz
- Amplitude frequencies: 80.0 Hz, 100.0 Hz, 120.0 Hz
- Noise levels: 0.1, 0.2, 0.3
- 20 samples per frequency combination

### Classification
- 50 training epochs
- Batch size of 32
- Learning rate of 0.01
- 10 phase and amplitude frequency bands
- 20% test data ratio

### Learnability Visualization
- Target phase frequency: 8.0 Hz
- Target amplitude frequency: 100.0 Hz
- 100 training epochs
- Learning rate of 0.01
- 10 phase and amplitude frequency bands
- Noise level of 0.1

## Output

The scripts generate the following outputs:

### Synthetic Data
- Saved as both PyTorch tensor (.pt) and NumPy array (.npz) formats in the `data/` directory
- Example signal visualizations

### Classification
- Model training history plots
- Trained model weights
- Performance comparison across noise levels
- Comprehensive classification report

### Learnability Visualization
- Frequency band visualizations (before and after training)
- Parameter adaptation plots
- Performance metrics and error reduction analysis
- Detailed adaptation report

## Dependencies

- torch
- numpy
- matplotlib
- seaborn
- pandas
- mngs
- gpac

## References

For more information about Phase-Amplitude Coupling and the gPAC module, please refer to:

- The gPAC GitHub repository
- [relevant papers on PAC in neural signals]
- [documentation on trainable neural signal processing]
EOL < /dev/null
