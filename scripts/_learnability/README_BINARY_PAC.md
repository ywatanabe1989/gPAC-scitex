# Binary PAC Analysis Guide

This README provides instructions for using the binary PAC analysis tools to identify and visualize discriminative PAC features between different signal classes, and understand the learnability of these features.

## Overview

The binary PAC analysis system consists of four key scripts:

1. `generate_binary_pac_data.py` - Generates synthetic PAC signals with two distinct classes:
   - Class A: Low-frequency phase (4-8 Hz) coupled with medium-frequency amplitude (80-100 Hz)
   - Class B: Medium-frequency phase (10-14 Hz) coupled with high-frequency amplitude (120-150 Hz)

2. `visualize_pac_features.py` - Creates comprehensive visualizations of PAC features:
   - Extracts PAC features from signals using the gPAC module
   - Identifies discriminative patterns between classes
   - Generates publication-quality visualizations

3. `visualize_pac_learnability.py` - Tracks how PAC features evolve during training:
   - Shows how frequency bands adapt during training
   - Visualizes the optimization path of bandpass filter parameters
   - Demonstrates how bands converge to discriminative regions
   - Creates animations of feature evolution

4. `classify_binary_pac_signals.py` - Trains and evaluates binary classifiers with:
   - Trainable PAC parameters (learns optimal frequency bands)
   - Fixed PAC parameters (uses predefined frequency bands)

## Key Visualizations

The system produces several informative visualizations:

1. **Signal Examples**:
   - `class_examples.png` - Shows example signals from each class with their PAC parameters

2. **PAC Feature Analysis**:
   - `pac_class_means.png` - Heatmaps showing mean PAC values for each class and their difference
   - `discriminative_features.png` - Highlights the most discriminative PAC features
   - `feature_importance_3d.png` - 3D surface plot of feature importance by frequency band

3. **Feature Space Visualization**:
   - `tsne_projection.png` - t-SNE projection of PAC features showing class separation

4. **Learnability Visualization**:
   - `band_evolution.png` - Shows how phase and amplitude frequency bands change during training
   - `optimization_path.png` - Visualizes the path taken by frequency bands in 2D space
   - `accuracy_vs_bands.png` - Shows relationship between accuracy and band optimization
   - `feature_convergence.gif` - Animation showing how bands converge to optimal regions

5. **Classification Analysis** (when using classify_binary_pac_signals.py):
   - `accuracy_by_epoch.png` - Training and validation accuracy over epochs
   - `accuracy_by_iteration.png` - Batch accuracy changes with iterations
   - `frequency_band_map.png` - 2D visualization of learned frequency bands

## Setup Requirements

Ensure your Python environment includes these packages:
- torch
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- scikit-learn
- mngs
- gpac
- tqdm

## Running the Analysis

1. **Complete Pipeline**:
   ```bash
   ./run_binary_pac_analysis.sh
   ```
   This runs the entire pipeline: data generation, feature visualization, and learnability visualization.

2. **Individual Steps**:

   a. Generate binary PAC data:
   ```bash
   python -m scripts.learnability.generate_binary_pac_data --n_samples 100
   ```

   b. Visualize PAC features:
   ```bash
   python -m scripts.learnability.visualize_pac_features --pha_n_bands 10 --amp_n_bands 10 --save_figs
   ```

   c. Visualize PAC learnability:
   ```bash
   python -m scripts.learnability.visualize_pac_learnability --pha_n_bands 5 --amp_n_bands 5 --n_epochs 10
   ```

   d. (Optional) Run binary classification:
   ```bash
   python -m scripts.learnability.classify_binary_pac_signals --n_epochs 30 --pha_n_bands 8 --amp_n_bands 8
   ```

## Visualization Details

### 1. Class Examples
Shows example signals from each class with their unique PAC properties:
- Top row: Class A signals (phase: 4-8 Hz, amplitude: 80-100 Hz)
- Bottom row: Class B signals (phase: 10-14 Hz, amplitude: 120-150 Hz)
- Includes noise level information for each signal

### 2. PAC Features
Shows PAC values distribution across frequency bands:
- Class A Mean: Average PAC values for Class A signals
- Class B Mean: Average PAC values for Class B signals
- Difference: Class B - Class A, highlighting discriminative regions

### 3. Discriminative Features
Heat map showing absolute difference between classes with:
- Contour lines highlighting regions of interest
- Star markers indicating top discriminative frequency pairs
- Translucent overlays showing expected class regions

### 4. t-SNE Projection
2D projection of high-dimensional PAC features:
- Each point represents one signal
- Colors indicate class membership
- Clustering shows separability of classes based on PAC features

### 5. Learnability Visualizations

#### Band Evolution
Shows how frequency bands change over training iterations:
- Separate plots for phase and amplitude bands
- Initial and final band positions
- Direction and magnitude of changes
- Highlighted regions showing ground truth class ranges

#### Optimization Path
2D visualization of band parameter changes:
- Each point represents a phase-amplitude frequency pair
- Arrows show direction of optimization
- Color-coded by band index
- Highlighted regions showing ground truth class distributions

#### Accuracy vs. Bands
Shows relationship between model accuracy and band optimization:
- Top panel: Classification accuracy over training
- Middle panel: Distance between phase bands and optimal regions
- Bottom panel: Distance between amplitude bands and optimal regions

#### Feature Convergence Animation
Animated visualization showing band evolution:
- Dynamically shows how bands move during training
- Demonstrates convergence to discriminative regions
- Visualizes the optimization process in real-time

## Customization

You can customize the analysis with these parameters:

For data generation:
- `--n_samples`: Number of samples per class

For feature visualization:
- `--pha_n_bands`: Number of phase frequency bands
- `--amp_n_bands`: Number of amplitude frequency bands
- `--dpi`: Resolution of saved figures

For learnability visualization:
- `--n_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimization
- `--track_interval`: Interval for tracking band changes

For classification:
- `--hidden_size`: Size of hidden layer in classifier

## Directory Structure

Results are organized as follows:
- `data/`: Contains generated binary PAC signals
- `results/visualizations/`: Static PAC feature visualizations
- `results/learnability/`: Learnability visualizations and animations
- `results/binary_figures/`: Classification results (if applicable)