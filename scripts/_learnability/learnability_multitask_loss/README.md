# MultiTaskLoss PAC Band Learning

This experiment investigates the use of uncertainty-weighted multitask learning to improve the balance between phase and amplitude band learning in Phase-Amplitude Coupling (PAC) analysis.

## Overview

In PAC analysis with trainable frequency bands, we observed that amplitude bands learn more effectively than phase bands, resulting in an imbalance. This experiment implements a solution based on [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Kendall et al., 2017).

The MultiTaskLoss dynamically adjusts the relative importance of the phase and amplitude classification tasks during training based on their uncertainty, providing a more balanced learning process.

## Implementation Details

This experiment:

1. **Implements** the MultiTaskLoss class to automatically balance task weights
2. **Trains** PAC classifiers using:
   - Standard approach (fixed loss weighting)
   - Uncertainty-weighted multitask approach
3. **Tracks** the evolution of:
   - Frequency bands throughout training
   - Task weights assigned by the uncertainty weighting
   - Classification accuracy for both phase and amplitude
4. **Visualizes** and compares the learning process between standard and multitask approaches

## Running the Experiment

```bash
# Run from project root
./scripts/learnability/learnability_multitask_loss/run_experiment.sh
```

Or run with custom parameters:

```bash
python -m scripts.learnability.learnability_multitask_loss.classify_binary_pac_multitask \
  --n_epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --pha_n_bands 10 \
  --amp_n_bands 10 \
  --test_ratio 0.2 \
  --track_interval 5
```

## Understanding the Results

The experiment generates several visualizations:

1. **Convergence Visualization**: Shows how phase and amplitude bands evolve during training
2. **Band Convergence Animation**: Creates an animated visualization of parameter evolution
3. **Standard vs Multitask Comparison**: Directly compares the learning dynamics between approaches
4. **Comparison Report**: Provides quantitative analysis of differences between approaches

Look for these indicators of success:
- More balanced movement in phase vs amplitude parameters
- Dynamic adjustment of task weights during training
- Improved phase band accuracy compared to standard approach
- Better phase/amplitude ratio in the parameter changes

## Methodology

The implementation uses the following approach:

1. **Task Weighting**: Tasks are weighted by the homoscedastic uncertainty of each task
2. **Weight Derivation**:
   - For classification tasks: weight = 1/(σ²)
   - For regression tasks: weight = 1/(2σ²)
3. **Learning Process**: The model learns task weights automatically alongside model parameters
4. **Log-Likelihood**: Task weights are derived from maximizing the Gaussian log-likelihood

## References

Kendall, A., Gal, Y., & Cipolla, R. (2017). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *arXiv preprint arXiv:1705.07115*.