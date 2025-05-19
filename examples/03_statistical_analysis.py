#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 11:32:45 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/examples/03_statistical_analysis.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/03_statistical_analysis.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Comprehensive statistical analysis example with multiple comparison correction.

This example demonstrates:
1. Creating synthetic signals with and without PAC
2. Calculating PAC with surrogate distributions
3. Extracting p-values from surrogate distributions
4. Applying different multiple comparison correction methods
5. Visualizing threshold-masked PAC values
6. Comparing different statistical approaches
7. Evaluating sensitivity and specificity

Requirements:
- gpac
- numpy
- torch
- matplotlib
- seaborn
- statsmodels (for multiple comparison correction)
- scikit-learn (for ROC analysis)
"""

import gpac
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats

# For multiple comparison correction
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not found. Will use basic Bonferroni correction.")

# For ROC analysis
try:
    from sklearn.metrics import roc_curve, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not found. ROC analysis will be skipped.")

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create output directory for figures
os.makedirs("figures", exist_ok=True)

# Parameters
FS = 1000  # Sampling frequency in Hz
DURATION = 10  # Signal duration in seconds
N_TRIALS = 10  # Number of trials
N_SUBJECTS = 6  # Number of subjects (half with PAC, half without)

# Define true PAC parameters for the positive cases
PAC_PARAMS = [
    (6, 80, 0.8),    # Strong theta-gamma coupling
    (8, 60, 0.6),    # Medium alpha-gamma coupling
    (4.5, 100, 0.7)  # Strong delta-high gamma coupling
]

def create_synthetic_dataset(
    fs: float,
    duration: float,
    n_trials: int,
    n_subjects: int,
    pac_params: List[Tuple[float, float, float]],
    with_contamination: bool = False,
    noise_level: float = 0.3,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create a synthetic dataset with positive and negative examples.
    
    Args:
        fs: Sampling frequency in Hz
        duration: Signal duration in seconds
        n_trials: Number of trials per subject
        n_subjects: Number of subjects (half with PAC, half without)
        pac_params: List of (phase_hz, amplitude_hz, strength) tuples
        with_contamination: Whether to add weak PAC to control subjects
        noise_level: Strength of noise to add
        seed: Random seed
        
    Returns:
        Dictionary with dataset information and signals
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Time vector
    t = np.arange(0, duration, 1/fs)
    time_points = len(t)
    
    # Distribute subjects half with PAC, half without
    n_pac_subjects = n_subjects // 2
    n_control_subjects = n_subjects - n_pac_subjects
    
    # Create empty tensor for signals: (subjects, trials, time)
    signals = torch.zeros((n_subjects, n_trials, time_points), dtype=torch.float32)
    
    # Create dictionary to store ground truth
    ground_truth = {
        'subject_has_pac': np.zeros(n_subjects, dtype=bool),  # True if subject has PAC
        'pac_params': [None] * n_subjects,  # Parameters used for each subject
        'control_contamination': np.zeros(n_subjects, dtype=bool)  # Contaminated control
    }
    
    # Generate signals for subjects with PAC
    for subj_idx in range(n_pac_subjects):
        # Select PAC parameters for this subject from the list
        pha_freq, amp_freq, strength = pac_params[subj_idx % len(pac_params)]
        ground_truth['subject_has_pac'][subj_idx] = True
        ground_truth['pac_params'][subj_idx] = (pha_freq, amp_freq, strength)
        
        for trial in range(n_trials):
            # Phase signal (slow oscillation)
            # Add small random frequency variation between trials
            phase_freq_jitter = pha_freq * (1 + 0.05 * np.random.randn())
            phase_signal = np.sin(2 * np.pi * phase_freq_jitter * t + np.random.rand() * np.pi)
            
            # Amplitude signal (fast oscillation)
            # Add small random frequency variation between trials
            amp_freq_jitter = amp_freq * (1 + 0.02 * np.random.randn())
            amp_carrier = np.sin(2 * np.pi * amp_freq_jitter * t + np.random.rand() * np.pi)
            
            # Modulate amplitude by phase
            amplitude_modulation = (1 + strength * phase_signal) / 2
            pac_signal = amplitude_modulation * amp_carrier
            
            # Add phase signal and noise
            noise = np.random.normal(0, noise_level, len(t))
            signal = phase_signal * 0.3 + pac_signal * 0.5 + noise
            
            # Store in tensor
            signals[subj_idx, trial, :] = torch.from_numpy(signal.astype(np.float32))
    
    # Generate signals for control subjects
    for subj_idx in range(n_pac_subjects, n_subjects):
        # Sometimes add very weak PAC to control subjects if contamination is enabled
        if with_contamination and np.random.rand() < 0.5:
            # Choose random PAC parameters with weak coupling
            pha_freq = np.random.uniform(2, 12)
            amp_freq = np.random.uniform(50, 150)
            strength = np.random.uniform(0.05, 0.2)  # Very weak coupling
            ground_truth['control_contamination'][subj_idx] = True
            ground_truth['pac_params'][subj_idx] = (pha_freq, amp_freq, strength)
        
        for trial in range(n_trials):
            # Multiple oscillations without coupling
            frequencies = np.random.uniform(2, 20, size=5)  # Low frequencies
            phases = np.random.uniform(0, 2*np.pi, size=5)
            low_freq_signal = np.zeros_like(t)
            for freq, phase in zip(frequencies, phases):
                low_freq_signal += np.sin(2 * np.pi * freq * t + phase)
            low_freq_signal /= 5  # Normalize
            
            frequencies = np.random.uniform(40, 160, size=5)  # High frequencies
            phases = np.random.uniform(0, 2*np.pi, size=5)
            high_freq_signal = np.zeros_like(t)
            for freq, phase in zip(frequencies, phases):
                high_freq_signal += np.sin(2 * np.pi * freq * t + phase)
            high_freq_signal /= 5  # Normalize
            
            # Combine signals
            if ground_truth['control_contamination'][subj_idx]:
                # Add very weak coupling if this is a contaminated control
                pha_freq, amp_freq, strength = ground_truth['pac_params'][subj_idx]
                phase_signal = np.sin(2 * np.pi * pha_freq * t + np.random.rand() * np.pi)
                amplitude_modulation = (1 + strength * phase_signal) / 2
                pac_signal = amplitude_modulation * np.sin(2 * np.pi * amp_freq * t)
                
                # Add to signal with very low weight
                signal = low_freq_signal * 0.4 + high_freq_signal * 0.4 + pac_signal * 0.1
            else:
                signal = low_freq_signal * 0.4 + high_freq_signal * 0.4
            
            # Add noise
            noise = np.random.normal(0, noise_level, len(t))
            signal += noise
            
            # Store in tensor
            signals[subj_idx, trial, :] = torch.from_numpy(signal.astype(np.float32))
    
    # Reshape to match gpac expected format: (batch=subjects, channels=1, trials, time)
    signals_gpac = signals.unsqueeze(1)
    
    return {
        'signals': signals_gpac,
        't': t,
        'ground_truth': ground_truth,
        'fs': fs,
        'duration': duration,
        'n_trials': n_trials,
        'n_subjects': n_subjects
    }

def calculate_pac_for_dataset(dataset: Dict[str, Any], device: str = 'cpu') -> Dict[str, Any]:
    """
    Calculate PAC for all subjects in the dataset.
    
    Args:
        dataset: Dataset dictionary from create_synthetic_dataset
        device: Device to use for computation ('cpu' or 'cuda')
        
    Returns:
        Dictionary with PAC results
    """
    signals = dataset['signals']
    fs = dataset['fs']
    n_subjects = dataset['n_subjects']
    
    # PAC calculation parameters
    pha_start_hz = 2.0
    pha_end_hz = 20.0
    pha_n_bands = 15
    amp_start_hz = 40.0
    amp_end_hz = 160.0
    amp_n_bands = 15
    n_perm = 200
    
    # Results storage
    results = {
        'pac_values': [],
        'surrogate_dists': [],
        'pha_freqs': None,
        'amp_freqs': None,
        'p_values': []
    }
    
    # Process each subject
    for subj_idx in range(n_subjects):
        print(f"Calculating PAC for subject {subj_idx+1}/{n_subjects}...")
        
        # Extract this subject's data
        signal = signals[subj_idx:subj_idx+1]  # Keep batch dimension
        
        # Move to device
        signal_device = signal.to(device) if device == 'cuda' else signal
        
        # Calculate PAC with surrogate distribution
        pac_values, surrogate_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal=signal_device,
            fs=fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            return_dist=True,
            device=device,
            fp16=False  # Use fp32 for better precision
        )
        
        # Store frequency information on first pass
        if results['pha_freqs'] is None:
            results['pha_freqs'] = pha_freqs
            results['amp_freqs'] = amp_freqs
        
        # Calculate p-values from surrogate distribution
        # First ensure tensors are on CPU
        pac_cpu = pac_values[0, 0].cpu()
        surrogate_cpu = surrogate_dist[:, 0, 0].cpu()
        
        # Convert to numpy for calculations
        pac_np = pac_cpu.numpy()  # Remove batch and channel dimensions
        surrogate_np = surrogate_cpu.numpy()
        
        # Calculate p-values
        p_values = np.zeros_like(pac_np)
        
        # Check dimensions of pac_np and adjust loop accordingly
        if len(pac_np.shape) == 1:
            # If pac_np is 1D, reshape it to 2D with one dimension
            n_pha = len(results['pha_freqs'])
            n_amp = len(results['amp_freqs'])
            pac_np = pac_np.reshape(n_pha, n_amp)
            surrogate_np = surrogate_np.reshape(surrogate_np.shape[0], n_pha, n_amp)
        
        for i in range(pac_np.shape[0]):
            for j in range(pac_np.shape[1]):
                # One-sided p-value: proportion of surrogates >= observed
                p_values[i, j] = np.mean(surrogate_np[:, i, j] >= pac_np[i, j])
        
        # Store results
        results['pac_values'].append(pac_values[0, 0].cpu())  # Store as tensor
        results['surrogate_dists'].append(surrogate_dist[:, 0, 0].cpu())
        results['p_values'].append(p_values)
    
    # Convert lists to tensors/arrays
    results['pac_values'] = torch.stack(results['pac_values'])
    results['surrogate_dists'] = torch.stack(results['surrogate_dists'])
    results['p_values'] = np.array(results['p_values'])
    
    return results

def apply_multiple_comparison_corrections(p_values: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply different multiple comparison correction methods to p-values.
    
    Args:
        p_values: Array of p-values with shape (n_subjects, n_phase, n_amplitude)
        
    Returns:
        Dictionary with corrected p-values
    """
    n_subjects, n_phase, n_amplitude = p_values.shape
    n_tests = n_phase * n_amplitude
    
    # Results storage
    corrected = {
        'bonferroni': np.zeros_like(p_values),
        'fdr_bh': np.zeros_like(p_values) if HAS_STATSMODELS else None,
        'fdr_by': np.zeros_like(p_values) if HAS_STATSMODELS else None,
        'holm': np.zeros_like(p_values) if HAS_STATSMODELS else None
    }
    
    # Apply corrections for each subject
    for subj_idx in range(n_subjects):
        # Get p-values for this subject
        subj_p = p_values[subj_idx].flatten()
        
        # 1. Bonferroni correction
        bonferroni_p = np.minimum(subj_p * n_tests, 1.0)
        corrected['bonferroni'][subj_idx] = bonferroni_p.reshape(n_phase, n_amplitude)
        
        # 2. Other corrections from statsmodels
        if HAS_STATSMODELS:
            # False Discovery Rate (FDR) - Benjamini/Hochberg
            _, fdr_bh_p, _, _ = multipletests(subj_p, method='fdr_bh')
            corrected['fdr_bh'][subj_idx] = fdr_bh_p.reshape(n_phase, n_amplitude)
            
            # False Discovery Rate (FDR) - Benjamini/Yekutieli (more conservative)
            _, fdr_by_p, _, _ = multipletests(subj_p, method='fdr_by')
            corrected['fdr_by'][subj_idx] = fdr_by_p.reshape(n_phase, n_amplitude)
            
            # Holm-Bonferroni method
            _, holm_p, _, _ = multipletests(subj_p, method='holm')
            corrected['holm'][subj_idx] = holm_p.reshape(n_phase, n_amplitude)
    
    return corrected

def create_binary_significance_masks(
    corrected_p_values: Dict[str, np.ndarray], 
    alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Create binary masks of significant PAC values using different corrections.
    
    Args:
        corrected_p_values: Dictionary of corrected p-values
        alpha: Significance threshold
        
    Returns:
        Dictionary with binary masks
    """
    masks = {}
    for method, p_vals in corrected_p_values.items():
        if p_vals is not None:
            masks[method] = (p_vals < alpha).astype(np.int32)
    
    return masks

def plot_ground_truth_comparison(
    dataset: Dict[str, Any],
    pac_results: Dict[str, Any],
    masks: Dict[str, np.ndarray],
    output_file: str = 'figures/03_statistical_comparison.png'
) -> None:
    """
    Plot comparison of ground truth and statistical results.
    
    Args:
        dataset: Dataset dictionary from create_synthetic_dataset
        pac_results: Results from calculate_pac_for_dataset
        masks: Significance masks from create_binary_significance_masks
        output_file: Path to save the figure
    """
    n_subjects = dataset['n_subjects']
    ground_truth = dataset['ground_truth']
    pac_values = pac_results['pac_values']
    pha_freqs = pac_results['pha_freqs']
    amp_freqs = pac_results['amp_freqs']
    
    # Determine number of rows and columns for subplots
    methods = list(masks.keys())
    n_methods = len(methods)
    
    # Create figure with n_subjects rows and 1+n_methods columns
    # (one for raw PAC, n_methods for each correction method)
    fig, axes = plt.subplots(
        n_subjects, 1+n_methods, 
        figsize=(4 * (1+n_methods), 3 * n_subjects),
        gridspec_kw={'width_ratios': [1] + [1] * n_methods}
    )
    
    # Create a custom colormap that highlights significant values
    colors = [(0.95, 0.95, 0.95), (1, 1, 0.7), (1, 0.7, 0.5), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('pac_cmap', colors, N=100)
    
    # Plot each subject
    for subj_idx in range(n_subjects):
        # Get subject info and results
        has_pac = ground_truth['subject_has_pac'][subj_idx]
        params = ground_truth['pac_params'][subj_idx]
        contaminated = ground_truth['control_contamination'][subj_idx]
        pac_data = pac_values[subj_idx].numpy()
        
        # Get global min and max for colormap
        vmin = 0  # Threshold negative Z-scores
        vmax = np.max(pac_data) * 1.1
        
        # 1. Raw PAC values
        ax = axes[subj_idx, 0]
        im = ax.imshow(
            pac_data,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation='none'
        )
        
        # Plot target PAC frequency if known
        if params is not None:
            pha_target, amp_target, _ = params
            pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
            amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
            
            ax.plot(
                amp_idx, pha_idx,
                'o', markerfacecolor='none', markeredgecolor='black',
                markersize=10, markeredgewidth=1.5
            )
        
        # Add title with subject info
        if has_pac:
            ax.set_title(f"Subject {subj_idx+1}: PAC+ ({params[0]:.1f}-{params[1]:.1f} Hz)")
        elif contaminated:
            ax.set_title(f"Subject {subj_idx+1}: Control (contaminated)")
        else:
            ax.set_title(f"Subject {subj_idx+1}: Control")
        
        # Set axis labels for first row
        if subj_idx == 0:
            ax.set_title(f"Raw PAC Z-scores\n{ax.get_title()}")
        
        # Set axis labels for last row
        if subj_idx == n_subjects - 1:
            ax.set_xlabel('Amplitude Frequency (Hz)')
        
        # Set y-axis label
        ax.set_ylabel('Phase Frequency (Hz)')
        
        # Set ticks
        ax.set_xticks(np.arange(0, len(amp_freqs), len(amp_freqs)//4))
        ax.set_xticklabels([f"{f:.0f}" for f in amp_freqs[::len(amp_freqs)//4]])
        ax.set_yticks(np.arange(0, len(pha_freqs), len(pha_freqs)//4))
        ax.set_yticklabels([f"{f:.0f}" for f in pha_freqs[::len(pha_freqs)//4]])
        
        # 2. Method-specific masked PAC values
        for m_idx, method in enumerate(methods):
            ax = axes[subj_idx, m_idx + 1]
            
            # Create masked array
            mask = masks[method][subj_idx]
            masked_pac = pac_data * mask
            
            # Create masked array for plotting
            masked_array = np.ma.array(masked_pac, mask=(masked_pac == 0))
            
            im = ax.imshow(
                masked_array,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                interpolation='none'
            )
            
            # Add markers for target frequencies if known
            if params is not None:
                pha_target, amp_target, _ = params
                pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
                amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
                
                ax.plot(
                    amp_idx, pha_idx,
                    'o', markerfacecolor='none', markeredgecolor='black',
                    markersize=10, markeredgewidth=1.5
                )
            
            # Add title with method name
            if subj_idx == 0:
                ax.set_title(f"{method.replace('_', '-').capitalize()}\n{axes[subj_idx, 0].get_title()}")
            else:
                ax.set_title(f"{axes[subj_idx, 0].get_title()}")
            
            # Set axis labels for last row
            if subj_idx == n_subjects - 1:
                ax.set_xlabel('Amplitude Frequency (Hz)')
            
            # Set ticks
            ax.set_xticks(np.arange(0, len(amp_freqs), len(amp_freqs)//4))
            ax.set_xticklabels([f"{f:.0f}" for f in amp_freqs[::len(amp_freqs)//4]])
            ax.set_yticks([])  # No y-ticks for method columns
    
    # Add a colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('PAC Z-score')
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_statistical_performance(
    dataset: Dict[str, Any],
    masks: Dict[str, np.ndarray],
    pac_results: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate statistical performance of different correction methods.
    
    Args:
        dataset: Dataset dictionary from create_synthetic_dataset
        masks: Significance masks from create_binary_significance_masks
        pac_results: Results from calculate_pac_for_dataset
        
    Returns:
        Dictionary with performance metrics for each method
    """
    ground_truth = dataset['ground_truth']
    pha_freqs = pac_results['pha_freqs']
    amp_freqs = pac_results['amp_freqs']
    
    # Results storage
    results = {method: {} for method in masks.keys()}
    
    # For each method, evaluate:
    # 1. True positive rate (sensitivity): % of true PAC detected
    # 2. False positive rate: % of false positives in control subjects
    # 3. Accuracy: Overall correct classifications
    for method, method_masks in masks.items():
        # Track performance metrics
        n_subjects = len(ground_truth['subject_has_pac'])
        true_positives = 0
        true_positives_total = 0
        false_positives = 0
        false_positives_total = 0
        
        # Evaluate each subject
        for subj_idx in range(n_subjects):
            has_pac = ground_truth['subject_has_pac'][subj_idx]
            params = ground_truth['pac_params'][subj_idx]
            mask = method_masks[subj_idx]
            
            if has_pac:
                # For PAC-positive subjects, we expect a hit at the target frequency
                pha_target, amp_target, _ = params
                pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
                amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
                
                # Check if target frequency is significant
                if mask[pha_idx, amp_idx] > 0:
                    true_positives += 1
                true_positives_total += 1
            else:
                # For control subjects, we don't expect any significant results
                # Count total significant points
                false_positives += np.sum(mask)
                false_positives_total += mask.size
        
        # Calculate metrics
        sensitivity = true_positives / true_positives_total if true_positives_total > 0 else 0
        specificity = 1 - (false_positives / false_positives_total) if false_positives_total > 0 else 1
        
        # Store results
        results[method]['sensitivity'] = sensitivity
        results[method]['specificity'] = specificity
        results[method]['true_positives'] = true_positives
        results[method]['total_positives'] = true_positives_total
        results[method]['false_positives'] = false_positives
        results[method]['total_tests_negative'] = false_positives_total
    
    return results

def plot_performance_comparison(
    performance: Dict[str, Dict[str, float]],
    output_file: str = 'figures/03_method_performance.png'
) -> None:
    """
    Plot performance comparison of different correction methods.
    
    Args:
        performance: Performance metrics from evaluate_statistical_performance
        output_file: Path to save the figure
    """
    # Extract data
    methods = list(performance.keys())
    sensitivity = [performance[m]['sensitivity'] for m in methods]
    specificity = [performance[m]['specificity'] for m in methods]
    
    # Format method names for display
    display_methods = [m.replace('_', '-').capitalize() for m in methods]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot bar chart
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, sensitivity, width, label='Sensitivity', color='#2C7BB6')
    plt.bar(x + width/2, specificity, width, label='Specificity', color='#D7191C')
    
    # Customize plot
    plt.xlabel('Correction Method')
    plt.ylabel('Performance')
    plt.title('Statistical Performance Comparison')
    plt.xticks(x, display_methods)
    plt.ylim(0, 1.05)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(sensitivity):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    for i, v in enumerate(specificity):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    # Add explanation text
    plt.figtext(
        0.5, 0.01, 
        "Sensitivity: True positive rate (higher is better)\n"
        "Specificity: 1 - False positive rate (higher is better)",
        ha='center', fontsize=10
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_comparison(
    dataset: Dict[str, Any],
    pac_results: Dict[str, Any],
    output_file: str = 'figures/03_roc_comparison.png'
) -> None:
    """
    Plot ROC curve comparison using different frequency-band approaches.
    
    Args:
        dataset: Dataset dictionary from create_synthetic_dataset
        pac_results: Results from calculate_pac_for_dataset
        output_file: Path to save the figure
    """
    if not HAS_SKLEARN:
        print("Skipping ROC analysis as sklearn is not available.")
        return
    
    ground_truth = dataset['ground_truth']
    pac_values = pac_results['pac_values'].numpy()
    pha_freqs = pac_results['pha_freqs']
    amp_freqs = pac_results['amp_freqs']
    
    # Get binary ground truth: subjects with PAC
    y_true = ground_truth['subject_has_pac']
    n_subjects = len(y_true)
    
    # Create several scoring approaches:
    # 1. Maximum Z-score across all frequency pairs
    # 2. Target frequency Z-score (where true PAC is expected)
    # 3. Mean Z-score in theta-gamma bands
    # 4. Mean Z-score in alpha-gamma bands
    scores = {
        'max_zscore': np.zeros(n_subjects),
        'target_zscore': np.zeros(n_subjects),
        'theta_gamma': np.zeros(n_subjects),
        'alpha_gamma': np.zeros(n_subjects)
    }
    
    # Calculate scores for each subject
    for subj_idx in range(n_subjects):
        pac_data = pac_values[subj_idx]
        
        # 1. Maximum Z-score
        scores['max_zscore'][subj_idx] = np.max(pac_data)
        
        # 2. Target frequency Z-score (if available)
        params = ground_truth['pac_params'][subj_idx]
        if params is not None:
            pha_target, amp_target, _ = params
            pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
            amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
            scores['target_zscore'][subj_idx] = pac_data[pha_idx, amp_idx]
        
        # 3. Mean theta-gamma Z-score
        theta_mask = (pha_freqs >= 4) & (pha_freqs <= 8)
        gamma_mask = (amp_freqs >= 30) & (amp_freqs <= 100)
        theta_gamma_values = pac_data[np.ix_(theta_mask, gamma_mask)]
        scores['theta_gamma'][subj_idx] = np.mean(theta_gamma_values)
        
        # 4. Mean alpha-gamma Z-score
        alpha_mask = (pha_freqs >= 8) & (pha_freqs <= 13)
        alpha_gamma_values = pac_data[np.ix_(alpha_mask, gamma_mask)]
        scores['alpha_gamma'][subj_idx] = np.mean(alpha_gamma_values)
    
    # Calculate ROC curves and AUC
    plt.figure(figsize=(10, 8))
    
    for name, score_values in scores.items():
        # Skip target score if no PAC subjects
        if name == 'target_zscore' and not any(y_true):
            continue
            
        fpr, tpr, thresholds = roc_curve(y_true, score_values)
        auc = roc_auc_score(y_true, score_values)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC = {auc:.2f})')
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    
    # Customize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the example."""
    # Determine computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create a synthetic dataset for statistical analysis
    print("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(
        fs=FS,
        duration=DURATION,
        n_trials=N_TRIALS,
        n_subjects=N_SUBJECTS,
        pac_params=PAC_PARAMS,
        with_contamination=True  # Add weak PAC to some control subjects
    )
    
    # Calculate PAC for all subjects
    print("Calculating PAC...")
    pac_results = calculate_pac_for_dataset(dataset, device=device)
    
    # Apply multiple comparison corrections
    print("Applying multiple comparison corrections...")
    corrected_p_values = apply_multiple_comparison_corrections(pac_results['p_values'])
    
    # Create binary significance masks
    print("Creating binary significance masks...")
    significance_masks = create_binary_significance_masks(corrected_p_values)
    
    # Plot ground truth comparison
    print("Plotting ground truth comparison...")
    plot_ground_truth_comparison(dataset, pac_results, significance_masks)
    
    # Evaluate statistical performance
    print("Evaluating statistical performance...")
    performance = evaluate_statistical_performance(dataset, significance_masks, pac_results)
    
    # Plot performance comparison
    print("Plotting performance comparison...")
    plot_performance_comparison(performance)
    
    # Plot ROC comparison
    print("Plotting ROC comparison...")
    plot_roc_comparison(dataset, pac_results)
    
    print("Done! Figures saved in the 'figures' directory:")
    print("  - figures/03_statistical_comparison.png")
    print("  - figures/03_method_performance.png")
    if HAS_SKLEARN:
        print("  - figures/03_roc_comparison.png")


if __name__ == "__main__":
    main()

# EOF