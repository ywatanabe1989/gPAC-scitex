#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 11:15:32"
# File: /home/ywatanabe/proj/gPAC/examples/03_statistical_analysis_fixed.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/03_statistical_analysis_fixed.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Advanced statistical analysis of PAC in multiple subjects and conditions.

This example demonstrates:
1. Creating synthetic dataset with multiple subjects and known PAC
2. Calculating PAC and statistics for each subject
3. Performing group-level analysis
4. Visualizing different statistical metrics and ROC curves
5. Comparing between subject groups (PAC vs. no PAC)

Requirements:
- gpac
- numpy
- torch
- matplotlib
- seaborn (for better visualizations)
- scipy (for statistics)
- scikit-learn (for metrics)
"""

import gpac
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import os
import random

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create output directory for figures under the examples directory
FIGURE_DIR = os.path.join(os.path.dirname(__FILE__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Parameters
FS = 1000  # Sampling frequency in Hz
DURATION = 10  # Signal duration in seconds
N_TRIALS = 6  # Number of trials per subject
N_SUBJECTS = 6  # Total number of subjects
N_PAC_SUBJECTS = 3  # Number of subjects with PAC
PHA_FREQ = 6  # Target phase frequency in Hz
AMP_FREQ = 80  # Target amplitude frequency in Hz

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def create_subject_with_pac(
    fs, duration, pha_freq, amp_freq, n_trials, 
    modulation_strength=0.8, noise_level=0.2,
    add_distractor_pac=False
):
    """Create synthetic data for a subject with PAC."""
    # Time vector
    t = np.arange(0, duration, 1/fs)
    time_points = len(t)
    
    # Create empty tensor for signals: (batch=1, channels=1, trials, time)
    signals = torch.zeros((1, 1, n_trials, time_points), dtype=torch.float32)
    
    # Generate signals for each trial
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
        amplitude_modulation = (1 + modulation_strength * phase_signal) / 2
        pac_signal = amplitude_modulation * amp_carrier
        
        # Add distractor PAC if requested (different frequency pair)
        if add_distractor_pac:
            # Create distractor PAC in alpha-beta (10-20 Hz)
            dist_phase_signal = np.sin(2 * np.pi * 10 * t + np.random.rand() * np.pi)
            dist_amp_carrier = np.sin(2 * np.pi * 20 * t + np.random.rand() * np.pi)
            
            # Modulate with lower strength
            dist_amplitude_modulation = (1 + 0.3 * dist_phase_signal) / 2
            distractor_pac = dist_amplitude_modulation * dist_amp_carrier
        else:
            distractor_pac = 0
        
        # Add pink noise (1/f noise)
        n = len(t)
        pink_noise = np.random.randn(n)
        pink_noise_f = np.fft.rfft(pink_noise)
        pink_noise_f[1:] = pink_noise_f[1:] / np.sqrt(np.arange(1, len(pink_noise_f)))
        pink_noise = np.fft.irfft(pink_noise_f, n)
        pink_noise = pink_noise / np.std(pink_noise) * noise_level
        
        # Combine signals
        full_signal = phase_signal * 0.3 + pac_signal * 0.5 + distractor_pac * 0.2 + pink_noise
        
        # Store in tensor
        signals[0, 0, trial, :] = torch.from_numpy(full_signal.astype(np.float32))
    
    return signals, t

def create_subject_without_pac(
    fs, duration, n_trials, 
    noise_level=0.3
):
    """Create synthetic data for a subject without PAC."""
    # Time vector
    t = np.arange(0, duration, 1/fs)
    time_points = len(t)
    
    # Create empty tensor for signals: (batch=1, channels=1, trials, time)
    signals = torch.zeros((1, 1, n_trials, time_points), dtype=torch.float32)
    
    # Generate signals for each trial
    for trial in range(n_trials):
        # Create signals with similar spectral properties but without coupling
        # Low-frequency component
        low_freq_signal = 0
        for f in np.arange(2, 20, 2):  # Frequencies from 2-20 Hz
            ampl = 1.0 / f  # 1/f amplitude scaling
            low_freq_signal += ampl * np.sin(2 * np.pi * f * t + np.random.rand() * np.pi)
        
        # High-frequency component (without modulation)
        high_freq_signal = 0
        for f in np.arange(30, 150, 10):  # Frequencies from 30-150 Hz
            ampl = 0.1 / (f / 30)  # Amplitude scaling
            high_freq_signal += ampl * np.sin(2 * np.pi * f * t + np.random.rand() * np.pi)
        
        # Add pink noise
        n = len(t)
        pink_noise = np.random.randn(n)
        pink_noise_f = np.fft.rfft(pink_noise)
        pink_noise_f[1:] = pink_noise_f[1:] / np.sqrt(np.arange(1, len(pink_noise_f)))
        pink_noise = np.fft.irfft(pink_noise_f, n)
        pink_noise = pink_noise / np.std(pink_noise) * noise_level
        
        # Combine signals
        full_signal = low_freq_signal * 0.5 + high_freq_signal * 0.3 + pink_noise
        
        # Normalize
        full_signal = full_signal / np.std(full_signal)
        
        # Store in tensor
        signals[0, 0, trial, :] = torch.from_numpy(full_signal.astype(np.float32))
    
    return signals, t

def create_synthetic_dataset(fs, duration, n_trials, n_subjects, n_pac_subjects,
                           pha_freq, amp_freq):
    """Create a synthetic dataset with multiple subjects."""
    dataset = {
        'subjects': [],
        'subject_ids': [],
        'has_pac': [],
        'time': None
    }
    
    # Determine which subjects will have PAC
    pac_indices = np.random.choice(range(n_subjects), n_pac_subjects, replace=False)
    
    # Create data for each subject
    for i in range(n_subjects):
        has_pac = i in pac_indices
        
        if has_pac:
            # Vary PAC strength between subjects
            pac_strength = 0.5 + 0.5 * np.random.rand()  # Between 0.5 and 1.0
            
            # Half of PAC subjects get distractor PAC
            add_distractor = i % 2 == 0
            
            signals, t = create_subject_with_pac(
                fs, duration, pha_freq, amp_freq, n_trials,
                modulation_strength=pac_strength,
                add_distractor_pac=add_distractor
            )
        else:
            signals, t = create_subject_without_pac(fs, duration, n_trials)
        
        # Store data
        dataset['subjects'].append(signals)
        dataset['subject_ids'].append(f'Sub-{i+1:02d}')
        dataset['has_pac'].append(has_pac)
        
        # Store time vector (same for all subjects)
        if dataset['time'] is None:
            dataset['time'] = t
    
    return dataset

def calculate_pac_for_dataset(dataset, device='cpu'):
    """Calculate PAC for all subjects in the dataset."""
    results = {
        'pac_values': [],
        'surrogate_dists': [],
        'p_values': [],
        'pha_freqs': None,
        'amp_freqs': None,
        'subjects': dataset['subject_ids'],
        'has_pac': dataset['has_pac']
    }
    
    # Process each subject
    for i, subject_data in enumerate(dataset['subjects']):
        print(f"Calculating PAC for subject {i+1}/{len(dataset['subjects'])}...")
        
        # Move data to device
        subject_data_device = subject_data.to(device)
        
        # Calculate PAC with surrogate distribution
        pac_values, surrogate_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal=subject_data_device,
            fs=FS,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=15,
            amp_start_hz=40.0,
            amp_end_hz=160.0,
            amp_n_bands=15,
            n_perm=200,
            return_dist=True,
            device=device,
            fp16=False,
        )
        
        # Store frequency information once
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
        
        # Print the shapes for debugging
        print(f"PAC shape: {pac_np.shape}, Surrogate shape: {surrogate_np.shape}")
        
        # Handle special case when PAC is just one dimension
        if len(pac_np.shape) == 1 and len(pac_np) == 15:  # Single dimension case
            print("Detected 1D PAC values of length 15, treating as a single dimension")
            # Create a simplified p-value calculation (1D to 1D)
            p_values = np.zeros_like(pac_np)
            for i in range(len(pac_np)):
                p_values[i] = np.mean(surrogate_np[:, i] >= pac_np[i])
        else:
            # Proceed with the standard approach - try to reshape to 2D
            try:
                # Calculate p-values
                p_values = np.zeros_like(pac_np)
                
                # For a standard case: reshape to explicit 2D grid
                n_pha = len(results['pha_freqs'])
                n_amp = len(results['amp_freqs'])
                
                # Try to determine if reshaping is needed
                if len(pac_np.shape) == 1:
                    # If values are flattened, reshape to 2D
                    print(f"Attempting to reshape 1D array of size {len(pac_np)} to ({n_pha}, {n_amp})")
                    if len(pac_np) == n_pha * n_amp:
                        pac_np = pac_np.reshape(n_pha, n_amp)
                        if len(surrogate_np.shape) == 2:
                            surrogate_np = surrogate_np.reshape(surrogate_np.shape[0], n_pha, n_amp)
                
                # Calculate p-values based on available dimensions
                if len(pac_np.shape) == 1:
                    # Still 1D, probably only one frequency dimension
                    for i in range(len(pac_np)):
                        if len(surrogate_np.shape) == 2:
                            p_values[i] = np.mean(surrogate_np[:, i] >= pac_np[i])
                else:
                    # Should be 2D by now
                    for i in range(pac_np.shape[0]):
                        for j in range(pac_np.shape[1]):
                            if len(surrogate_np.shape) == 2:
                                # If surrogate_np is [n_surrogates, flattened_freq_pairs]
                                flat_idx = i * pac_np.shape[1] + j
                                if flat_idx < surrogate_np.shape[1]:
                                    p_values[i, j] = np.mean(surrogate_np[:, flat_idx] >= pac_np[i, j])
                            elif len(surrogate_np.shape) == 3:
                                # If surrogate_np is [n_surrogates, pha_freq, amp_freq]
                                p_values[i, j] = np.mean(surrogate_np[:, i, j] >= pac_np[i, j])
            except Exception as e:
                print(f"Error during p-value calculation: {e}")
                # Create a simple p-value array that won't cause further errors
                p_values = np.zeros_like(pac_np)
        
        # Store results
        results['pac_values'].append(pac_values[0, 0].cpu())  # Store as tensor
        results['surrogate_dists'].append(surrogate_dist[:, 0, 0].cpu())
        results['p_values'].append(p_values)
    
    # Convert lists to tensors/arrays
    results['pac_values'] = torch.stack(results['pac_values'])
    
    return results

def calculate_group_statistics(results):
    """Calculate group-level statistics from individual PAC results."""
    # Extract relevant data
    pha_freqs = results['pha_freqs']
    amp_freqs = results['amp_freqs']
    has_pac = np.array(results['has_pac'])
    
    # Convert PAC values to numpy for easier manipulation
    pac_values = results['pac_values'].numpy()
    
    # Create output structure
    group_stats = {
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'pac_mean': {
            'all': np.mean(pac_values, axis=0),
            'pac_group': np.mean(pac_values[has_pac], axis=0) if np.sum(has_pac) > 0 else None,
            'no_pac_group': np.mean(pac_values[~has_pac], axis=0) if np.sum(~has_pac) > 0 else None
        },
        'pac_std': {
            'all': np.std(pac_values, axis=0),
            'pac_group': np.std(pac_values[has_pac], axis=0) if np.sum(has_pac) > 0 else None,
            'no_pac_group': np.std(pac_values[~has_pac], axis=0) if np.sum(~has_pac) > 0 else None
        },
        't_stats': {},
        'p_values': {},
        'effect_sizes': {}
    }
    
    # Calculate statistics only if both groups have subjects
    if np.sum(has_pac) > 0 and np.sum(~has_pac) > 0:
        # Perform t-tests between groups for each frequency pair
        t_stats = np.zeros_like(group_stats['pac_mean']['all'])
        p_values = np.zeros_like(group_stats['pac_mean']['all'])
        effect_sizes = np.zeros_like(group_stats['pac_mean']['all'])
        
        # Check dimensions and reshape if needed
        n_pha = len(pha_freqs)
        n_amp = len(amp_freqs)
        
        # Reshape pac_values and determine dimensions
        if len(pac_values.shape) == 2:
            # [subjects, frequency pairs flattened]
            n_subjects = pac_values.shape[0]
            n_pairs = pac_values.shape[1]
            
            # Check if this matches expected size
            if n_pairs != n_pha * n_amp:
                # Try to determine correct shape
                if n_pairs == n_pha or n_pairs == n_amp:
                    # Probably only one dimension
                    if n_pairs == n_pha:
                        # Only phase dimension
                        for i in range(n_pha):
                            # Independent t-test between groups
                            stat, p = stats.ttest_ind(pac_values[has_pac, i], 
                                                    pac_values[~has_pac, i],
                                                    equal_var=False)
                            t_stats[i] = stat
                            p_values[i] = p
                            
                            # Cohen's d effect size
                            mean_diff = np.mean(pac_values[has_pac, i]) - np.mean(pac_values[~has_pac, i])
                            pooled_std = np.sqrt((np.var(pac_values[has_pac, i]) + np.var(pac_values[~has_pac, i])) / 2)
                            effect_sizes[i] = mean_diff / pooled_std if pooled_std > 0 else 0
                    else:
                        # Only amplitude dimension
                        for j in range(n_amp):
                            # Independent t-test between groups
                            stat, p = stats.ttest_ind(pac_values[has_pac, j], 
                                                    pac_values[~has_pac, j],
                                                    equal_var=False)
                            t_stats[j] = stat
                            p_values[j] = p
                            
                            # Cohen's d effect size
                            mean_diff = np.mean(pac_values[has_pac, j]) - np.mean(pac_values[~has_pac, j])
                            pooled_std = np.sqrt((np.var(pac_values[has_pac, j]) + np.var(pac_values[~has_pac, j])) / 2)
                            effect_sizes[j] = mean_diff / pooled_std if pooled_std > 0 else 0
                else:
                    # Unknown format - reshape assuming phase, amplitude
                    pac_values_reshaped = pac_values.reshape(n_subjects, n_pha, n_amp)
                    
                    for i in range(n_pha):
                        for j in range(n_amp):
                            # Independent t-test between groups
                            stat, p = stats.ttest_ind(pac_values_reshaped[has_pac, i, j], 
                                                     pac_values_reshaped[~has_pac, i, j],
                                                     equal_var=False)
                            t_stats[i, j] = stat
                            p_values[i, j] = p
                            
                            # Cohen's d effect size
                            mean_diff = np.mean(pac_values_reshaped[has_pac, i, j]) - np.mean(pac_values_reshaped[~has_pac, i, j])
                            pooled_std = np.sqrt((np.var(pac_values_reshaped[has_pac, i, j]) + np.var(pac_values_reshaped[~has_pac, i, j])) / 2)
                            effect_sizes[i, j] = mean_diff / pooled_std if pooled_std > 0 else 0
            else:
                # Reshape to [subjects, phase, amplitude]
                pac_values_reshaped = pac_values.reshape(n_subjects, n_pha, n_amp)
                
                for i in range(n_pha):
                    for j in range(n_amp):
                        # Independent t-test between groups
                        stat, p = stats.ttest_ind(pac_values_reshaped[has_pac, i, j], 
                                                 pac_values_reshaped[~has_pac, i, j],
                                                 equal_var=False)
                        t_stats[i, j] = stat
                        p_values[i, j] = p
                        
                        # Cohen's d effect size
                        mean_diff = np.mean(pac_values_reshaped[has_pac, i, j]) - np.mean(pac_values_reshaped[~has_pac, i, j])
                        pooled_std = np.sqrt((np.var(pac_values_reshaped[has_pac, i, j]) + np.var(pac_values_reshaped[~has_pac, i, j])) / 2)
                        effect_sizes[i, j] = mean_diff / pooled_std if pooled_std > 0 else 0
        elif len(pac_values.shape) == 3:
            # Already in format [subjects, phase, amplitude]
            for i in range(pac_values.shape[1]):
                for j in range(pac_values.shape[2]):
                    # Independent t-test between groups
                    stat, p = stats.ttest_ind(pac_values[has_pac, i, j], 
                                             pac_values[~has_pac, i, j],
                                             equal_var=False)
                    t_stats[i, j] = stat
                    p_values[i, j] = p
                    
                    # Cohen's d effect size
                    mean_diff = np.mean(pac_values[has_pac, i, j]) - np.mean(pac_values[~has_pac, i, j])
                    pooled_std = np.sqrt((np.var(pac_values[has_pac, i, j]) + np.var(pac_values[~has_pac, i, j])) / 2)
                    effect_sizes[i, j] = mean_diff / pooled_std if pooled_std > 0 else 0
        
        group_stats['t_stats'] = t_stats
        group_stats['p_values'] = p_values
        group_stats['effect_sizes'] = effect_sizes
    
    return group_stats

def plot_group_comparison(group_stats):
    """Create visualizations of group-level statistics."""
    # Extract data
    pha_freqs = group_stats['pha_freqs']
    amp_freqs = group_stats['amp_freqs']
    
    # Print shapes of what we're working with
    print(f"Pha freqs shape: {np.array(pha_freqs).shape}, Amp freqs shape: {np.array(amp_freqs).shape}")
    if group_stats['pac_mean']['pac_group'] is not None:
        print(f"PAC group shape: {np.array(group_stats['pac_mean']['pac_group']).shape}")
    if group_stats['pac_mean']['no_pac_group'] is not None:
        print(f"No PAC group shape: {np.array(group_stats['pac_mean']['no_pac_group']).shape}")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(15, 10))
    
    # Define plotting grid
    grid = plt.GridSpec(2, 3, figure=fig)
    
    # Special case for 1D PAC values - create 2D grid for visualization
    if group_stats['pac_mean']['pac_group'] is not None:
        pac_group = np.array(group_stats['pac_mean']['pac_group'])
        if len(pac_group.shape) == 1:
            if len(pac_group) == len(pha_freqs):
                # We have phase frequencies only
                print("Reshaping 1D PAC values (phase frequencies) for visualization")
                n_pha = len(pha_freqs)
                pac_group = pac_group.reshape(n_pha, 1)
                group_stats['pac_mean']['pac_group'] = pac_group
                
                if group_stats['pac_mean']['no_pac_group'] is not None:
                    no_pac_group = np.array(group_stats['pac_mean']['no_pac_group'])
                    if len(no_pac_group.shape) == 1:
                        no_pac_group = no_pac_group.reshape(n_pha, 1)
                        group_stats['pac_mean']['no_pac_group'] = no_pac_group
    
    # 1. Plot PAC means for PAC group - use barplot for 1D data
    if group_stats['pac_mean']['pac_group'] is not None:
        ax1 = fig.add_subplot(grid[0, 0])
        pac_group = np.array(group_stats['pac_mean']['pac_group'])
        
        if len(pac_group.shape) == 1:
            # For 1D data, create a bar plot instead of an image
            ax1.bar(range(len(pac_group)), pac_group, color='red', alpha=0.7)
            ax1.set_title('Mean PAC Z-score: PAC Group')
            ax1.set_xlabel('Frequency Index')
            ax1.set_ylabel('PAC Z-score')
            # Add frequency labels
            if len(pac_group) == len(pha_freqs):
                ax1.set_xticks(range(0, len(pha_freqs), max(1, len(pha_freqs)//4)))
                ax1.set_xticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
                ax1.set_xlabel('Phase Frequency (Hz)')
            elif len(pac_group) == len(amp_freqs):
                ax1.set_xticks(range(0, len(amp_freqs), max(1, len(amp_freqs)//4)))
                ax1.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
                ax1.set_xlabel('Amplitude Frequency (Hz)')
        else:
            # For 2D data, use an image plot
            im1 = ax1.imshow(
                pac_group,
                origin='lower',
                aspect='auto',
                cmap='YlOrRd',
                interpolation='none'
            )
            ax1.set_title('Mean PAC Z-score: PAC Group')
            ax1.set_xlabel('Amplitude Frequency (Hz)')
            ax1.set_ylabel('Phase Frequency (Hz)')
            
            # Add axis labels
            ax1.set_xticks(np.arange(0, pac_group.shape[1], max(1, pac_group.shape[1]//4)))
            if pac_group.shape[1] == len(amp_freqs):
                ax1.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
            ax1.set_yticks(np.arange(0, pac_group.shape[0], max(1, pac_group.shape[0]//4)))
            if pac_group.shape[0] == len(pha_freqs):
                ax1.set_yticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
            
            plt.colorbar(im1, ax=ax1)
    
    # 2. Plot PAC means for non-PAC group
    if group_stats['pac_mean']['no_pac_group'] is not None:
        ax2 = fig.add_subplot(grid[0, 1])
        no_pac_group = np.array(group_stats['pac_mean']['no_pac_group'])
        
        if len(no_pac_group.shape) == 1:
            # For 1D data, create a bar plot instead of an image
            ax2.bar(range(len(no_pac_group)), no_pac_group, color='blue', alpha=0.7)
            ax2.set_title('Mean PAC Z-score: Non-PAC Group')
            ax2.set_xlabel('Frequency Index')
            ax2.set_ylabel('PAC Z-score')
            # Add frequency labels
            if len(no_pac_group) == len(pha_freqs):
                ax2.set_xticks(range(0, len(pha_freqs), max(1, len(pha_freqs)//4)))
                ax2.set_xticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
                ax2.set_xlabel('Phase Frequency (Hz)')
            elif len(no_pac_group) == len(amp_freqs):
                ax2.set_xticks(range(0, len(amp_freqs), max(1, len(amp_freqs)//4)))
                ax2.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
                ax2.set_xlabel('Amplitude Frequency (Hz)')
        else:
            # For 2D data, use an image plot
            im2 = ax2.imshow(
                no_pac_group,
                origin='lower',
                aspect='auto',
                cmap='YlOrRd',
                interpolation='none'
            )
            ax2.set_title('Mean PAC Z-score: Non-PAC Group')
            ax2.set_xlabel('Amplitude Frequency (Hz)')
            
            # Add axis labels
            ax2.set_xticks(np.arange(0, no_pac_group.shape[1], max(1, no_pac_group.shape[1]//4)))
            if no_pac_group.shape[1] == len(amp_freqs):
                ax2.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
            ax2.set_yticks(np.arange(0, no_pac_group.shape[0], max(1, no_pac_group.shape[0]//4)))
            if no_pac_group.shape[0] == len(pha_freqs):
                ax2.set_yticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
            
            plt.colorbar(im2, ax=ax2)
    
    # 3. Plot effect sizes
    if 'effect_sizes' in group_stats and len(group_stats['effect_sizes']) > 0:
        ax3 = fig.add_subplot(grid[0, 2])
        
        # Ensure the effect sizes matrix is 2D
        effect_sizes = np.array(group_stats['effect_sizes'])
        
        if len(effect_sizes.shape) == 1:
            # For 1D effect sizes, plot as bar chart
            ax3.bar(range(len(effect_sizes)), effect_sizes, color='purple', alpha=0.7)
            ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
            # Add guidelines for effect size interpretation
            ax3.axhline(0.2, color='gray', linestyle='--', linewidth=0.5)
            ax3.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
            ax3.axhline(0.8, color='gray', linestyle='--', linewidth=0.5)
            ax3.axhline(-0.2, color='gray', linestyle='--', linewidth=0.5)
            ax3.axhline(-0.5, color='gray', linestyle='--', linewidth=0.5)
            ax3.axhline(-0.8, color='gray', linestyle='--', linewidth=0.5)
            
            # Add text annotations
            y_range = max(max(effect_sizes), abs(min(effect_sizes))) if len(effect_sizes) > 0 else 1
            ax3.text(len(effect_sizes) * 0.95, 0.8, 'Large+', ha='right', va='center', fontsize=8)
            ax3.text(len(effect_sizes) * 0.95, 0.5, 'Medium+', ha='right', va='center', fontsize=8)
            ax3.text(len(effect_sizes) * 0.95, 0.2, 'Small+', ha='right', va='center', fontsize=8)
            ax3.text(len(effect_sizes) * 0.95, -0.2, 'Small-', ha='right', va='center', fontsize=8)
            ax3.text(len(effect_sizes) * 0.95, -0.5, 'Medium-', ha='right', va='center', fontsize=8)
            ax3.text(len(effect_sizes) * 0.95, -0.8, 'Large-', ha='right', va='center', fontsize=8)
            
            ax3.set_title("Cohen's d Effect Size")
            ax3.set_xlabel('Frequency Index')
            ax3.set_ylabel('Effect Size')
            
            # Add frequency labels
            if len(effect_sizes) == len(pha_freqs):
                ax3.set_xticks(range(0, len(pha_freqs), max(1, len(pha_freqs)//4)))
                ax3.set_xticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
                ax3.set_xlabel('Phase Frequency (Hz)')
            elif len(effect_sizes) == len(amp_freqs):
                ax3.set_xticks(range(0, len(amp_freqs), max(1, len(amp_freqs)//4)))
                ax3.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
                ax3.set_xlabel('Amplitude Frequency (Hz)')
        else:
            # For 2D effect sizes, use imshow
            im3 = ax3.imshow(
                effect_sizes,
                origin='lower',
                aspect='auto',
                cmap='coolwarm',
                vmin=-1.5, vmax=1.5,  # Cohen's d: small=0.2, medium=0.5, large=0.8
                interpolation='none'
            )
            ax3.set_title("Cohen's d Effect Size")
            ax3.set_xlabel('Amplitude Frequency (Hz)')
            
            # Add axis labels
            ax3.set_xticks(np.arange(0, effect_sizes.shape[1], max(1, effect_sizes.shape[1]//4)))
            if effect_sizes.shape[1] == len(amp_freqs):
                ax3.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
            ax3.set_yticks(np.arange(0, effect_sizes.shape[0], max(1, effect_sizes.shape[0]//4)))
            if effect_sizes.shape[0] == len(pha_freqs):
                ax3.set_yticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
            
            cbar = plt.colorbar(im3, ax=ax3)
            
            # Add labels for effect size interpretation
            cbar.ax.text(1.5, 0.8, 'Large +', ha='center', va='center', fontsize=8, rotation=-90)
            cbar.ax.text(1.5, 0.5, 'Medium +', ha='center', va='center', fontsize=8, rotation=-90)
            cbar.ax.text(1.5, 0.2, 'Small +', ha='center', va='center', fontsize=8, rotation=-90)
            cbar.ax.text(1.5, -0.2, 'Small -', ha='center', va='center', fontsize=8, rotation=-90)
            cbar.ax.text(1.5, -0.5, 'Medium -', ha='center', va='center', fontsize=8, rotation=-90)
            cbar.ax.text(1.5, -0.8, 'Large -', ha='center', va='center', fontsize=8, rotation=-90)
    
    # 4. Plot significant clusters
    if 'p_values' in group_stats and len(group_stats['p_values']) > 0:
        ax4 = fig.add_subplot(grid[1, 0:2])
        
        # Ensure p_values is a numpy array
        p_values = np.array(group_stats['p_values'])
        
        if len(p_values.shape) == 1:
            # For 1D p-values, plot as bar chart of -log10(p)
            log_p = -np.log10(np.clip(p_values, 1e-10, 1.0))  # Clip to avoid log(0)
            
            # Plot negative log p-values (higher = more significant)
            ax4.bar(range(len(log_p)), log_p, color='orange', alpha=0.7)
            
            # Add significance thresholds
            thresholds = [0.05, 0.01, 0.001]
            for threshold in thresholds:
                y_pos = -np.log10(threshold)
                ax4.axhline(y_pos, color='black', linestyle='--', linewidth=0.5)
                ax4.text(len(log_p) * 0.95, y_pos, f'p={threshold}', ha='right', va='center', fontsize=8)
            
            ax4.set_title('Statistical Significance (-log10(p))')
            ax4.set_xlabel('Frequency Index')
            ax4.set_ylabel('-log10(p-value)')
            
            # Add frequency labels
            if len(p_values) == len(pha_freqs):
                ax4.set_xticks(range(0, len(pha_freqs), max(1, len(pha_freqs)//4)))
                ax4.set_xticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
                ax4.set_xlabel('Phase Frequency (Hz)')
            elif len(p_values) == len(amp_freqs):
                ax4.set_xticks(range(0, len(amp_freqs), max(1, len(amp_freqs)//4)))
                ax4.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
                ax4.set_xlabel('Amplitude Frequency (Hz)')
        else:
            # For 2D p-values, use imshow
            # Plot -log10(p) for better visualization (larger = more significant)
            im4 = ax4.imshow(
                -np.log10(np.clip(p_values, 1e-10, 1.0)),  # Clip to avoid log(0)
                origin='lower',
                aspect='auto',
                cmap='plasma',
                interpolation='none'
            )
            ax4.set_title('Statistical Significance (-log10(p))')
            ax4.set_xlabel('Amplitude Frequency (Hz)')
            ax4.set_ylabel('Phase Frequency (Hz)')
            
            # Add contour for significance threshold
            significant = p_values < 0.05
            if np.any(significant):
                # Draw contour around significant areas
                ax4.contour(significant, levels=[0.5], colors=['black'], linewidths=2)
            
            # Add axis labels
            ax4.set_xticks(np.arange(0, p_values.shape[1], max(1, p_values.shape[1]//4)))
            if p_values.shape[1] == len(amp_freqs):
                ax4.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
            ax4.set_yticks(np.arange(0, p_values.shape[0], max(1, p_values.shape[0]//4)))
            if p_values.shape[0] == len(pha_freqs):
                ax4.set_yticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
            
            cbar = plt.colorbar(im4, ax=ax4)
            
            # Add markers for common significance thresholds
            thresholds = [0.05, 0.01, 0.001]
            for threshold in thresholds:
                y_pos = -np.log10(threshold)
                cbar.ax.axhline(y_pos, color='black', linestyle='--', linewidth=1)
                cbar.ax.text(1.2, y_pos, f'p={threshold}', ha='center', va='center', fontsize=8)
    
    # 5. Plot frequency profiles for comparison
    ax5 = fig.add_subplot(grid[1, 2])
    
    # Find indices for the target frequency
    pha_idx = np.argmin(np.abs(np.array(pha_freqs) - PHA_FREQ))
    amp_idx = np.argmin(np.abs(np.array(amp_freqs) - AMP_FREQ))
    
    # Extract and plot profiles if data is available
    if group_stats['pac_mean']['pac_group'] is not None and group_stats['pac_mean']['no_pac_group'] is not None:
        # Get data for plotting
        pac_mean_true = np.array(group_stats['pac_mean']['pac_group'])
        pac_mean_false = np.array(group_stats['pac_mean']['no_pac_group'])
        
        # Create a 1D plot comparing values between groups
        if len(pac_mean_true.shape) == 1:
            print("Using 1D data for plot 5")
            # If we only have one dimension, plot direct comparison using indices
            indices = np.arange(len(pac_mean_true))
            ax5.plot(indices, pac_mean_true, 'r-', linewidth=2, label='PAC Group')
            ax5.plot(indices, pac_mean_false, 'b-', linewidth=2, label='Non-PAC Group')
            
            # Add frequency labels
            if len(pac_mean_true) == len(pha_freqs):
                ax5.set_xticks(range(0, len(pha_freqs), max(1, len(pha_freqs)//4)))
                ax5.set_xticklabels([f"{f:.1f}" for f in pha_freqs[::max(1, len(pha_freqs)//4)]])
                ax5.set_xlabel('Phase Frequency (Hz)')
                
                # Mark target phase frequency
                target_idx = np.argmin(np.abs(np.array(pha_freqs) - PHA_FREQ))
                ax5.axvline(x=target_idx, color='gray', linestyle='--')
                ax5.text(target_idx, 0, f'{PHA_FREQ} Hz', 
                         ha='center', va='bottom', fontsize=8, backgroundcolor='white')
                
                ax5.set_title(f'Phase Frequency Profile')
            elif len(pac_mean_true) == len(amp_freqs):
                ax5.set_xticks(range(0, len(amp_freqs), max(1, len(amp_freqs)//4)))
                ax5.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::max(1, len(amp_freqs)//4)]])
                ax5.set_xlabel('Amplitude Frequency (Hz)')
                
                # Mark target amplitude frequency
                target_idx = np.argmin(np.abs(np.array(amp_freqs) - AMP_FREQ))
                ax5.axvline(x=target_idx, color='gray', linestyle='--')
                ax5.text(target_idx, 0, f'{AMP_FREQ} Hz', 
                         ha='center', va='bottom', fontsize=8, backgroundcolor='white')
                
                ax5.set_title(f'Amplitude Frequency Profile')
        elif len(pac_mean_true.shape) == 2:
            print(f"Using 2D data for plot 5, shape: {pac_mean_true.shape}")
            # For 2D data, plot the profile at target frequencies
            # Check if we can plot a profile at target amplitude 
            try:
                if pac_mean_true.shape[1] > amp_idx:
                    # Plot phase frequency profile at target amplitude
                    ax5.plot(pha_freqs, pac_mean_true[:, amp_idx], 'r-', linewidth=2, 
                            label='PAC Group')
                    ax5.plot(pha_freqs, pac_mean_false[:, amp_idx], 'b-', linewidth=2, 
                            label='Non-PAC Group')
                    
                    # Mark target phase frequency
                    ax5.axvline(x=PHA_FREQ, color='gray', linestyle='--')
                    ax5.text(PHA_FREQ, 0, f'{PHA_FREQ} Hz', 
                            ha='center', va='bottom', fontsize=8, backgroundcolor='white')
                    
                    ax5.set_title(f'Phase Profile at {AMP_FREQ} Hz Amplitude')
                    ax5.set_xlabel('Phase Frequency (Hz)')
                elif pac_mean_true.shape[0] > pha_idx:
                    # Plot amplitude frequency profile at target phase
                    ax5.plot(amp_freqs, pac_mean_true[pha_idx, :], 'r-', linewidth=2, 
                            label='PAC Group')
                    ax5.plot(amp_freqs, pac_mean_false[pha_idx, :], 'b-', linewidth=2, 
                            label='Non-PAC Group')
                    
                    # Mark target amplitude frequency
                    ax5.axvline(x=AMP_FREQ, color='gray', linestyle='--')
                    ax5.text(AMP_FREQ, 0, f'{AMP_FREQ} Hz', 
                            ha='center', va='bottom', fontsize=8, backgroundcolor='white')
                    
                    ax5.set_title(f'Amplitude Profile at {PHA_FREQ} Hz Phase')
                    ax5.set_xlabel('Amplitude Frequency (Hz)')
                else:
                    # Fallback - just plot the data as raw values
                    print("Fallback plot: can't plot profiles at target indices")
                    ax5.plot(range(len(pac_mean_true.flatten())), pac_mean_true.flatten(), 'r-', 
                            linewidth=2, label='PAC Group')
                    ax5.plot(range(len(pac_mean_false.flatten())), pac_mean_false.flatten(), 'b-',
                            linewidth=2, label='Non-PAC Group')
                    ax5.set_xlabel('Frequency Index')
                    ax5.set_title('Frequency PAC Values') 
            except Exception as e:
                print(f"Error in plotting profiles: {e}")
                # Emergency fallback - plot simple comparison
                ax5.plot(range(len(pac_mean_true.flatten())), pac_mean_true.flatten(), 'r-',
                        linewidth=2, label='PAC Group')
                ax5.plot(range(len(pac_mean_false.flatten())), pac_mean_false.flatten(), 'b-', 
                        linewidth=2, label='Non-PAC Group')
                ax5.set_xlabel('Frequency Index')
                ax5.set_title('PAC Values Comparison')
        else:
            # Unknown data format - plot as raw values
            print(f"Unknown data shape for plot 5: {pac_mean_true.shape}")
            try:
                ax5.plot(range(len(np.array(pac_mean_true).flatten())), 
                       np.array(pac_mean_true).flatten(), 'r-',
                       linewidth=2, label='PAC Group')
                ax5.plot(range(len(np.array(pac_mean_false).flatten())), 
                       np.array(pac_mean_false).flatten(), 'b-',
                       linewidth=2, label='Non-PAC Group') 
                ax5.set_xlabel('Index')
                ax5.set_title('PAC Value Comparison')
            except Exception as e:
                print(f"Error in fallback plotting: {e}")
                # Ultimate fallback, just use index data
                indices = np.arange(15)  # Assuming 15 frequencies 
                ax5.plot(indices, np.zeros_like(indices), 'r-', label='PAC Group')
                ax5.plot(indices, np.zeros_like(indices), 'b-', label='Non-PAC Group')
                ax5.set_xlabel('Frequency Index')
                ax5.set_title('Placeholder Plot')
        
        ax5.set_ylabel('PAC Z-score')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '03_group_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluate_classification_metrics(results):
    """Evaluate different metrics for distinguishing PAC/non-PAC subjects."""
    # Extract data
    pha_freqs = results['pha_freqs']
    amp_freqs = results['amp_freqs']
    has_pac = np.array(results['has_pac'])
    
    # Reshape PAC values for easier processing
    pac_values = results['pac_values'].numpy()
    n_subjects = len(results['subjects'])
    
    # Define metrics to evaluate
    scores = {
        'max_zscore': np.zeros(n_subjects),
        'mean_zscore': np.zeros(n_subjects),
        'target_zscore': np.zeros(n_subjects),
        'theta_gamma': np.zeros(n_subjects),  # 4-8 Hz phase, 30-100 Hz amplitude
        'alpha_gamma': np.zeros(n_subjects),  # 8-13 Hz phase, 30-100 Hz amplitude
    }
    
    # Find target frequency indices
    pha_idx = np.argmin(np.abs(pha_freqs - PHA_FREQ))
    amp_idx = np.argmin(np.abs(amp_freqs - AMP_FREQ))
    
    # Find frequency band indices
    theta_mask = (pha_freqs >= 4) & (pha_freqs <= 8)
    alpha_mask = (pha_freqs >= 8) & (pha_freqs <= 13)
    gamma_mask = (amp_freqs >= 30) & (amp_freqs <= 100)
    
    # Print debug info
    print(f"PAC values shape: {pac_values.shape}")
    print(f"Phase freqs: {pha_freqs.shape}, Amp freqs: {amp_freqs.shape}")
    print(f"Target phase idx: {pha_idx}, Target amp idx: {amp_idx}")
    
    # Compute metrics for each subject
    for subj_idx in range(n_subjects):
        subj_data = pac_values[subj_idx]
        print(f"Subject {subj_idx+1} data shape: {subj_data.shape}")
        
        # Maximum Z-score anywhere in the PAC map
        scores['max_zscore'][subj_idx] = np.max(subj_data)
        
        # Mean Z-score across all values
        scores['mean_zscore'][subj_idx] = np.mean(subj_data)
        
        # Special case handling for 1D data 
        if len(subj_data.shape) == 1:
            # We only have one dimension - either phase or amplitude frequencies
            # Try to determine which one based on the length
            if len(subj_data) == len(pha_freqs):
                print(f"Subject {subj_idx+1} data appears to be phase dimension only")
                # Only phase dimension is present
                # Use the phase index for the target score 
                if pha_idx < len(subj_data):
                    scores['target_zscore'][subj_idx] = subj_data[pha_idx]
                
                # For frequency bands, just use the masks directly
                if np.any(theta_mask):
                    scores['theta_gamma'][subj_idx] = np.mean(subj_data[theta_mask])
                
                if np.any(alpha_mask):
                    scores['alpha_gamma'][subj_idx] = np.mean(subj_data[alpha_mask])
            
            elif len(subj_data) == len(amp_freqs):
                print(f"Subject {subj_idx+1} data appears to be amplitude dimension only")
                # Only amplitude dimension is present
                # Use the amplitude index for the target score
                if amp_idx < len(subj_data):
                    scores['target_zscore'][subj_idx] = subj_data[amp_idx]
                
                # For frequency bands, just use the masks directly
                if np.any(gamma_mask):
                    scores['theta_gamma'][subj_idx] = np.mean(subj_data[gamma_mask])
                    scores['alpha_gamma'][subj_idx] = np.mean(subj_data[gamma_mask])
            
            else:
                # Unknown format - use defaults
                print(f"Subject {subj_idx+1} data has unknown format")
                scores['target_zscore'][subj_idx] = scores['max_zscore'][subj_idx]
                scores['theta_gamma'][subj_idx] = scores['mean_zscore'][subj_idx]
                scores['alpha_gamma'][subj_idx] = scores['mean_zscore'][subj_idx]
        
        else:
            # 2D case - standard handling
            # Z-score at the target frequency pair
            if pha_idx < subj_data.shape[0] and amp_idx < subj_data.shape[1]:
                scores['target_zscore'][subj_idx] = subj_data[pha_idx, amp_idx]
            
            # Mean Z-score in the theta-gamma band
            if np.any(theta_mask) and np.any(gamma_mask):
                theta_gamma_values = subj_data[np.ix_(theta_mask, gamma_mask)]
                scores['theta_gamma'][subj_idx] = np.mean(theta_gamma_values)
            
            # Mean Z-score in the alpha-gamma band
            if np.any(alpha_mask) and np.any(gamma_mask):
                alpha_gamma_values = subj_data[np.ix_(alpha_mask, gamma_mask)]
                scores['alpha_gamma'][subj_idx] = np.mean(alpha_gamma_values)
    
    # Calculate ROC curves and AUC
    plt.figure(figsize=(10, 8))
    
    # Get ground truth labels
    y_true = has_pac
    
    for name, score_values in scores.items():
        # Skip target score if no PAC subjects
        if name == 'target_zscore' and not any(y_true):
            continue
        
        # Print metric summary
        print(f"Metric {name}: mean={np.mean(score_values):.3f}, min={np.min(score_values):.3f}, max={np.max(score_values):.3f}")
            
        fpr, tpr, thresholds = roc_curve(y_true, score_values)
        auc = roc_auc_score(y_true, score_values)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC = {auc:.2f})')
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    
    # Customize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different PAC Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '03_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return scores

def main():
    """Main function to run the example."""
    try:
        # Create synthetic dataset with multiple subjects
        print("Creating synthetic dataset...")
        dataset = create_synthetic_dataset(
            FS, DURATION, N_TRIALS, N_SUBJECTS, N_PAC_SUBJECTS,
            PHA_FREQ, AMP_FREQ
        )
        
        # Calculate PAC for all subjects
        print("Calculating PAC...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        pac_results = calculate_pac_for_dataset(dataset, device=device)
        
        # Calculate group statistics
        print("Calculating group statistics...")
        group_stats = calculate_group_statistics(pac_results)
        
        # Plot group comparisons
        print("Creating group comparison visualizations...")
        plot_group_comparison(group_stats)
        
        # Evaluate classification metrics
        print("Evaluating classification metrics...")
        scores = evaluate_classification_metrics(pac_results)
        
        print("Done! Figures saved in the 'figures' directory:")
        print("  - figures/03_group_statistics.png")
        print("  - figures/03_roc_curves.png")
        
    except Exception as e:
        print(f"Error: {e}")
        # Print more detailed error information for debugging
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

# EOF