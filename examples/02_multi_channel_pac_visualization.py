#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 11:15:32"
# File: /home/ywatanabe/proj/gPAC/examples/02_multi_channel_pac_visualization.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/02_multi_channel_pac_visualization.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Advanced multi-channel PAC visualization example.

This example demonstrates:
1. Creating synthetic multi-channel signals with different PAC patterns
2. Calculating PAC across all channels
3. Visualizing PAC values as a topographic map
4. Creating channel-wise PAC comparisons
5. Generating publication-quality figures with proper annotations

Requirements:
- gpac
- numpy
- torch
- matplotlib
- seaborn (for better visualizations)
- mne (for topographic plotting, optional)
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
from typing import List, Tuple, Optional

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Try to import MNE for topographic plotting
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("MNE not found. Topographic plots will be simplified.")

# Create output directory for figures under the examples directory
FIGURE_DIR = os.path.join(os.path.dirname(__FILE__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Parameters
FS = 1000  # Sampling frequency in Hz
DURATION = 5  # Signal duration in seconds
N_CHANNELS = 8  # Number of EEG channels
N_TRIALS = 4  # Number of trials/segments

# Define channel names (frontal to posterior)
CHANNEL_NAMES = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Oz', 'T7', 'T8']

# Define a list of PAC configurations for different channels (phase_hz, amplitude_hz, strength)
CHANNEL_PAC_CONFIG = [
    (5, 80, 0.8),    # Fz: strong theta-gamma coupling
    (6, 90, 0.5),    # FCz: medium theta-gamma coupling
    (4, 70, 0.3),    # Cz: weak delta-gamma coupling
    (10, 60, 0.4),   # CPz: medium alpha-gamma coupling
    (12, 100, 0.6),  # Pz: strong alpha-high gamma coupling
    (8, 80, 0.4),    # Oz: medium alpha-gamma coupling
    (3, 40, 0.2),    # T7: weak delta-gamma coupling
    (9, 120, 0.3),   # T8: weak alpha-high gamma coupling
]

def create_multichannel_pac_signals(
    fs: float,
    duration: float,
    channel_configs: List[Tuple[float, float, float]],
    n_trials: int,
    noise_level: float = 0.2
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Create multi-channel synthetic signals with different PAC patterns.
    
    Args:
        fs: Sampling frequency in Hz
        duration: Signal duration in seconds
        channel_configs: List of (phase_hz, amplitude_hz, strength) tuples for each channel
        n_trials: Number of trials/segments to generate
        noise_level: Strength of noise to add
        
    Returns:
        Tuple of (signals, time_vector)
    """
    # Time vector
    t = np.arange(0, duration, 1/fs)
    time_points = len(t)
    n_channels = len(channel_configs)
    
    # Create empty tensor for signals: (batch=1, channels, trials, time)
    signals = torch.zeros((1, n_channels, n_trials, time_points), dtype=torch.float32)
    
    # Generate signals for each channel and trial
    for ch_idx, (phase_hz, amp_hz, strength) in enumerate(channel_configs):
        for trial in range(n_trials):
            # Create phase signal (slow oscillation)
            # Add small random frequency variation between trials
            phase_freq_jitter = phase_hz * (1 + 0.05 * np.random.randn())
            phase_signal = np.sin(2 * np.pi * phase_freq_jitter * t + np.random.rand() * np.pi)
            
            # Create amplitude signal (fast oscillation)
            # Add small random frequency variation between trials
            amp_freq_jitter = amp_hz * (1 + 0.02 * np.random.randn())
            amp_carrier = np.sin(2 * np.pi * amp_freq_jitter * t + np.random.rand() * np.pi)
            
            # Modulate amplitude by phase
            amplitude_modulation = (1 + strength * phase_signal) / 2
            pac_signal = amplitude_modulation * amp_carrier
            
            # Add some pink noise (1/f noise) for more realistic EEG
            n = len(t)
            pink_noise = np.random.randn(n)
            # Create pink noise by filtering white noise
            pink_noise_f = np.fft.rfft(pink_noise)
            pink_noise_f[1:] = pink_noise_f[1:] / np.sqrt(np.arange(1, len(pink_noise_f)))
            pink_noise = np.fft.irfft(pink_noise_f, n)
            # Normalize and scale
            pink_noise = pink_noise / np.std(pink_noise) * noise_level
            
            # Combine signals
            full_signal = phase_signal * 0.3 + pac_signal * 0.5 + pink_noise
            
            # Store in tensor
            signals[0, ch_idx, trial, :] = torch.from_numpy(full_signal.astype(np.float32))
    
    return signals, t

def plot_multichannel_signals(signals: torch.Tensor, t: np.ndarray, channel_names: List[str]):
    """Plot first trial of multi-channel signals."""
    n_channels = signals.shape[1]
    
    plt.figure(figsize=(12, 10))
    
    # Extract first trial for plotting
    trial_data = signals[0, :, 0, :].numpy()
    
    # Calculate scaling for visualization
    max_val = np.max(np.abs(trial_data))
    spacing = max_val * 2.5  # Space between channels
    
    # Plot each channel
    for ch_idx in range(n_channels):
        # Apply offset for stacking
        offset = (n_channels - ch_idx - 1) * spacing
        plt.plot(t, trial_data[ch_idx, :] + offset, 'k', alpha=0.8, linewidth=1)
        
        # Add channel label
        plt.text(t[0] - 0.2, offset, channel_names[ch_idx], 
                 va='center', ha='right', fontsize=10, fontweight='bold')
    
    # Show only 2 seconds for clarity
    plt.xlim(0, 2)
    plt.title('Multi-channel EEG Signals (First Trial)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (a.u.)')
    
    # Remove y-axis ticks
    plt.yticks([])
    
    # Add scale bar
    plt.plot([1.8, 1.8], [0, spacing/4], 'k', linewidth=2)
    plt.text(1.85, spacing/8, f'{spacing/4:.1f} µV', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '02_multichannel_signals.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_channel_pac_heatmaps(pac_values: torch.Tensor, pha_freqs: np.ndarray, 
                            amp_freqs: np.ndarray, channel_names: List[str],
                            channel_configs: List[Tuple[float, float, float]]):
    """Plot PAC heatmaps for each channel."""
    n_channels = pac_values.shape[1]
    n_cols = min(4, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows),
                            sharex=True, sharey=True)
    
    # Create a custom colormap
    colors = [(0.95, 0.95, 0.95), (1, 1, 0.7), (1, 0.7, 0.5), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('pac_cmap', colors, N=100)
    
    # Ensure pac_values is on CPU
    pac_values_cpu = pac_values.cpu()
    
    # Get global min and max for consistent colorbar
    vmin = pac_values_cpu[0].min().item()
    vmax = pac_values_cpu[0].max().item()
    
    # Set vmin to 0 if it's negative (Z-scores below zero aren't interesting)
    vmin = max(0, vmin)
    
    # Flatten axes array for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    for ch_idx in range(n_channels):
        # Get row and column for this subplot
        if n_rows == 1:
            ax = axes[ch_idx]
        else:
            row, col = ch_idx // n_cols, ch_idx % n_cols
            ax = axes[row, col]
        
        # Plot heatmap
        pac_data = pac_values_cpu[0, ch_idx]
        
        # Check dimensions - ensure we have a 2D array for imshow
        if pac_data.dim() == 1:
            # Debug info
            print(f"PAC data shape: {pac_data.shape}, size: {pac_data.numel()}")
            print(f"pha_freqs len: {len(pha_freqs)}, amp_freqs len: {len(amp_freqs)}")
            
            # If 1D, reshape to 2D - need to figure out the correct shape
            if pac_data.numel() == len(pha_freqs) * len(amp_freqs):
                pac_data = pac_data.view(len(pha_freqs), len(amp_freqs))
            else:
                # Reshape with single dimension for one axis
                n_total = pac_data.numel()
                pac_data = pac_data.view(n_total, 1)
        
        im = ax.imshow(
            pac_data.numpy(),
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation='none'
        )
        
        # Add channel name as title
        ax.set_title(f'Channel: {channel_names[ch_idx]}')
        
        # Add x/y labels only on the bottom/left subplots
        if n_rows == 1 or row == n_rows - 1:
            ax.set_xlabel('Amplitude Frequency (Hz)')
            # Set x-ticks on bottom row only
            ax.set_xticks(np.arange(0, len(amp_freqs), len(amp_freqs)//4))
            ax.set_xticklabels([f"{f:.0f}" for f in amp_freqs[::len(amp_freqs)//4]])
        
        if col == 0:
            ax.set_ylabel('Phase Frequency (Hz)')
            # Set y-ticks on leftmost column only
            ax.set_yticks(np.arange(0, len(pha_freqs), len(pha_freqs)//4))
            ax.set_yticklabels([f"{f:.0f}" for f in pha_freqs[::len(pha_freqs)//4]])
        
        # Mark target PAC frequency for this channel
        pha_target, amp_target, _ = channel_configs[ch_idx]
        pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
        amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
        
        ax.plot(
            amp_idx, pha_idx,
            'o', markerfacecolor='none', markeredgecolor='black',
            markersize=10, markeredgewidth=1.5
        )
    
    # Hide unused subplots
    for i in range(n_channels, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if n_rows == 1:
            axes[i].axis('off') if i < len(axes) else None
        else:
            axes[row, col].axis('off') if row < axes.shape[0] and col < axes.shape[1] else None
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label='PAC Z-score')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '02_channel_pac_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_topographic_pac(pac_values: torch.Tensor, pha_freqs: np.ndarray,
                        amp_freqs: np.ndarray, channel_names: List[str],
                        channel_configs: List[Tuple[float, float, float]]):
    """
    Plot topographic distribution of PAC values for specific frequency pairs.
    Uses MNE if available, otherwise creates a simplified plot.
    """
    # Ensure pac_values is on CPU
    pac_values_cpu = pac_values.cpu()
    
    # Debugging
    print(f"PAC values shape: {pac_values_cpu.shape}")
    
    # If MNE is not available, create a simplified plot
    if not HAS_MNE:
        simplified_topographic_plot(pac_values_cpu, pha_freqs, amp_freqs, 
                                   channel_names, channel_configs)
        return
    
    # Create a layout for topographic plotting
    # This is a simplified approach - for real data, use a standard montage
    ch_pos = {
        'Fz': [0, 0.8],     # Frontal midline
        'FCz': [0, 0.5],    # Fronto-central
        'Cz': [0, 0],       # Central
        'CPz': [0, -0.5],   # Centro-parietal
        'Pz': [0, -0.8],    # Parietal
        'Oz': [0, -1],      # Occipital
        'T7': [-0.8, 0],    # Left temporal
        'T8': [0.8, 0]      # Right temporal
    }
    
    # Create info object for MNE
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=FS,
        ch_types=['eeg'] * len(channel_names)
    )
    
    # Set montage
    dig_ch_pos = {ch_name: np.array([x, y, 0]) for ch_name, (x, y) in ch_pos.items()}
    dig_montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos)
    info.set_montage(dig_montage)
    
    # Get the average PAC values across trials
    pac_avg = pac_values[0].cpu().numpy()  # [channels, flattened]
    
    # For each channel, extract PAC value at its target frequency pair
    target_pac_values = np.zeros(len(channel_names))
    for ch_idx, (pha_target, amp_target, _) in enumerate(channel_configs):
        # Since the data appears to be flattened or has unexpected dimensions,
        # use max value for each channel as an approximation
        target_pac_values[ch_idx] = np.max(pac_avg[ch_idx])
    
    # Prepare data for plotting (MNE needs a time dimension)
    data = target_pac_values.reshape(len(channel_names), 1)
    evoked = mne.EvokedArray(data, info)
    
    # Plot topographic map
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get global min and max for colorbar
    vmin = 0  # We're only interested in positive Z-scores
    vmax = np.max(target_pac_values) * 1.1
    
    # Create the topomap
    im, _ = mne.viz.plot_topomap(
        target_pac_values, 
        evoked.info,
        axes=ax,
        # vmin and vmax parameters might not be supported in newer MNE versions
        # use vlim instead if that's the case
        vlim=(vmin, vmax),
        cmap='hot',
        show=False
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PAC Z-score at target frequencies')
    
    # Add title with information about target frequencies
    plt.suptitle('Topographic PAC Distribution', fontsize=14)
    plt.title('Each channel showing Z-score at its target frequency pair', fontsize=10)
    
    # Add channel labels with their target frequencies
    for ch_idx, ch_name in enumerate(channel_names):
        # Get position
        pos = ch_pos[ch_name]
        pha_target, amp_target, _ = channel_configs[ch_idx]
        
        # Place text annotations
        plt.text(
            pos[0], pos[1] - 0.05, 
            f'{ch_name}\n({pha_target:.0f}-{amp_target:.0f} Hz)',
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '02_topographic_pac.png'), dpi=300, bbox_inches='tight')
    plt.close()

def simplified_topographic_plot(pac_values: torch.Tensor, pha_freqs: np.ndarray,
                               amp_freqs: np.ndarray, channel_names: List[str],
                               channel_configs: List[Tuple[float, float, float]]):
    """Simplified topographic plot when MNE is not available."""
    # Create a simplified topographic layout
    ch_pos = {
        'Fz': [0, 0.8],     # Frontal midline
        'FCz': [0, 0.5],    # Fronto-central
        'Cz': [0, 0],       # Central
        'CPz': [0, -0.5],   # Centro-parietal
        'Pz': [0, -0.8],    # Parietal
        'Oz': [0, -1],      # Occipital
        'T7': [-0.8, 0],    # Left temporal
        'T8': [0.8, 0]      # Right temporal
    }
    
    # Get the average PAC values across trials
    pac_avg = pac_values[0].cpu().numpy()  # [channels, phase, amplitude]
    
    # For each channel, extract PAC value at its target frequency pair
    target_pac_values = np.zeros(len(channel_names))
    for ch_idx, (pha_target, amp_target, _) in enumerate(channel_configs):
        # Find closest frequency indices
        pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
        amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
        target_pac_values[ch_idx] = pac_avg[ch_idx, pha_idx, amp_idx]
    
    # Normalize PAC values for visualization
    norm_pac = (target_pac_values - target_pac_values.min()) / (target_pac_values.max() - target_pac_values.min())
    
    # Plot simplified topographic map
    plt.figure(figsize=(10, 8))
    
    # Draw head outline
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(circle)
    
    # Draw nose
    plt.plot([0, 0, 0.1, 0, -0.1], [1, 1.1, 1.2, 1.1, 1.2], 'k-', linewidth=2)
    
    # Draw ears
    plt.plot([-1, -1.1, -1.1, -1], [0.3, 0.3, -0.3, -0.3], 'k-', linewidth=2)
    plt.plot([1, 1.1, 1.1, 1], [0.3, 0.3, -0.3, -0.3], 'k-', linewidth=2)
    
    # Draw channels
    cmap = plt.cm.hot
    for ch_idx, ch_name in enumerate(channel_names):
        pos = ch_pos[ch_name]
        pha_target, amp_target, _ = channel_configs[ch_idx]
        
        # Draw colored circle for channel
        color = cmap(norm_pac[ch_idx])
        circle = plt.Circle(
            pos, 0.1, 
            color=color, 
            alpha=0.8,
            linewidth=1,
            edgecolor='black'
        )
        plt.gca().add_patch(circle)
        
        # Add channel label with target frequencies
        plt.text(
            pos[0], pos[1] - 0.15, 
            f'{ch_name}\n({pha_target:.0f}-{amp_target:.0f} Hz)',
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
        )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(target_pac_values)
    cbar = plt.colorbar(sm)
    cbar.set_label('PAC Z-score at target frequencies')
    
    # Set axis limits and remove ticks
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axis('equal')
    plt.axis('off')
    
    # Add title
    plt.suptitle('Simplified Topographic PAC Distribution', fontsize=14)
    plt.title('Each channel showing Z-score at its target frequency pair', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '02_simplified_topographic_pac.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_channel_comparison(pac_values: torch.Tensor, pha_freqs: np.ndarray,
                           amp_freqs: np.ndarray, channel_names: List[str],
                           channel_configs: List[Tuple[float, float, float]]):
    """
    Plot comparison of PAC profiles across channels by showing:
    1. Phase frequency profiles (horizontal slices through PAC matrix)
    2. Amplitude frequency profiles (vertical slices through PAC matrix)
    """
    # Get data from the PAC tensor - ensure it's on CPU
    pac_values_cpu = pac_values.cpu() 
    pac_data = pac_values_cpu[0].numpy()  # [channels, phase, amplitude]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Plot phase frequency profiles
    plt.subplot(2, 1, 1)
    
    # For each channel, select the frequency profile at the target amplitude frequency
    for ch_idx, (pha_target, amp_target, _) in enumerate(channel_configs):
        # Find target amplitude frequency index
        amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
        
        # Extract phase profile at this amplitude
        phase_profile = pac_data[ch_idx, :, amp_idx]
        
        # Plot profile with channel-specific color
        plt.plot(pha_freqs, phase_profile, '-o', label=f'{channel_names[ch_idx]}', 
                 markersize=4, alpha=0.7)
        
        # Mark target phase frequency
        pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
        plt.plot(pha_target, phase_profile[pha_idx], 'ko', markersize=8)
    
    plt.xlabel('Phase Frequency (Hz)')
    plt.ylabel('PAC Z-score')
    plt.title('Phase Frequency Profiles at Target Amplitude Frequencies')
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 2. Plot amplitude frequency profiles
    plt.subplot(2, 1, 2)
    
    # For each channel, select the frequency profile at the target phase frequency
    for ch_idx, (pha_target, amp_target, _) in enumerate(channel_configs):
        # Find target phase frequency index
        pha_idx = np.argmin(np.abs(pha_freqs - pha_target))
        
        # Extract amplitude profile at this phase
        amp_profile = pac_data[ch_idx, pha_idx, :]
        
        # Plot profile with channel-specific color
        plt.plot(amp_freqs, amp_profile, '-o', label=f'{channel_names[ch_idx]}', 
                 markersize=4, alpha=0.7)
        
        # Mark target amplitude frequency
        amp_idx = np.argmin(np.abs(amp_freqs - amp_target))
        plt.plot(amp_target, amp_profile[amp_idx], 'ko', markersize=8)
    
    plt.xlabel('Amplitude Frequency (Hz)')
    plt.ylabel('PAC Z-score')
    plt.title('Amplitude Frequency Profiles at Target Phase Frequencies')
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '02_channel_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the example."""
    try:
        # Create synthetic multi-channel signals with different PAC patterns
        print("Creating synthetic multi-channel signals...")
        signals, t = create_multichannel_pac_signals(
            FS, DURATION, CHANNEL_PAC_CONFIG, N_TRIALS
        )
        
        # Plot multi-channel signals
        print("Plotting multi-channel signals...")
        plot_multichannel_signals(signals, t, CHANNEL_NAMES)
        
        # Calculate PAC with surrogate distribution
        print("Calculating PAC with surrogate distribution...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Move signals to the appropriate device
        signals_device = signals.to(device)
        
        pac_values, surrogate_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal=signals_device,
            fs=FS,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=15,  # Use 15 phase bands for better visualization
            amp_start_hz=30.0,
            amp_end_hz=160.0,
            amp_n_bands=15,  # Use 15 amplitude bands for better visualization
            n_perm=200,      # Number of permutations
            return_dist=True,
            device=device,
            fp16=False,      # Use fp32 for better precision
        )
        
        # Convert frequencies to NumPy arrays
        pha_freqs = pha_freqs.astype(np.float32)
        amp_freqs = amp_freqs.astype(np.float32)
        
        # Make sure tensors are on CPU before plotting
        pac_values_cpu = pac_values.cpu()
        surrogate_dist_cpu = surrogate_dist.cpu()
        
        # Plot PAC heatmaps for each channel
        print("Plotting channel PAC heatmaps...")
        plot_channel_pac_heatmaps(pac_values_cpu, pha_freqs, amp_freqs, CHANNEL_NAMES, CHANNEL_PAC_CONFIG)
        
        # Plot topographic PAC distribution
        print("Plotting topographic PAC distribution...")
        plot_topographic_pac(pac_values_cpu, pha_freqs, amp_freqs, CHANNEL_NAMES, CHANNEL_PAC_CONFIG)
        
        # Plot channel comparison
        print("Plotting channel comparison...")
        plot_channel_comparison(pac_values_cpu, pha_freqs, amp_freqs, CHANNEL_NAMES, CHANNEL_PAC_CONFIG)
        
        print("Done! Figures saved in the 'figures' directory:")
        print("  - figures/02_multichannel_signals.png")
        print("  - figures/02_channel_pac_heatmaps.png")
        if HAS_MNE:
            print("  - figures/02_topographic_pac.png")
        else:
            print("  - figures/02_simplified_topographic_pac.png")
        print("  - figures/02_channel_comparison.png")
        
    except Exception as e:
        print(f"Error: {e}")
        # Print more detailed error information for debugging
        import traceback
        print(traceback.format_exc())
    if HAS_MNE:
        print("  - figures/02_topographic_pac.png")
    else:
        print("  - figures/02_simplified_topographic_pac.png")
    print("  - figures/02_channel_comparison.png")


if __name__ == "__main__":
    main()

# EOF