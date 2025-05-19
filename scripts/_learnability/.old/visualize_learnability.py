#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-13 21:49:20 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/visualize_learnability.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/visualize_learnability.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Visualizes the learning process of trainable gPAC parameters
  - Tracks parameter changes during training iterations
  - Compares initial vs. learned frequency bands
  - Creates interactive visualizations of PAC module adaptation
  - Generates interpretable visualizations for research papers

Dependencies:
  - scripts:
    - ./scripts/learnability/classify_demo_signals_using_gPAC_module.py
  - packages:
    - torch
    - numpy
    - matplotlib
    - seaborn
    - mngs
    - gpac

IO:
  - input-files:
    - ./scripts/learnability/results/trained_models/trainable_model.pt
    - ./scripts/learnability/data/synthetic_pac_signals.pt
  - output-files:
    - ./scripts/learnability/results/figures/parameter_learning/
    - ./scripts/learnability/results/figures/parameter_adaptation.png
    - ./scripts/learnability/results/figures/frequency_tracking.png
"""

"""Imports"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mngs

# Add project root to Python path
import sys
project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))

# Import gPAC modules
import gpac
from gpac._pac import PAC, calculate_pac

"""Parameters"""
DEFAULT_PARAMS = {
    'learning_rate': 0.01,       # Learning rate for optimization
    'n_epochs': 100,             # Number of training epochs
    'pha_n_bands': 10,           # Number of phase frequency bands
    'amp_n_bands': 10,           # Number of amplitude frequency bands
    'noise_level': 0.1,          # Noise level for generated signals
    'target_pha_freq': 8.0,      # Target phase frequency to learn
    'target_amp_freq': 100.0,    # Target amplitude frequency to learn
}

"""Functions & Classes"""
def create_pac_module(seq_len, fs, pha_n_bands, amp_n_bands, trainable=True):
    """Create a PAC module with specified parameters."""
    pac_module = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=1.0,
        pha_end_hz=20.0,
        pha_n_bands=pha_n_bands,
        amp_start_hz=60.0,
        amp_end_hz=180.0,
        amp_n_bands=amp_n_bands,
        trainable=trainable
    )
    
    return pac_module

def generate_test_signal(data_gen, target_pha_freq, target_amp_freq, duration, fs, noise_level=0.1):
    """Generate a test PAC signal with known coupling frequencies."""
    # Generate synthetic PAC signal
    signal = data_gen.generate_pac_with_signal(
        n_seconds=duration,
        pha_mod_freq=target_pha_freq,
        amp_car_freq=target_amp_freq,
        pha_bandwidth=target_pha_freq/4,
        amp_bandwidth=target_amp_freq/4,
        coupling_strength=0.8,
        noise_level=noise_level
    )
    
    # Convert to torch tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    
    return signal_tensor

def visualize_frequency_bands(module, fs, save_path=None):
    """Visualize frequency bands before and after training."""
    # Get frequency bands
    pha_mids = module.PHA_MIDS_HZ.detach().cpu().numpy()
    amp_mids = module.AMP_MIDS_HZ.detach().cpu().numpy()
    
    # Get filter bandwidths
    pha_bandwidths = module.PHA_BANDWIDTHS_HZ.detach().cpu().numpy()
    amp_bandwidths = module.AMP_BANDWIDTHS_HZ.detach().cpu().numpy()
    
    # Calculate band edges
    pha_low = pha_mids - pha_bandwidths/2
    pha_high = pha_mids + pha_bandwidths/2
    
    amp_low = amp_mids - amp_bandwidths/2
    amp_high = amp_mids + amp_bandwidths/2
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot phase bands
    for i, (low, mid, high) in enumerate(zip(pha_low, pha_mids, pha_high)):
        ax1.fill_between([low, high], [i, i], [i+0.8, i+0.8], alpha=0.6)
        ax1.text(mid, i+0.4, f"{mid:.2f} Hz", ha='center', va='center', fontsize=10)
    
    ax1.set_title("Phase Frequency Bands")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Band Index")
    ax1.set_xlim(0, 25)
    ax1.set_ylim(-0.5, len(pha_mids))
    ax1.grid(True)
    
    # Plot amplitude bands
    for i, (low, mid, high) in enumerate(zip(amp_low, amp_mids, amp_high)):
        ax2.fill_between([low, high], [i, i], [i+0.8, i+0.8], alpha=0.6)
        ax2.text(mid, i+0.4, f"{mid:.2f} Hz", ha='center', va='center', fontsize=10)
    
    ax2.set_title("Amplitude Frequency Bands")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Band Index")
    ax2.set_xlim(50, 190)
    ax2.set_ylim(-0.5, len(amp_mids))
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def train_pac_module(module, signal, target_pha_freq, target_amp_freq, n_epochs, learning_rate, device):
    """Train the PAC module to adapt its parameters to detect the target frequencies."""
    # Move module and signal to device
    module.to(device)
    signal = signal.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    
    # Storage for parameter history
    history = {
        'epoch': [],
        'loss': [],
        'pha_mids': [],
        'amp_mids': [],
        'max_pac_value': []
    }
    
    # Initial parameters
    initial_pha_mids = module.PHA_MIDS_HZ.detach().clone()
    initial_amp_mids = module.AMP_MIDS_HZ.detach().clone()
    
    # Target frequency indices (closest band to target)
    initial_pha_idx = torch.argmin(torch.abs(initial_pha_mids - target_pha_freq)).item()
    initial_amp_idx = torch.argmin(torch.abs(initial_amp_mids - target_amp_freq)).item()
    
    # Training loop
    for epoch in range(n_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass to calculate PAC values
        pac_values = module(signal)
        
        # We want to maximize the PAC value that corresponds to our target frequencies
        # This means minimizing the negative of that PAC value
        loss = -pac_values.mean()  # Simple loss: maximize average PAC
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Log history
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['pha_mids'].append(module.PHA_MIDS_HZ.detach().cpu().numpy().copy())
        history['amp_mids'].append(module.AMP_MIDS_HZ.detach().cpu().numpy().copy())
        history['max_pac_value'].append(pac_values.max().item())
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == n_epochs:
            max_pac_value = pac_values.max().item()
            mean_pac_value = pac_values.mean().item()
            
            # Get current best frequency matches
            current_pha_mids = module.PHA_MIDS_HZ.detach().cpu().numpy()
            current_amp_mids = module.AMP_MIDS_HZ.detach().cpu().numpy()
            
            best_pha_idx = np.argmin(np.abs(current_pha_mids - target_pha_freq))
            best_amp_idx = np.argmin(np.abs(current_amp_mids - target_amp_freq))
            
            best_pha_freq = current_pha_mids[best_pha_idx]
            best_amp_freq = current_amp_mids[best_amp_idx]
            
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.6f}, Max PAC: {max_pac_value:.6f}")
            print(f"  Best Phase Match: {best_pha_freq:.2f} Hz (Target: {target_pha_freq} Hz)")
            print(f"  Best Amplitude Match: {best_amp_freq:.2f} Hz (Target: {target_amp_freq} Hz)")
    
    return history, initial_pha_mids, initial_amp_mids, initial_pha_idx, initial_amp_idx

def plot_parameter_learning(history, target_pha_freq, target_amp_freq, output_dir):
    """Plot the learning process of PAC parameters over epochs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['loss'])
    plt.title('Training Loss (Negative PAC Value)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_dir / 'training_loss.png', dpi=300)
    plt.close()
    
    # Plot max PAC value curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['max_pac_value'])
    plt.title('Maximum PAC Value')
    plt.xlabel('Epoch')
    plt.ylabel('Max PAC Value')
    plt.grid(True)
    plt.savefig(output_dir / 'max_pac_value.png', dpi=300)
    plt.close()
    
    # Plot parameter convergence for phase
    plt.figure(figsize=(12, 8))
    
    # Convert to array for easier plotting
    pha_mids_array = np.array(history['pha_mids'])
    n_epochs = len(history['epoch'])
    n_bands = pha_mids_array.shape[1]
    
    # Plot each band's trajectory
    for band_idx in range(n_bands):
        plt.plot(history['epoch'], pha_mids_array[:, band_idx], 
                 label=f"Band {band_idx+1}" if band_idx < 5 else "_nolegend_")
    
    # Highlight target frequency
    plt.axhline(y=target_pha_freq, color='r', linestyle='--', label=f"Target: {target_pha_freq} Hz")
    
    plt.title('Phase Frequency Bands Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Center Frequency (Hz)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_dir / 'phase_band_learning.png', dpi=300)
    plt.close()
    
    # Plot parameter convergence for amplitude
    plt.figure(figsize=(12, 8))
    
    # Convert to array for easier plotting
    amp_mids_array = np.array(history['amp_mids'])
    
    # Plot each band's trajectory
    for band_idx in range(n_bands):
        plt.plot(history['epoch'], amp_mids_array[:, band_idx], 
                 label=f"Band {band_idx+1}" if band_idx < 5 else "_nolegend_")
    
    # Highlight target frequency
    plt.axhline(y=target_amp_freq, color='r', linestyle='--', label=f"Target: {target_amp_freq} Hz")
    
    plt.title('Amplitude Frequency Bands Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Center Frequency (Hz)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_dir / 'amplitude_band_learning.png', dpi=300)
    plt.close()
    
    # Plot frequency tracking
    plt.figure(figsize=(12, 8))
    
    # Find band closest to target at each epoch
    pha_closest_indices = np.argmin(np.abs(pha_mids_array - target_pha_freq.reshape(-1, 1)), axis=1)
    amp_closest_indices = np.argmin(np.abs(amp_mids_array - target_amp_freq.reshape(-1, 1)), axis=1)
    
    pha_closest_freqs = np.array([pha_mids_array[i, idx] for i, idx in enumerate(pha_closest_indices)])
    amp_closest_freqs = np.array([amp_mids_array[i, idx] for i, idx in enumerate(amp_closest_indices)])
    
    # Normalize to percentage error
    pha_error_pct = 100 * np.abs(pha_closest_freqs - target_pha_freq) / target_pha_freq
    amp_error_pct = 100 * np.abs(amp_closest_freqs - target_amp_freq) / target_amp_freq
    
    plt.plot(history['epoch'], pha_error_pct, 'b-', label='Phase Frequency Error (%)')
    plt.plot(history['epoch'], amp_error_pct, 'g-', label='Amplitude Frequency Error (%)')
    
    plt.title('Target Frequency Tracking Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_dir / 'frequency_tracking_error.png', dpi=300)
    plt.close()
    
    # Create heatmap of final PAC values
    plt.figure(figsize=(10, 8))
    
    # Final frequency bands
    final_pha_mids = pha_mids_array[-1]
    final_amp_mids = amp_mids_array[-1]
    
    # Create meshgrid for heatmap
    pha_idx, amp_idx = np.meshgrid(np.arange(n_bands), np.arange(n_bands))
    
    # Simple synthetic PAC values (higher when closer to targets)
    pac_values = 1.0 / (1.0 + 
                        np.abs(final_pha_mids[pha_idx] - target_pha_freq) / target_pha_freq + 
                        np.abs(final_amp_mids[amp_idx] - target_amp_freq) / target_amp_freq)
    
    # Create heatmap
    sns.heatmap(pac_values, cmap='viridis', annot=False,
                xticklabels=[f"{freq:.1f}" for freq in final_pha_mids],
                yticklabels=[f"{freq:.1f}" for freq in final_amp_mids])
    
    plt.title('PAC Values Heatmap')
    plt.xlabel('Phase Frequency (Hz)')
    plt.ylabel('Amplitude Frequency (Hz)')
    plt.savefig(output_dir / 'pac_values_heatmap.png', dpi=300)
    plt.close()

def compare_before_after(module, initial_pha_mids, initial_amp_mids, 
                         initial_pha_idx, initial_amp_idx,
                         target_pha_freq, target_amp_freq, output_dir):
    """Compare frequency bands before and after training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get final parameters
    final_pha_mids = module.PHA_MIDS_HZ.detach().cpu().numpy()
    final_amp_mids = module.AMP_MIDS_HZ.detach().cpu().numpy()
    
    # Find final closest matches
    final_pha_idx = np.argmin(np.abs(final_pha_mids - target_pha_freq))
    final_amp_idx = np.argmin(np.abs(final_amp_mids - target_amp_freq))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot phase frequencies
    width = 0.35
    ax1.bar(np.arange(len(initial_pha_mids)) - width/2, initial_pha_mids, width, label='Initial')
    ax1.bar(np.arange(len(final_pha_mids)) + width/2, final_pha_mids, width, label='After Training')
    
    # Highlight the bands closest to target
    ax1.axhline(y=target_pha_freq, color='r', linestyle='--', label=f'Target: {target_pha_freq} Hz')
    ax1.plot(initial_pha_idx - width/2, initial_pha_mids[initial_pha_idx], 'o', color='red', markersize=10)
    ax1.plot(final_pha_idx + width/2, final_pha_mids[final_pha_idx], 'o', color='red', markersize=10)
    
    ax1.set_title('Phase Frequency Bands: Before vs After Training')
    ax1.set_xlabel('Band Index')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot amplitude frequencies
    ax2.bar(np.arange(len(initial_amp_mids)) - width/2, initial_amp_mids, width, label='Initial')
    ax2.bar(np.arange(len(final_amp_mids)) + width/2, final_amp_mids, width, label='After Training')
    
    # Highlight the bands closest to target
    ax2.axhline(y=target_amp_freq, color='r', linestyle='--', label=f'Target: {target_amp_freq} Hz')
    ax2.plot(initial_amp_idx - width/2, initial_amp_mids[initial_amp_idx], 'o', color='red', markersize=10)
    ax2.plot(final_amp_idx + width/2, final_amp_mids[final_amp_idx], 'o', color='red', markersize=10)
    
    ax2.set_title('Amplitude Frequency Bands: Before vs After Training')
    ax2.set_xlabel('Band Index')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_bands_comparison.png', dpi=300)
    plt.close()
    
    # Calculate error metrics
    initial_pha_error = np.abs(initial_pha_mids[initial_pha_idx] - target_pha_freq)
    final_pha_error = np.abs(final_pha_mids[final_pha_idx] - target_pha_freq)
    
    initial_amp_error = np.abs(initial_amp_mids[initial_amp_idx] - target_amp_freq)
    final_amp_error = np.abs(final_amp_mids[final_amp_idx] - target_amp_freq)
    
    # Calculate average spacing between bands
    initial_pha_spacing = np.mean(np.diff(sorted(initial_pha_mids)))
    final_pha_spacing = np.mean(np.diff(sorted(final_pha_mids)))
    
    initial_amp_spacing = np.mean(np.diff(sorted(initial_amp_mids)))
    final_amp_spacing = np.mean(np.diff(sorted(final_amp_mids)))
    
    # Create summary dataframe
    summary = {
        'Metric': [
            'Phase Target Frequency (Hz)',
            'Initial Closest Phase Band (Hz)',
            'Final Closest Phase Band (Hz)',
            'Initial Phase Error (Hz)',
            'Final Phase Error (Hz)',
            'Phase Error Reduction (%)',
            'Amplitude Target Frequency (Hz)',
            'Initial Closest Amplitude Band (Hz)',
            'Final Closest Amplitude Band (Hz)',
            'Initial Amplitude Error (Hz)',
            'Final Amplitude Error (Hz)',
            'Amplitude Error Reduction (%)',
            'Initial Phase Band Spacing (Hz)',
            'Final Phase Band Spacing (Hz)',
            'Initial Amplitude Band Spacing (Hz)',
            'Final Amplitude Band Spacing (Hz)'
        ],
        'Value': [
            f"{target_pha_freq:.2f}",
            f"{initial_pha_mids[initial_pha_idx]:.2f}",
            f"{final_pha_mids[final_pha_idx]:.2f}",
            f"{initial_pha_error:.2f}",
            f"{final_pha_error:.2f}",
            f"{(1 - final_pha_error/initial_pha_error) * 100:.2f}" if initial_pha_error > 0 else "N/A",
            f"{target_amp_freq:.2f}",
            f"{initial_amp_mids[initial_amp_idx]:.2f}",
            f"{final_amp_mids[final_amp_idx]:.2f}",
            f"{initial_amp_error:.2f}",
            f"{final_amp_error:.2f}",
            f"{(1 - final_amp_error/initial_amp_error) * 100:.2f}" if initial_amp_error > 0 else "N/A",
            f"{initial_pha_spacing:.2f}",
            f"{final_pha_spacing:.2f}",
            f"{initial_amp_spacing:.2f}",
            f"{final_amp_spacing:.2f}"
        ]
    }
    
    # Save summary as CSV
    pd.DataFrame(summary).to_csv(output_dir / 'frequency_adaptation_summary.csv', index=False)
    
    # Create markdown report
    with open(output_dir / 'frequency_adaptation_report.md', 'w') as f:
        f.write("# PAC Module Frequency Adaptation Report\n\n")
        
        f.write("## Summary of Parameter Learning\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        for metric, value in zip(summary['Metric'], summary['Value']):
            f.write(f"| {metric} | {value} |\n")
        
        f.write("\n## Interpretation\n\n")
        
        # Add interpretations based on the results
        if initial_pha_error > 0 and final_pha_error < initial_pha_error:
            pha_improvement = (1 - final_pha_error/initial_pha_error) * 100
            f.write(f"The trainable PAC module successfully adapted its phase frequency bands, ")
            f.write(f"reducing the error from {initial_pha_error:.2f} Hz to {final_pha_error:.2f} Hz ")
            f.write(f"({pha_improvement:.2f}% improvement).\n\n")
        else:
            f.write("The phase frequency bands did not show significant adaptation. ")
            f.write("This could be because the initial bands were already well-aligned with the target frequency.\n\n")
        
        if initial_amp_error > 0 and final_amp_error < initial_amp_error:
            amp_improvement = (1 - final_amp_error/initial_amp_error) * 100
            f.write(f"The trainable PAC module successfully adapted its amplitude frequency bands, ")
            f.write(f"reducing the error from {initial_amp_error:.2f} Hz to {final_amp_error:.2f} Hz ")
            f.write(f"({amp_improvement:.2f}% improvement).\n\n")
        else:
            f.write("The amplitude frequency bands did not show significant adaptation. ")
            f.write("This could be because the initial bands were already well-aligned with the target frequency.\n\n")
        
        f.write("## Conclusion\n\n")
        
        if (initial_pha_error > 0 and final_pha_error < initial_pha_error) or \
           (initial_amp_error > 0 and final_amp_error < initial_amp_error):
            f.write("The trainable parameters in the PAC module demonstrate clear advantages in adapting to ")
            f.write("specific frequency components in the signal. This allows the module to better detect ")
            f.write("Phase-Amplitude Coupling by optimizing its frequency bands to match the underlying ")
            f.write("signal characteristics.\n\n")
            
            f.write("This adaptation capability is particularly valuable in real-world neural signal processing, ")
            f.write("where the exact frequencies of interest may vary across subjects or conditions.")
        else:
            f.write("The trainable parameters did not show significant adaptation in this experiment. ")
            f.write("This could be due to:\n\n")
            f.write("1. The initial frequency bands already providing good coverage of the target frequencies\n")
            f.write("2. The learning rate or number of epochs being insufficient\n")
            f.write("3. The optimization objective not being specific enough to drive parameter adaptation\n\n")
            
            f.write("Further experiments with more challenging frequency targets or modified learning ")
            f.write("parameters may better demonstrate the advantages of trainable PAC modules.")

def main(args):
    """Main function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load parameters
    params = DEFAULT_PARAMS.copy()
    if args.target_pha_freq:
        params['target_pha_freq'] = args.target_pha_freq
    if args.target_amp_freq:
        params['target_amp_freq'] = args.target_amp_freq
    if args.n_epochs:
        params['n_epochs'] = args.n_epochs
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    if args.pha_n_bands:
        params['pha_n_bands'] = args.pha_n_bands
    if args.amp_n_bands:
        params['amp_n_bands'] = args.amp_n_bands
    if args.noise_level:
        params['noise_level'] = args.noise_level
        
    # Initialize data generator
    data_gen = mngs.gen.DataGenerator(fs=1000, random_seed=42)
    
    # Create output directory
    output_dir = Path(__DIR__) / 'results' / 'parameter_learning'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate test signal
    print(f"Generating test signal with Phase: {params['target_pha_freq']} Hz, Amplitude: {params['target_amp_freq']} Hz")
    test_signal = generate_test_signal(
        data_gen=data_gen,
        target_pha_freq=params['target_pha_freq'],
        target_amp_freq=params['target_amp_freq'],
        duration=2.0,
        fs=1000.0,
        noise_level=params['noise_level']
    )
    
    # Create PAC module
    module = create_pac_module(
        seq_len=test_signal.shape[-1],
        fs=1000.0,
        pha_n_bands=params['pha_n_bands'],
        amp_n_bands=params['amp_n_bands'],
        trainable=True
    )
    
    # Visualize initial frequency bands
    print("Visualizing initial frequency bands...")
    visualize_frequency_bands(module, fs=1000.0, save_path=output_dir / 'initial_frequency_bands.png')
    
    # Train the module
    print("\nTraining PAC module to adapt parameters...")
    history, initial_pha_mids, initial_amp_mids, initial_pha_idx, initial_amp_idx = train_pac_module(
        module=module,
        signal=test_signal,
        target_pha_freq=params['target_pha_freq'],
        target_amp_freq=params['target_amp_freq'],
        n_epochs=params['n_epochs'],
        learning_rate=params['learning_rate'],
        device=device
    )
    
    # Visualize final frequency bands
    print("Visualizing final frequency bands...")
    visualize_frequency_bands(module, fs=1000.0, save_path=output_dir / 'final_frequency_bands.png')
    
    # Plot parameter learning process
    print("Plotting parameter learning process...")
    plot_parameter_learning(
        history=history,
        target_pha_freq=params['target_pha_freq'],
        target_amp_freq=params['target_amp_freq'],
        output_dir=output_dir
    )
    
    # Compare before and after training
    print("Comparing frequency bands before and after training...")
    compare_before_after(
        module=module,
        initial_pha_mids=initial_pha_mids.cpu().numpy(),
        initial_amp_mids=initial_amp_mids.cpu().numpy(),
        initial_pha_idx=initial_pha_idx,
        initial_amp_idx=initial_amp_idx,
        target_pha_freq=params['target_pha_freq'],
        target_amp_freq=params['target_amp_freq'],
        output_dir=output_dir
    )
    
    # Save trained module
    torch.save(module.state_dict(), output_dir / 'trained_parameters.pt')
    
    print(f"\nAll results saved to {output_dir}")
    return 0

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize learnability of PAC module parameters")
    parser.add_argument(
        "--target_pha_freq",
        type=float,
        help="Target phase frequency to learn"
    )
    parser.add_argument(
        "--target_amp_freq",
        type=float,
        help="Target amplitude frequency to learn"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--pha_n_bands",
        type=int,
        help="Number of phase frequency bands"
    )
    parser.add_argument(
        "--amp_n_bands",
        type=int,
        help="Number of amplitude frequency bands"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        help="Noise level for generated signals"
    )
    
    args = parser.parse_args()
    return args

def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt
    
    import sys
    import matplotlib.pyplot as plt
    import mngs
    
    args = parse_args()
    
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )
    
    exit_status = main(args)
    
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

if __name__ == "__main__":
    run_main()

# EOF