#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-13 21:48:04 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/generate_synthetic_data.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/generate_synthetic_data.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Generates synthetic Phase-Amplitude Coupling (PAC) signals
  - Creates data with known coupling frequencies for training and testing
  - Saves generated signals to disk for later use
  - Supports various signal parameters and noise levels

Dependencies:
  - packages:
    - mngs
    - numpy
    - torch

IO:
  - output-files:
    - ./scripts/learnability/data/synthetic_pac_signals.npz
    - ./scripts/learnability/data/synthetic_pac_signals.pt
"""

"""Imports"""
import argparse
from pathlib import Path
import numpy as np
import torch
import mngs
import matplotlib.pyplot as plt

# Create a simple local version of DataGenerator since mngs.gen doesn't provide it
class DataGenerator:
    """
    Simple data generator for creating synthetic PAC signals.
    """
    def __init__(self, fs=1000.0, random_seed=None):
        self.fs = fs
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def generate_pac_with_signal(self, n_seconds, pha_mod_freq, amp_car_freq, 
                                pha_bandwidth, amp_bandwidth, coupling_strength=0.8, 
                                noise_level=0.1):
        """Generate a PAC signal with a specified coupling relationship."""
        # Create time vector
        t = np.arange(0, n_seconds, 1/self.fs)
        
        # Create phase signal (slow oscillation)
        phase_signal = np.sin(2 * np.pi * pha_mod_freq * t)
        
        # Create amplitude modulation based on phase
        modulation = (1 + coupling_strength * np.cos(2 * np.pi * pha_mod_freq * t)) / 2
        
        # Create carrier signal (fast oscillation)
        carrier = np.sin(2 * np.pi * amp_car_freq * t)
        
        # Apply amplitude modulation to carrier
        modulated_carrier = modulation * carrier
        
        # Create final signal with both components
        pac_signal = phase_signal + modulated_carrier
        
        # Add some noise
        noise = np.random.normal(0, noise_level, len(t))
        signal = pac_signal + noise
        
        return signal

"""Parameters"""
DEFAULT_PARAMS = {
    'fs': 1000.0,           # Sampling frequency in Hz
    'duration': 2.0,        # Signal duration in seconds
    'pha_freqs': [4.0, 8.0, 12.0],  # Phase frequencies to generate
    'amp_freqs': [80.0, 100.0, 120.0],  # Amplitude frequencies to generate
    'n_samples': 20,        # Number of samples per frequency combination
    'n_channels': 3,        # Number of channels
    'n_segments': 1,        # Number of segments
    'noise_levels': [0.1, 0.2, 0.3]  # Noise levels to use
}

"""Functions & Classes"""
def generate_synthetic_pac_signals(params):
    """
    Generate synthetic PAC signals with known coupling frequencies.
    
    Args:
        params: Dictionary with signal generation parameters
        
    Returns:
        Dictionary containing generated signals and metadata
    """
    data_gen = DataGenerator(fs=params['fs'], random_seed=42)
    
    # Calculate total number of signals
    n_pha_freqs = len(params['pha_freqs'])
    n_amp_freqs = len(params['amp_freqs'])
    n_noise_levels = len(params['noise_levels'])
    total_signals = n_pha_freqs * n_amp_freqs * n_noise_levels * params['n_samples']
    
    # Pre-allocate arrays
    seq_len = int(params['duration'] * params['fs'])
    signals_np = np.zeros((
        total_signals, 
        params['n_channels'], 
        params['n_segments'], 
        seq_len
    ))
    
    # Create metadata arrays
    metadata = {
        'pha_freqs': np.zeros(total_signals),
        'amp_freqs': np.zeros(total_signals),
        'noise_levels': np.zeros(total_signals),
        'sample_ids': np.zeros(total_signals, dtype=int)
    }
    
    # Generate signals
    signal_idx = 0
    for pha_idx, pha_freq in enumerate(params['pha_freqs']):
        for amp_idx, amp_freq in enumerate(params['amp_freqs']):
            for noise_idx, noise_level in enumerate(params['noise_levels']):
                for sample_idx in range(params['n_samples']):
                    # Generate PAC signal with MNGS
                    pac_data = data_gen.generate_pac_with_signal(
                        n_seconds=params['duration'],
                        pha_mod_freq=pha_freq,
                        amp_car_freq=amp_freq,
                        pha_bandwidth=pha_freq/4,
                        amp_bandwidth=amp_freq/4,
                        coupling_strength=0.8,
                        noise_level=noise_level
                    )
                    
                    # Reshape to match expected dimensions
                    signal_1d = pac_data.reshape(1, 1, 1, -1)
                    
                    # Repeat for all channels
                    for channel_idx in range(params['n_channels']):
                        for segment_idx in range(params['n_segments']):
                            signals_np[signal_idx, channel_idx, segment_idx, :] = signal_1d[0, 0, 0, :]
                    
                    # Store metadata
                    metadata['pha_freqs'][signal_idx] = pha_freq
                    metadata['amp_freqs'][signal_idx] = amp_freq
                    metadata['noise_levels'][signal_idx] = noise_level
                    metadata['sample_ids'][signal_idx] = sample_idx
                    
                    signal_idx += 1
                    
                    # Progress update
                    if signal_idx % 10 == 0:
                        print(f"Generated {signal_idx}/{total_signals} signals")
    
    # Create PyTorch tensor version
    signals_pt = torch.from_numpy(signals_np.astype(np.float32))
    
    return {
        'signals_np': signals_np,
        'signals_pt': signals_pt,
        'metadata': metadata,
        'params': params
    }

def save_synthetic_data(data, output_dir):
    """Save generated data to disk."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save numpy version
    np.savez(
        output_dir / 'synthetic_pac_signals.npz',
        signals=data['signals_np'],
        pha_freqs=data['metadata']['pha_freqs'],
        amp_freqs=data['metadata']['amp_freqs'],
        noise_levels=data['metadata']['noise_levels'],
        sample_ids=data['metadata']['sample_ids'],
        **{f'param_{k}': v for k, v in data['params'].items()}
    )
    
    # Save PyTorch version
    torch.save({
        'signals': data['signals_pt'],
        'metadata': data['metadata'],
        'params': data['params']
    }, output_dir / 'synthetic_pac_signals.pt')
    
    print(f"Data saved to {output_dir}")

def plot_example_signals(data, output_dir):
    """Plot example signals for visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot a few example signals
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Example Synthetic PAC Signals')
    
    # Get random indices for different combinations
    np.random.seed(42)
    example_indices = np.random.choice(len(data['signals_np']), 9, replace=False)
    
    for i, ax in enumerate(axes.flat):
        idx = example_indices[i]
        signal = data['signals_np'][idx, 0, 0, :]  # First channel, first segment
        
        # Time vector
        fs = data['params']['fs']
        duration = data['params']['duration']
        time_vector = np.linspace(0, duration, len(signal))
        
        # Plot signal
        ax.plot(time_vector, signal)
        
        # Add metadata
        pha_freq = data['metadata']['pha_freqs'][idx]
        amp_freq = data['metadata']['amp_freqs'][idx]
        noise = data['metadata']['noise_levels'][idx]
        
        ax.set_title(f'P:{pha_freq:.1f}Hz, A:{amp_freq:.1f}Hz, N:{noise:.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'example_signals.png', dpi=300)
    plt.close(fig)

def main(args):
    """Main function to generate and save synthetic data."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Prepare parameters
    params = DEFAULT_PARAMS.copy()
    if args.pha_freqs:
        params['pha_freqs'] = args.pha_freqs
    if args.amp_freqs:
        params['amp_freqs'] = args.amp_freqs
    if args.n_samples:
        params['n_samples'] = args.n_samples
    
    # Create output directory
    output_dir = Path(__DIR__) / 'data'
    
    # Generate synthetic data
    print("Generating synthetic PAC signals...")
    generated_data = generate_synthetic_pac_signals(params)
    
    # Save data
    print("Saving generated signals...")
    save_synthetic_data(generated_data, output_dir)
    
    # Generate visualizations
    print("Creating visualizations...")
    plot_example_signals(generated_data, output_dir)
    
    print(f"Generation completed successfully. {len(generated_data['signals_np'])} signals generated.")
    return 0

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic PAC signals")
    parser.add_argument(
        "--pha_freqs",
        type=float,
        nargs='+',
        help="Phase frequencies to generate"
    )
    parser.add_argument(
        "--amp_freqs",
        type=float,
        nargs='+',
        help="Amplitude frequencies to generate"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples per frequency combination"
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