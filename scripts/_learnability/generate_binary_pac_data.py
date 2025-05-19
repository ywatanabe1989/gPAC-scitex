#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 01:25:04 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/generate_binary_pac_data.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/generate_binary_pac_data.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Generates synthetic Phase-Amplitude Coupling (PAC) signals for binary classification
  - Creates two distinct signal classes with different PAC characteristics
  - Class A: low-frequency phase coupled with medium-frequency amplitude
  - Class B: medium-frequency phase coupled with high-frequency amplitude
  - Saves generated signals to disk for later use in classification tasks
  - Supports various signal parameters and noise levels

Dependencies:
  - packages:
    - mngs
    - numpy
    - torch
    - matplotlib

IO:
  - output-files:
    - ./scripts/learnability/data/binary_pac_signals.npz
    - ./scripts/learnability/data/binary_pac_signals.pt
    - ./scripts/learnability/data/binary_pac_examples.png
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
    'class_a_pha_range': [4.0, 8.0],  # Class A: Phase frequency range (theta)
    'class_a_amp_range': [80.0, 100.0],  # Class A: Amplitude frequency range (low gamma)
    'class_b_pha_range': [10.0, 14.0],  # Class B: Phase frequency range (alpha)
    'class_b_amp_range': [120.0, 150.0],  # Class B: Amplitude frequency range (high gamma)
    'n_samples': 100,       # Number of samples per class
    'n_channels': 3,        # Number of channels
    'n_segments': 1,        # Number of segments
    'noise_levels': [0.1, 0.2, 0.3, 0.4, 0.5]  # Noise levels to use
}

"""Functions & Classes"""
def generate_binary_pac_signals(params):
    """
    Generate synthetic PAC signals for binary classification.
    
    Args:
        params: Dictionary with signal generation parameters
        
    Returns:
        Dictionary containing generated signals and metadata
    """
    data_gen = DataGenerator(fs=params['fs'], random_seed=42)
    
    # Calculate total number of signals (two classes)
    n_noise_levels = len(params['noise_levels'])
    total_signals = 2 * params['n_samples'] * n_noise_levels
    
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
        'class_labels': np.zeros(total_signals, dtype=int),  # 0 for Class A, 1 for Class B
        'pha_freqs': np.zeros(total_signals),
        'amp_freqs': np.zeros(total_signals),
        'noise_levels': np.zeros(total_signals),
        'sample_ids': np.zeros(total_signals, dtype=int)
    }
    
    # Generate signals
    signal_idx = 0
    
    # Generate Class A signals
    for noise_idx, noise_level in enumerate(params['noise_levels']):
        for sample_idx in range(params['n_samples']):
            # Random frequencies from ranges
            pha_freq = np.random.uniform(params['class_a_pha_range'][0], params['class_a_pha_range'][1])
            amp_freq = np.random.uniform(params['class_a_amp_range'][0], params['class_a_amp_range'][1])
            
            # Generate PAC signal
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
            metadata['class_labels'][signal_idx] = 0  # Class A
            metadata['pha_freqs'][signal_idx] = pha_freq
            metadata['amp_freqs'][signal_idx] = amp_freq
            metadata['noise_levels'][signal_idx] = noise_level
            metadata['sample_ids'][signal_idx] = sample_idx
            
            signal_idx += 1
            
            # Progress update
            if signal_idx % 20 == 0:
                print(f"Generated {signal_idx}/{total_signals} signals")
    
    # Generate Class B signals
    for noise_idx, noise_level in enumerate(params['noise_levels']):
        for sample_idx in range(params['n_samples']):
            # Random frequencies from ranges
            pha_freq = np.random.uniform(params['class_b_pha_range'][0], params['class_b_pha_range'][1])
            amp_freq = np.random.uniform(params['class_b_amp_range'][0], params['class_b_amp_range'][1])
            
            # Generate PAC signal
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
            metadata['class_labels'][signal_idx] = 1  # Class B
            metadata['pha_freqs'][signal_idx] = pha_freq
            metadata['amp_freqs'][signal_idx] = amp_freq
            metadata['noise_levels'][signal_idx] = noise_level
            metadata['sample_ids'][signal_idx] = sample_idx
            
            signal_idx += 1
            
            # Progress update
            if signal_idx % 20 == 0:
                print(f"Generated {signal_idx}/{total_signals} signals")
    
    # Create PyTorch tensor version
    signals_pt = torch.from_numpy(signals_np.astype(np.float32))
    
    return {
        'signals_np': signals_np,
        'signals_pt': signals_pt,
        'metadata': metadata,
        'params': params
    }

def save_binary_data(data, output_dir):
    """Save generated data to disk."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save numpy version
    np.savez(
        output_dir / 'binary_pac_signals.npz',
        signals=data['signals_np'],
        class_labels=data['metadata']['class_labels'],
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
    }, output_dir / 'binary_pac_signals.pt')
    
    print(f"Data saved to {output_dir}")

def plot_example_signals(data, output_dir):
    """Plot example signals for visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot examples from each class
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Example Synthetic PAC Signals for Binary Classification')
    
    # Get indices for class A (top row)
    class_a_indices = np.where(data['metadata']['class_labels'] == 0)[0]
    # Get indices for class B (bottom row)
    class_b_indices = np.where(data['metadata']['class_labels'] == 1)[0]
    
    # Select three random examples from each class
    np.random.seed(42)
    class_a_examples = np.random.choice(class_a_indices, 3, replace=False)
    class_b_examples = np.random.choice(class_b_indices, 3, replace=False)
    
    # Plot Class A examples (top row)
    for i, ax in enumerate(axes[0]):
        idx = class_a_examples[i]
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
        
        ax.set_title(f'Class A - P:{pha_freq:.1f}Hz, A:{amp_freq:.1f}Hz, N:{noise:.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    # Plot Class B examples (bottom row)
    for i, ax in enumerate(axes[1]):
        idx = class_b_examples[i]
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
        
        ax.set_title(f'Class B - P:{pha_freq:.1f}Hz, A:{amp_freq:.1f}Hz, N:{noise:.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_pac_examples.png', dpi=300)
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
    if args.n_samples:
        params['n_samples'] = args.n_samples
    
    # Create output directory
    output_dir = Path(__DIR__) / 'data'
    
    # Generate synthetic data
    print("Generating binary PAC signals...")
    generated_data = generate_binary_pac_signals(params)
    
    # Save data
    print("Saving generated signals...")
    save_binary_data(generated_data, output_dir)
    
    # Generate visualizations
    print("Creating visualizations...")
    plot_example_signals(generated_data, output_dir)
    
    print(f"Generation completed successfully. {len(generated_data['signals_np'])} signals generated.")
    return 0

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic PAC signals for binary classification")
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples per class"
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