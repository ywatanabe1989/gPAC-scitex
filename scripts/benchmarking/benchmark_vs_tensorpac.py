#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 03:32:11 (Claude)"
# File: /home/ywatanabe/proj/gPAC/scripts/benchmarking/benchmark_vs_tensorpac.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/benchmarking/benchmark_vs_tensorpac.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Benchmark gPAC against Tensorpac for PAC calculation performance and accuracy.

This script:
1. Generates synthetic signals with known PAC properties
2. Measures computation time for both implementations
3. Compares calculation results for accuracy
4. Tests various parameter combinations from config
5. Produces visualization data for manuscript figures

Dependencies:
  - torch
  - numpy
  - tensorpac
  - gpac
  - pandas
  - matplotlib
  - seaborn
  - pyyaml
  - psutil (for resource monitoring)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import yaml
from tqdm import tqdm

# Add project root to Python path
import sys
project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))

# Import gPAC
import gpac

# Import Tensorpac (wrapped in try-except in case it's not installed)
try:
    import tensorpac
    TENSORPAC_AVAILABLE = True
except ImportError:
    print("Tensorpac not found. Install with: pip install tensorpac")
    TENSORPAC_AVAILABLE = False

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Constants
OUTPUT_DIR = Path(__DIR__) / ".." / ".." / "paper" / "manuscript" / "figures" / "src"
RESULTS_DIR = Path(__DIR__) / ".." / ".." / "results" / "benchmarking"
CONFIG_PATH = Path(__DIR__) / ".." / ".." / "config" / "PARAMS.yaml"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


class ResourceMonitor:
    """Monitor and record system resource usage during computation."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.cpu_percent = []
        self.ram_usage = []
        self.gpu_memory = []
        self.start_time = None
        self.end_time = None
        
        # Check if GPU is available
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats()
    
    def start(self):
        """Start monitoring resources."""
        self.start_time = time.time()
        self.cpu_percent = []
        self.ram_usage = []
        self.gpu_memory = []
    
    def sample(self):
        """Take a sample of current resource usage."""
        self.cpu_percent.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().used / (1024 ** 3))  # GiB
        
        if self.has_gpu:
            self.gpu_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 3))  # GiB
    
    def stop(self):
        """Stop monitoring and return results."""
        self.end_time = time.time()
        
        results = {
            "elapsed_time": self.end_time - self.start_time,
            "cpu_percent_mean": np.mean(self.cpu_percent) if self.cpu_percent else 0,
            "cpu_percent_max": np.max(self.cpu_percent) if self.cpu_percent else 0,
            "ram_usage_mean": np.mean(self.ram_usage) if self.ram_usage else 0,
            "ram_usage_max": np.max(self.ram_usage) if self.ram_usage else 0,
        }
        
        if self.has_gpu:
            results["gpu_memory_max"] = np.max(self.gpu_memory) if self.gpu_memory else 0
            # Reset peak stats for next run
            torch.cuda.reset_peak_memory_stats()
            
        return results


def generate_synthetic_pac_signal(
    fs: float = 1000.0,
    duration: float = 5.0,
    pha_freq: float = 8.0,
    amp_freq: float = 100.0,
    coupling_strength: float = 0.8,
    noise_level: float = 0.2,
    batch_size: int = 1,
    n_channels: int = 1,
    n_segments: int = 1,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate synthetic signal with known PAC properties.
    
    Args:
        fs: Sampling frequency in Hz
        duration: Signal duration in seconds
        pha_freq: Frequency of phase component in Hz
        amp_freq: Frequency of amplitude component in Hz
        coupling_strength: Strength of PAC (0-1)
        noise_level: Amount of noise to add
        batch_size: Number of signal batches
        n_channels: Number of channels
        n_segments: Number of segments per channel
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (signal tensor, metadata dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Create time vector
    t = np.arange(0, duration, 1/fs)
    seq_len = len(t)
    
    # Create phase signal (slow oscillation)
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    
    # Create amplitude modulation based on phase
    modulation = (1 + coupling_strength * np.cos(2 * np.pi * pha_freq * t)) / 2
    
    # Create carrier signal (fast oscillation)
    carrier = np.sin(2 * np.pi * amp_freq * t)
    
    # Apply amplitude modulation to carrier
    modulated_carrier = modulation * carrier
    
    # Create final signal with both components
    pac_signal = phase_signal + modulated_carrier
    
    # Add some noise
    noise = np.random.normal(0, noise_level, len(t))
    signal = pac_signal + noise
    
    # Create tensor with batch and channel dimensions
    # Shape: (batch_size, n_channels, n_segments, seq_len)
    signal_tensor = np.zeros((batch_size, n_channels, n_segments, seq_len))
    
    # Fill all batches and channels with the same signal (for simplicity)
    # In a real scenario, you might want different signals per batch/channel
    for b in range(batch_size):
        for c in range(n_channels):
            for s in range(n_segments):
                signal_tensor[b, c, s, :] = signal
    
    signal_tensor = torch.from_numpy(signal_tensor.astype(np.float32))
    
    # Return the tensor and metadata
    metadata = {
        "fs": fs,
        "duration": duration,
        "pha_freq": pha_freq,
        "amp_freq": amp_freq,
        "coupling_strength": coupling_strength,
        "noise_level": noise_level,
        "shape": signal_tensor.shape
    }
    
    return signal_tensor, metadata


def benchmark_gpac(
    signal: torch.Tensor,
    fs: float,
    pha_start_hz: float,
    pha_end_hz: float,
    pha_n_bands: int,
    amp_start_hz: float,
    amp_end_hz: float,
    amp_n_bands: int,
    n_perm: Optional[int] = None,
    device: str = 'cpu',
    fp16: bool = False,
    chunk_size: Optional[int] = None,
    monitor_resources: bool = True
) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    Benchmark gPAC performance and return results.
    
    Args:
        signal: Input signal tensor (batch, channels, segments, time)
        fs: Sampling frequency in Hz
        pha_start_hz, pha_end_hz: Phase frequency range
        pha_n_bands: Number of phase frequency bands
        amp_start_hz, amp_end_hz: Amplitude frequency range 
        amp_n_bands: Number of amplitude frequency bands
        n_perm: Number of permutations for statistical testing
        device: Computation device ('cpu' or 'cuda')
        fp16: Whether to use half-precision
        chunk_size: Size of processing chunks
        monitor_resources: Whether to monitor resource usage
        
    Returns:
        Tuple of (PAC values, timing information, resource usage)
    """
    # Initialize resource monitor if requested
    monitor = ResourceMonitor() if monitor_resources else None
    
    # Record timing for different stages
    timings = {}
    
    # Start initialization timing
    t_start_init = time.time()
    
    # Move signal to device if needed
    if device == 'cuda' and torch.cuda.is_available():
        signal = signal.cuda()
    
    # Start resource monitoring
    if monitor:
        monitor.start()
    
    # Time the initialization stage
    t_end_init = time.time()
    timings['initialization'] = t_end_init - t_start_init
    
    # Start computation timing
    t_start_compute = time.time()
    
    # Sample resources periodically during computation
    if monitor:
        # Start a background thread for sampling
        import threading
        stop_sampling = threading.Event()
        
        def sample_thread():
            while not stop_sampling.is_set():
                monitor.sample()
                time.sleep(0.1)  # Sample every 100ms
        
        sampling_thread = threading.Thread(target=sample_thread)
        sampling_thread.start()
    
    # Run the PAC calculation
    try:
        pac_values, freqs_pha, freqs_amp = gpac.calculate_pac(
            signal=signal,
            fs=fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            trainable=False,
            fp16=fp16,
            device=device,
            chunk_size=chunk_size
        )
        
        computation_success = True
    except Exception as e:
        print(f"Error during gPAC calculation: {e}")
        pac_values = None
        freqs_pha = None
        freqs_amp = None
        computation_success = False
    
    # End computation timing
    t_end_compute = time.time()
    timings['computation'] = t_end_compute - t_start_compute
    timings['total'] = timings['initialization'] + timings['computation']
    
    # Stop resource monitoring
    if monitor:
        # Stop the sampling thread
        if 'stop_sampling' in locals() and 'sampling_thread' in locals():
            stop_sampling.set()
            sampling_thread.join()
        
        resource_usage = monitor.stop()
    else:
        resource_usage = {}
    
    # Return results
    result = {
        'computation_success': computation_success,
        'pac_values': pac_values,
        'freqs_pha': freqs_pha,
        'freqs_amp': freqs_amp,
        'timings': timings,
        'resources': resource_usage
    }
    
    # Convert to CPU for consistent return format
    if computation_success and device == 'cuda' and torch.cuda.is_available():
        result['pac_values'] = pac_values.cpu()
    
    return result


def benchmark_tensorpac(
    signal: torch.Tensor,
    fs: float,
    pha_start_hz: float,
    pha_end_hz: float,
    pha_n_bands: int,
    amp_start_hz: float,
    amp_end_hz: float,
    amp_n_bands: int,
    n_perm: Optional[int] = None,
    monitor_resources: bool = True
) -> Dict:
    """
    Benchmark Tensorpac performance and return results.
    
    Args:
        signal: Input signal tensor (batch, channels, segments, time)
        fs: Sampling frequency in Hz
        pha_start_hz, pha_end_hz: Phase frequency range
        pha_n_bands: Number of phase frequency bands
        amp_start_hz, amp_end_hz: Amplitude frequency range 
        amp_n_bands: Number of amplitude frequency bands
        n_perm: Number of permutations for statistical testing
        monitor_resources: Whether to monitor resource usage
        
    Returns:
        Dict with results and timing information
    """
    if not TENSORPAC_AVAILABLE:
        return {
            'computation_success': False,
            'error': "Tensorpac not installed"
        }
    
    # Initialize resource monitor if requested
    monitor = ResourceMonitor() if monitor_resources else None
    
    # Record timing for different stages
    timings = {}
    
    # Start initialization timing
    t_start_init = time.time()
    
    # Convert PyTorch tensor to NumPy array
    signal_np = signal.numpy()
    
    # Reshape to format expected by Tensorpac
    # Tensorpac expects (n_epochs, n_channels, n_points)
    # Our signal is (batch_size, n_channels, n_segments, n_points)
    batch_size, n_channels, n_segments, n_points = signal_np.shape
    signal_tp = signal_np.reshape(batch_size * n_segments, n_channels, n_points)
    
    # Create phase and amplitude frequency vectors
    pha_freqs = np.linspace(pha_start_hz, pha_end_hz, pha_n_bands)
    amp_freqs = np.linspace(amp_start_hz, amp_end_hz, amp_n_bands)
    
    # Initialize Tensorpac instance
    tp = tensorpac.Pac(idpac=(6, 0, 3), f_pha=pha_freqs, f_amp=amp_freqs)
    
    # Time the initialization stage
    t_end_init = time.time()
    timings['initialization'] = t_end_init - t_start_init
    
    # Start computation timing
    t_start_compute = time.time()
    
    # Start resource monitoring
    if monitor:
        monitor.start()
        
        # Sample resources periodically during computation
        import threading
        stop_sampling = threading.Event()
        
        def sample_thread():
            while not stop_sampling.is_set():
                monitor.sample()
                time.sleep(0.1)  # Sample every 100ms
        
        sampling_thread = threading.Thread(target=sample_thread)
        sampling_thread.start()
    
    # Run the PAC calculation
    try:
        # Compute PAC
        if n_perm:
            # With permutation testing
            pac_values = tp.filterfit(signal_tp, fs, n_perm=n_perm)
        else:
            # Without permutation testing
            pac_values = tp.filterfit(signal_tp, fs)
        
        computation_success = True
    except Exception as e:
        print(f"Error during Tensorpac calculation: {e}")
        pac_values = None
        computation_success = False
    
    # End computation timing
    t_end_compute = time.time()
    timings['computation'] = t_end_compute - t_start_compute
    timings['total'] = timings['initialization'] + timings['computation']
    
    # Stop resource monitoring
    if monitor:
        # Stop the sampling thread
        if 'stop_sampling' in locals() and 'sampling_thread' in locals():
            stop_sampling.set()
            sampling_thread.join()
            
        resource_usage = monitor.stop()
    else:
        resource_usage = {}
    
    # Format results to match gPAC output
    result = {
        'computation_success': computation_success,
        'pac_values': pac_values,
        'freqs_pha': pha_freqs,
        'freqs_amp': amp_freqs,
        'timings': timings,
        'resources': resource_usage
    }
    
    # Convert to torch tensor for consistent return format
    if computation_success:
        result['pac_values'] = torch.from_numpy(pac_values)
    
    return result


def run_parameter_sweep(config_path: Path = CONFIG_PATH) -> pd.DataFrame:
    """
    Run benchmarks with different parameter combinations from config file.
    
    Args:
        config_path: Path to YAML config file with parameter combinations
        
    Returns:
        DataFrame with benchmark results
    """
    # Load parameter configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Base parameter set
    base_params = config.get('base_params', {})
    
    # Parameter variations
    param_variations = config.get('param_variations', {})
    
    # Prepare results dataframe
    results = []
    
    # Run benchmark for base parameters
    print(f"Running benchmark with base parameters: {base_params}")
    base_result = run_single_benchmark(**base_params)
    base_result.update({'param_set': 'base'})
    results.append(base_result)
    
    # Run benchmark for each parameter variation
    for param_name, values in param_variations.items():
        for value in values:
            # Copy base parameters and update the varied parameter
            params = base_params.copy()
            params[param_name] = value
            
            print(f"Running benchmark with {param_name}={value}")
            result = run_single_benchmark(**params)
            result.update({
                'param_set': f"{param_name}={value}",
                'varied_param': param_name,
                'param_value': value
            })
            results.append(result)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"parameter_sweep_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return df


def run_single_benchmark(**kwargs) -> Dict:
    """
    Run a single benchmark with the specified parameters.
    
    Args:
        kwargs: Parameters for the benchmark
        
    Returns:
        Dict with benchmark results
    """
    # Default parameters
    params = {
        'fs': 1000.0,
        'duration': 5.0,
        'pha_freq': 8.0,
        'amp_freq': 100.0,
        'coupling_strength': 0.8,
        'noise_level': 0.2,
        'batch_size': 1,
        'n_channels': 1,
        'n_segments': 1,
        'pha_start_hz': 4.0,
        'pha_end_hz': 16.0,
        'pha_n_bands': 10,
        'amp_start_hz': 80.0,
        'amp_end_hz': 160.0,
        'amp_n_bands': 10,
        'n_perm': None,
        'device': 'cpu',
        'fp16': False,
        'chunk_size': None,
        'n_runs': 10,
        'random_seed': 42
    }
    
    # Update with provided parameters
    params.update(kwargs)
    
    # Generate synthetic signal
    signal, metadata = generate_synthetic_pac_signal(
        fs=params['fs'],
        duration=params['duration'],
        pha_freq=params['pha_freq'],
        amp_freq=params['amp_freq'],
        coupling_strength=params['coupling_strength'],
        noise_level=params['noise_level'],
        batch_size=params['batch_size'],
        n_channels=params['n_channels'],
        n_segments=params['n_segments'],
        random_seed=params['random_seed']
    )
    
    # Benchmark results
    gpac_results = []
    tp_results = []
    
    # Run benchmark multiple times for statistical significance
    for run in range(params['n_runs']):
        print(f"Run {run+1}/{params['n_runs']}")
        
        # Benchmark gPAC
        gpac_result = benchmark_gpac(
            signal=signal,
            fs=params['fs'],
            pha_start_hz=params['pha_start_hz'],
            pha_end_hz=params['pha_end_hz'],
            pha_n_bands=params['pha_n_bands'],
            amp_start_hz=params['amp_start_hz'],
            amp_end_hz=params['amp_end_hz'],
            amp_n_bands=params['amp_n_bands'],
            n_perm=params['n_perm'],
            device=params['device'],
            fp16=params['fp16'],
            chunk_size=params['chunk_size']
        )
        gpac_results.append(gpac_result)
        
        # Benchmark Tensorpac
        if TENSORPAC_AVAILABLE:
            tp_result = benchmark_tensorpac(
                signal=signal,
                fs=params['fs'],
                pha_start_hz=params['pha_start_hz'],
                pha_end_hz=params['pha_end_hz'],
                pha_n_bands=params['pha_n_bands'],
                amp_start_hz=params['amp_start_hz'],
                amp_end_hz=params['amp_end_hz'],
                amp_n_bands=params['amp_n_bands'],
                n_perm=params['n_perm']
            )
            tp_results.append(tp_result)
    
    # Compile results
    result = {
        # Input parameters
        **params,
        
        # Signal metadata
        **metadata,
        
        # gPAC results
        'gpac_success_rate': sum([r['computation_success'] for r in gpac_results]) / params['n_runs'],
        'gpac_time_mean': np.mean([r['timings']['total'] for r in gpac_results if r['computation_success']]),
        'gpac_time_std': np.std([r['timings']['total'] for r in gpac_results if r['computation_success']]),
        'gpac_init_time_mean': np.mean([r['timings']['initialization'] for r in gpac_results if r['computation_success']]),
        'gpac_compute_time_mean': np.mean([r['timings']['computation'] for r in gpac_results if r['computation_success']]),
        'gpac_ram_usage_mean': np.mean([r['resources'].get('ram_usage_mean', 0) for r in gpac_results if r['computation_success']]),
        'gpac_cpu_percent_mean': np.mean([r['resources'].get('cpu_percent_mean', 0) for r in gpac_results if r['computation_success']]),
    }
    
    # Add GPU metrics if available
    if params['device'] == 'cuda' and torch.cuda.is_available():
        result['gpac_gpu_memory_mean'] = np.mean([r['resources'].get('gpu_memory_max', 0) for r in gpac_results if r['computation_success']])
    
    # Add Tensorpac results if available
    if TENSORPAC_AVAILABLE and tp_results:
        result.update({
            'tp_success_rate': sum([r['computation_success'] for r in tp_results]) / params['n_runs'],
            'tp_time_mean': np.mean([r['timings']['total'] for r in tp_results if r['computation_success']]),
            'tp_time_std': np.std([r['timings']['total'] for r in tp_results if r['computation_success']]),
            'tp_init_time_mean': np.mean([r['timings']['initialization'] for r in tp_results if r['computation_success']]),
            'tp_compute_time_mean': np.mean([r['timings']['computation'] for r in tp_results if r['computation_success']]),
            'tp_ram_usage_mean': np.mean([r['resources'].get('ram_usage_mean', 0) for r in tp_results if r['computation_success']]),
            'tp_cpu_percent_mean': np.mean([r['resources'].get('cpu_percent_mean', 0) for r in tp_results if r['computation_success']])
        })
        
        # Calculate speedup
        if result['gpac_success_rate'] > 0 and result['tp_success_rate'] > 0:
            result['speedup'] = result['tp_time_mean'] / result['gpac_time_mean']
    
    # Calculate similarity between gPAC and Tensorpac results if both succeeded
    if (result.get('gpac_success_rate', 0) > 0 and 
        result.get('tp_success_rate', 0) > 0 and
        gpac_results[0]['computation_success'] and
        tp_results[0]['computation_success']):
        
        # Get first successful result from each
        gpac_pac = gpac_results[0]['pac_values']
        tp_pac = tp_results[0]['pac_values']
        
        # Reshape to match if needed
        if gpac_pac.dim() == 4 and tp_pac.dim() == 3:
            # gPAC: (batch, channels, pha, amp)
            # Tensorpac: (epochs, pha, amp)
            gpac_pac = gpac_pac.mean(dim=1)  # Average across channels
        
        # Calculate correlation
        flat_gpac = gpac_pac.flatten().numpy()
        flat_tp = tp_pac.flatten().numpy()
        
        correlation = np.corrcoef(flat_gpac, flat_tp)[0, 1] if len(flat_gpac) == len(flat_tp) else None
        mae = np.mean(np.abs(flat_gpac - flat_tp)) if len(flat_gpac) == len(flat_tp) else None
        
        result.update({
            'correlation': correlation,
            'mae': mae
        })
    
    return result


def generate_figures(results_df: pd.DataFrame):
    """
    Generate publication-quality figures from benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
    """
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Figure 1: PAC Calculation - Why parallel execution makes it faster
    # This is more of a schematic figure, not directly from results
    
    # Figure 2: Calculated PAC Similarity with Tensorpac
    plt.figure(figsize=(10, 8))
    
    # Filter for base parameter set
    base_results = results_df[results_df['param_set'] == 'base']
    
    if 'correlation' in base_results.columns and not base_results['correlation'].isna().all():
        correlations = base_results['correlation']
        maes = base_results['mae']
        
        # Create subplot for correlation
        plt.subplot(2, 2, 1)
        plt.bar(['Correlation'], [correlations.mean()])
        plt.ylim(0, 1)
        plt.title('Average Correlation with Tensorpac')
        
        # Create subplot for MAE
        plt.subplot(2, 2, 2)
        plt.bar(['MAE'], [maes.mean()])
        plt.title('Mean Absolute Error')
        
        # TODO: Add more detailed visualizations with heatmaps of PAC values
    else:
        plt.text(0.5, 0.5, "No correlation data available", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure_2_PAC_Similarity.png", dpi=300)
    plt.close()
    
    # Figure 3: Calculation Speed Comparison vs. Tensorpac
    plt.figure(figsize=(12, 10))
    
    # Filter for parameter variations related to dataset size
    size_params = ['batch_size', 'n_channels', 'duration', 'pha_n_bands', 'amp_n_bands']
    size_results = results_df[results_df['varied_param'].isin(size_params)]
    
    if not size_results.empty and 'speedup' in size_results.columns:
        # Group by parameter and create subplots
        for i, param in enumerate(size_params):
            param_results = size_results[size_results['varied_param'] == param]
            
            if not param_results.empty:
                plt.subplot(len(size_params), 1, i+1)
                
                # Sort by parameter value
                param_results = param_results.sort_values('param_value')
                
                # Create bar chart of speedup
                plt.bar(param_results['param_value'].astype(str), param_results['speedup'])
                plt.ylabel('Speedup Factor')
                plt.title(f'Speedup vs. {param}')
                
                # Log scale if values vary widely
                if param_results['speedup'].max() / param_results['speedup'].min() > 10:
                    plt.yscale('log')
    else:
        plt.text(0.5, 0.5, "No speedup data available", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure_3_Speed_Comparison.png", dpi=300)
    plt.close()
    
    # Figure 4: Requirements with parameter sets
    plt.figure(figsize=(15, 15))
    
    # Define metrics to visualize
    metrics = [
        ('gpac_time_mean', 'Computation Time (s)'),
        ('gpac_ram_usage_mean', 'RAM Usage (GiB)'),
        ('gpac_cpu_percent_mean', 'CPU Usage (%)'),
    ]
    
    if 'gpac_gpu_memory_mean' in results_df.columns:
        metrics.append(('gpac_gpu_memory_mean', 'GPU Memory (GiB)'))
    
    # Create a subplot grid
    n_rows = len(metrics)
    n_cols = len(results_df['varied_param'].unique())
    
    plot_idx = 1
    
    # Create a plot for each metric and parameter
    for metric, metric_name in metrics:
        for param in results_df['varied_param'].unique():
            param_results = results_df[results_df['varied_param'] == param]
            
            if not param_results.empty and metric in param_results.columns:
                plt.subplot(n_rows, n_cols, plot_idx)
                
                # Sort by parameter value
                param_results = param_results.sort_values('param_value')
                
                # Create box plot
                plt.errorbar(
                    param_results['param_value'],
                    param_results[metric],
                    yerr=param_results.get(f"{metric.replace('mean', 'std')}", 0),
                    marker='o',
                    linestyle='-'
                )
                
                plt.xlabel(param)
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} vs. {param}')
                
                # Log scale for time and memory if values vary widely
                if metric in ['gpac_time_mean', 'gpac_ram_usage_mean', 'gpac_gpu_memory_mean']:
                    if param_results[metric].max() / max(param_results[metric].min(), 1e-6) > 10:
                        plt.yscale('log')
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure_4_Resource_Requirements.png", dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark gPAC against Tensorpac'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(CONFIG_PATH),
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--results',
        type=str,
        help='Path to existing results CSV file (skips running benchmarks)'
    )
    parser.add_argument(
        '--skip-tensorpac',
        action='store_true',
        help='Skip Tensorpac benchmarks'
    )
    
    return parser.parse_args()


def main():
    """Main function to run benchmarks and generate figures."""
    args = parse_args()
    
    # Check if Tensorpac is available
    global TENSORPAC_AVAILABLE
    if args.skip_tensorpac:
        TENSORPAC_AVAILABLE = False
        print("Tensorpac benchmarks skipped as requested")
    elif not TENSORPAC_AVAILABLE:
        print("Warning: Tensorpac not available. Install with: pip install tensorpac")
    
    # Either load existing results or run benchmarks
    if args.results:
        print(f"Loading existing results from {args.results}")
        results_df = pd.read_csv(args.results)
    else:
        print(f"Running parameter sweep using config: {args.config}")
        results_df = run_parameter_sweep(Path(args.config))
    
    # Generate figures
    print("Generating figures")
    generate_figures(results_df)
    print(f"Figures saved to {OUTPUT_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())