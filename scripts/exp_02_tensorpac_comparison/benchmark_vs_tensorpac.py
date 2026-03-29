#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 12:25:42 (ywatanabe)"
# File: ./scripts/exp_02_tensorpac_comparison/benchmark_vs_tensorpac.py
# ----------------------------------------
import os

import scitex as stx

__FILE__ = os.path.abspath(__file__)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Benchmarks gPAC against Tensorpac for performance comparison
  - Tests different parameter combinations from config/PARAMS.yaml
  - Measures computation time, memory usage, and hardware utilization
  - Exports results as CSV files for further analysis

Dependencies:
  - packages:
    - PyTorch
    - Tensorpac
    - NumPy
    - scitex

IO:
  - input-files:
    - ./data/exp_01/synthetic_pac_signals.pt

  - output-files:
    - ./data/exp_02/benchmark_results.csv
    - ./data/exp_02/resource_usage.csv
"""

"""Imports"""
import argparse
import itertools
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import torch

try:
    import tensorpac
except ImportError:
    print(
        "Warning: Tensorpac not installed. Only gPAC benchmarks will be available."
    )

"""Warnings"""
import warnings

# Ignore specific warnings that could clutter output
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The number of bands in phase and power is unbalanced",
)
warnings.filterwarnings("ignore", category=FutureWarning)

"""Parameters"""
# Will be loaded via stx.io.load_configs() in main()

"""Functions & Classes"""
def setup_experiment_from_params(
    param_set: str,
    param_name: Optional[str] = None,
    param_value: Optional[any] = None,
) -> Dict:
    """
    Set up experiment parameters from config file.

    Args:
        param_set: Parameter set name ('BASELINE', 'VARIATIONS', 'ALL')
        param_name: Optional parameter name to vary
        param_value: Optional parameter value to use

    Returns:
        Dictionary of experiment parameters
    """
    # Get base parameters from config
    base_params = CONFIG.PARAMS[param_set]

    # Create a copy of the base parameters
    params = {k: v for k, v in base_params.items()}

    # Override specific parameter if specified
    if param_name is not None and param_value is not None:
        params[param_name] = param_value

    return params


def load_synthetic_data(
    data_path: str = "./data/exp_01/synthetic_pac_signals.pt",
) -> torch.Tensor:
    """
    Load synthetic PAC signals for benchmarking.

    Args:
        data_path: Path to synthetic data

    Returns:
        Tensor of synthetic signals
    """
    try:
        data = torch.load(data_path)
        signals = data["signals"]
        return signals
    except Exception as e:
        raise RuntimeError(f"Failed to load synthetic data: {e}")


def prepare_signals_for_benchmarking(
    signals: torch.Tensor,
    batch_size: int,
    n_chs: int,
    n_segments: int,
    t_sec: float,
    fs: float,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Prepare signals for benchmarking with both gPAC and Tensorpac.

    Args:
        signals: Input signals
        batch_size: Number of samples in batch
        n_chs: Number of channels
        n_segments: Number of segments
        t_sec: Duration in seconds
        fs: Sampling frequency

    Returns:
        Dictionary containing prepared signals for each framework
    """
    # Calculate sequence length based on duration and sampling rate
    seq_len = int(t_sec * fs)

    # Ensure we have enough data, otherwise repeat
    if signals.shape[0] < batch_size:
        repeat_factor = (batch_size // signals.shape[0]) + 1
        signals = signals.repeat(repeat_factor, 1, 1, 1)

    # Ensure we have enough channels
    if signals.shape[1] < n_chs:
        # Repeat channels
        current_chs = signals.shape[1]
        signals = signals.repeat_interleave(
            (n_chs + current_chs - 1) // current_chs, dim=1
        )

    # Ensure we have enough segments
    if signals.shape[2] < n_segments:
        # Repeat segments
        current_segments = signals.shape[2]
        signals = signals.repeat_interleave(
            (n_segments + current_segments - 1) // current_segments, dim=2
        )

    # Ensure we have enough sequence length
    if signals.shape[3] < seq_len:
        # Pad with zeros or repeat
        current_len = signals.shape[3]
        if seq_len <= 2 * current_len:
            # Pad
            padding = seq_len - current_len
            signals = torch.nn.functional.pad(
                signals, (0, padding), "constant", 0
            )
        else:
            # Repeat and trim
            repeat_factor = (seq_len // current_len) + 1
            signals = signals.repeat(1, 1, 1, repeat_factor)
            signals = signals[:, :, :, :seq_len]
    elif signals.shape[3] > seq_len:
        # Trim
        signals = signals[:, :, :, :seq_len]

    # Get the final batch
    signals = signals[:batch_size, :n_chs, :n_segments, :seq_len]

    # Prepare Tensorpac format (if needed)
    tensorpac_signals = None
    try:
        if "tensorpac" in sys.modules:
            # Tensorpac expects shape: (n_epochs, n_times)
            # Reshape to match: (batch_size * n_chs * n_segments, seq_len)
            tensorpac_signals = signals.reshape(-1, seq_len).cpu().numpy()
    except Exception as e:
        warnings.warn(f"Failed to prepare signals for Tensorpac: {e}")

    return {
        "gpac_signals": signals,
        "tensorpac_signals": tensorpac_signals,
        "batch_size": batch_size,
        "n_chs": n_chs,
        "n_segments": n_segments,
        "seq_len": seq_len,
    }


def benchmark_gpac(
    signals: torch.Tensor,
    params: Dict,
    n_trials: int = 3,
) -> Dict:
    """
    Benchmark gPAC performance.

    Args:
        signals: Input signals
        params: Benchmark parameters
        n_trials: Number of trials to run

    Returns:
        Dictionary of benchmark results
    """
    from gpac._pac import calculate_pac

    # Extract parameters
    batch_size = params["batch_size"]
    n_chs = params["n_chs"]
    n_segments = params["n_segments"]
    t_sec = params["t_sec"]
    fs = params["fs"]
    pha_n_bands = params["pha_n_bands"]
    amp_n_bands = params["amp_n_bands"]
    chunk_size = params["chunk_size"]
    n_perm = params["n_perm"]
    fp16 = params["fp16"]
    no_grad = params["no_grad"]
    in_place = params["in_place"]
    trainable = params["trainable"]
    device = params["device"]

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested but not available. Falling back to CPU."
        )
        device = "cpu"

    device = torch.device(device)
    signals = signals.to(device)

    # Gradient control context
    grad_context = torch.no_grad if no_grad else torch.enable_grad

    # Warmup run
    try:
        with grad_context():
            calculate_pac(
                signal=signals,
                fs=fs,
                pha_n_bands=pha_n_bands,
                amp_n_bands=amp_n_bands,
                n_perm=n_perm,
                trainable=trainable,
                fp16=fp16,
                device=device,
                chunk_size=chunk_size,
            )
    except Exception as e:
        print(f"Warmup run failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "init_time": 0,
            "calc_time": 0,
            "memory_cpu": 0,
            "memory_gpu": 0,
        }

    # Memory usage tracking
    peak_memory_cpu = 0
    peak_memory_gpu = (
        0
        if device.type == "cpu"
        else torch.cuda.max_memory_allocated(device=device)
    )

    # Measure initialization time
    init_times = []
    calc_times = []

    for trial in range(n_trials):
        # Clear cache if using GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)

        # Measure initialization time
        t0 = time.time()
        with grad_context():
            pac_values, pha_freqs, amp_freqs = calculate_pac(
                signal=signals,
                fs=fs,
                pha_n_bands=pha_n_bands,
                amp_n_bands=amp_n_bands,
                n_perm=n_perm,
                trainable=trainable,
                fp16=fp16,
                device=device,
                chunk_size=chunk_size,
            )
        t1 = time.time()

        # Record times
        calc_time = t1 - t0
        calc_times.append(calc_time)

        # Record memory usage
        peak_memory_cpu = max(peak_memory_cpu, get_process_memory_usage())
        if device.type == "cuda":
            peak_memory_gpu = max(
                peak_memory_gpu, torch.cuda.max_memory_allocated(device=device)
            )

    # Calculate average times
    avg_calc_time = np.mean(calc_times)
    std_calc_time = np.std(calc_times)

    # Normalize memory by batch size for fair comparison
    normalized_memory_cpu = peak_memory_cpu / batch_size
    normalized_memory_gpu = (
        peak_memory_gpu / batch_size if device.type == "cuda" else 0
    )

    return {
        "success": True,
        "calc_time": avg_calc_time,
        "calc_time_std": std_calc_time,
        "memory_cpu": normalized_memory_cpu,  # in MB
        "memory_gpu": normalized_memory_gpu,  # in bytes
        "pha_freqs": pha_freqs,
        "amp_freqs": amp_freqs,
    }


def benchmark_tensorpac(
    signals: np.ndarray,
    params: Dict,
    n_trials: int = 3,
) -> Dict:
    """
    Benchmark Tensorpac performance.

    Args:
        signals: Input signals in Tensorpac format
        params: Benchmark parameters
        n_trials: Number of trials to run

    Returns:
        Dictionary of benchmark results
    """
    if "tensorpac" not in sys.modules or signals is None:
        return {"success": False, "error": "Tensorpac not available"}

    try:
        from tensorpac import Pac
    except ImportError:
        return {"success": False, "error": "Failed to import Tensorpac"}

    # Extract parameters
    batch_size = params["batch_size"]
    n_chs = params["n_chs"]
    n_segments = params["n_segments"]
    fs = params["fs"]
    pha_n_bands = params["pha_n_bands"]
    amp_n_bands = params["amp_n_bands"]
    n_perm = params["n_perm"]
    use_threads = params.get("use_threads", False)

    # Create linear phase and amplitude frequency ranges
    # (Tensorpac requires explicit boundaries)
    pha_freqs = np.linspace(2, 20, pha_n_bands + 1)  # Add 1 for boundaries
    amp_freqs = np.linspace(50, 200, amp_n_bands + 1)  # Add 1 for boundaries

    # Convert to bands format required by Tensorpac
    p_bands = np.vstack((pha_freqs[:-1], pha_freqs[1:])).T
    a_bands = np.vstack((amp_freqs[:-1], amp_freqs[1:])).T

    # Set up PAC object
    p = Pac(
        idpac=(1, 2, 3),
        f_pha=p_bands,
        f_amp=a_bands,
        dcomplex="wavelet",
        n_jobs=-1 if use_threads else 1,
    )

    # Warmup run
    try:
        p.filterphase = p_bands[0]
        p.filteramp = a_bands[0]
        _ = p.filterfilt(signals)
    except Exception as e:
        print(f"Tensorpac warmup failed: {e}")
        return {"success": False, "error": str(e)}

    # Memory usage tracking
    peak_memory_cpu = 0

    # Measure calculation time
    calc_times = []

    for trial in range(n_trials):
        # Measure calculation time
        t0 = time.time()

        # Calculate PAC
        pac = p.filterfit(signals, n_perm=n_perm if n_perm is not None else 0)

        t1 = time.time()

        # Record times
        calc_time = t1 - t0
        calc_times.append(calc_time)

        # Record memory usage
        peak_memory_cpu = max(peak_memory_cpu, get_process_memory_usage())

    # Calculate average times
    avg_calc_time = np.mean(calc_times)
    std_calc_time = np.std(calc_times)

    # Normalize memory by batch size for fair comparison
    normalized_memory_cpu = peak_memory_cpu / batch_size

    return {
        "success": True,
        "calc_time": avg_calc_time,
        "calc_time_std": std_calc_time,
        "memory_cpu": normalized_memory_cpu,  # in MB
        "pha_freqs": p_bands,
        "amp_freqs": a_bands,
    }


def get_process_memory_usage() -> float:
    """Get memory usage of current process in MB."""
    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Convert to MB
    memory_usage_mb = memory_info.rss / (1024 * 1024)

    return memory_usage_mb


def get_system_resource_usage(device: str) -> Dict:
    """Get system-wide resource usage."""
    import psutil

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # RAM usage
    memory = psutil.virtual_memory()
    ram_usage_gb = memory.used / (1024**3)

    # GPU usage (if available)
    gpu_usage = None
    gpu_memory_usage = None

    if device == "cuda" and torch.cuda.is_available():
        try:
            # Try to get GPU stats via pynvml
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = util.gpu

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_usage = mem_info.used / (1024**3)  # GB

            pynvml.nvmlShutdown()
        except:
            # Fallback: attempt to parse nvidia-smi output (user-confirmed fallback)
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                )
                gpu_stats = result.stdout.strip().split(",")
                gpu_usage = float(gpu_stats[0])
                gpu_memory_usage = float(gpu_stats[1]) / 1024  # Convert to GB
            except:
                gpu_usage = None
                gpu_memory_usage = None

    return {
        "cpu_percent": cpu_percent,
        "ram_usage_gb": ram_usage_gb,
        "gpu_usage": gpu_usage,
        "gpu_memory_gb": gpu_memory_usage,
    }


def run_benchmarks(
    params: Dict,
    signals: Dict,
    results_dir: str,
    n_trials: int = 3,
) -> Dict:
    """
    Run benchmarks for both gPAC and Tensorpac.

    Args:
        params: Benchmark parameters
        signals: Dictionary of prepared signals
        results_dir: Directory to save results
        n_trials: Number of trials to run

    Returns:
        Dictionary of benchmark results
    """
    # Start time for this benchmark run
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Get system resource usage before benchmarks
    device = params.get("device", "cpu")
    baseline_resources = get_system_resource_usage(device)

    # Run gPAC benchmark
    print(f"Running gPAC benchmark with parameters: {params}")
    gpac_signals = signals["gpac_signals"]
    gpac_results = benchmark_gpac(gpac_signals, params, n_trials)

    # Get system resource usage after gPAC
    gpac_resources = get_system_resource_usage(device)

    # Run Tensorpac benchmark if available
    tensorpac_signals = signals["tensorpac_signals"]
    if "tensorpac" in sys.modules and tensorpac_signals is not None:
        print("Running Tensorpac benchmark...")
        tensorpac_results = benchmark_tensorpac(
            tensorpac_signals, params, n_trials
        )

        # Get system resource usage after Tensorpac
        tensorpac_resources = get_system_resource_usage(device)
    else:
        print("Skipping Tensorpac benchmark (not available)")
        tensorpac_results = {
            "success": False,
            "error": "Tensorpac not available",
        }
        tensorpac_resources = baseline_resources

    # Combine results
    benchmark_results = {
        "timestamp": start_time,
        "params": params,
        "gpac": gpac_results,
        "tensorpac": tensorpac_results,
        "resources": {
            "baseline": baseline_resources,
            "gpac": gpac_resources,
            "tensorpac": tensorpac_resources,
        },
    }

    # Use stx.io.save() with proper relative path format
    rel_path = f"./results/benchmark/benchmark_{start_time}.json"
    stx.io.save(benchmark_results, rel_path)

    return benchmark_results


def run_parameter_sweep(
    base_params: Dict,
    param_name: str,
    param_values: List,
    signals: Dict,
    results_dir: str,
    n_trials: int = 3,
) -> List[Dict]:
    """
    Run benchmarks across different values of a parameter.

    Args:
        base_params: Base parameters for the benchmark
        param_name: Name of the parameter to sweep
        param_values: List of values to test for the parameter
        signals: Dictionary of prepared signals
        results_dir: Directory to save results
        n_trials: Number of trials per configuration

    Returns:
        List of benchmark results
    """
    results = []

    for value in param_values:
        # Create a copy of the base parameters
        params = {k: v for k, v in base_params.items()}
        # Override the parameter being tested
        params[param_name] = value

        print(f"\n{'-' * 80}")
        print(f"Testing {param_name} = {value}")
        print(f"{'-' * 80}\n")

        # Run the benchmark
        result = run_benchmarks(params, signals, results_dir, n_trials)
        results.append(result)

    return results


@stx.session
def main(args):
    """Main function for benchmarking gPAC against Tensorpac."""
    global CONFIG

    CONFIG = stx.io.load_configs()

    # Save initial info
    stx.io.save(
        {"timestamp": str(datetime.now())},
        "./results/benchmark/results_info.json",
    )

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load signals
    print("Loading synthetic data...")
    if os.path.exists(args.data_path):
        signals_data = load_synthetic_data(args.data_path)
    else:
        signals_data = torch.randn(10, 2, 1, 2000)  # Fallback to random data
        print(f"Using random data (could not find {args.data_path})")

    # Get experiment parameters
    param_set = args.param_set
    base_params = setup_experiment_from_params(param_set)
    print(f"Using parameter set: {param_set}")

    # Prepare signals
    print("Preparing signals...")
    signals = prepare_signals_for_benchmarking(
        signals_data,
        base_params["batch_size"],
        base_params["n_chs"],
        base_params["n_segments"],
        base_params["t_sec"],
        base_params["fs"],
    )

    # Run benchmarks
    if args.param_sweep:
        # Parameter sweep mode
        param_name = args.param_name

        if param_name is None:
            print("Error: param_name must be specified for parameter sweep")
            return 1

        param_values = CONFIG.PARAMS["VARIATIONS"].get(param_name)

        if param_values is None:
            print(f"Error: No variations found for parameter {param_name}")
            return 1

        print(f"Running parameter sweep for {param_name}: {param_values}")

        # Run parameter sweep
        sweep_results = run_parameter_sweep(
            base_params,
            param_name,
            param_values,
            signals,
            args.output_dir,
            args.n_trials,
        )

        # Compile sweep results
        combined_results = {
            "param_name": param_name,
            "param_values": param_values,
            "results": sweep_results,
        }

        # Save combined results using stx.io.save
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rel_path = f"./results/benchmark/sweep_{param_name}_{timestamp}.json"
        stx.io.save(combined_results, rel_path)

    elif args.param_grid:
        # Grid search mode
        param_names = args.param_names.split(",") if args.param_names else []

        if not param_names:
            print("Error: param_names must be specified for grid search")
            return 1

        # Get parameter values for each parameter
        param_grid = {}
        for name in param_names:
            values = CONFIG.PARAMS["VARIATIONS"].get(name)
            if values is None:
                print(
                    f"Warning: No variations found for parameter {name}, using baseline"
                )
                values = [base_params.get(name)]
            param_grid[name] = values

        print(f"Running grid search for parameters: {param_grid}")

        # Generate all parameter combinations
        param_combinations = list(
            itertools.product(*[param_grid[name] for name in param_names])
        )

        # Run all combinations
        grid_results = []

        for combo in param_combinations:
            # Create parameter dict for this combination
            params = {k: v for k, v in base_params.items()}
            for i, name in enumerate(param_names):
                params[name] = combo[i]

            print(f"\n{'-' * 80}")
            print(f"Testing combination: {dict(zip(param_names, combo))}")
            print(f"{'-' * 80}\n")

            # Run benchmark
            result = run_benchmarks(
                params, signals, args.output_dir, args.n_trials
            )
            grid_results.append((combo, result))

        # Compile grid results
        combined_results = {
            "param_names": param_names,
            "param_grid": param_grid,
            "results": [
                (dict(zip(param_names, combo)), result)
                for combo, result in grid_results
            ],
        }

        # Save combined results using stx.io.save
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        grid_name = '_'.join(param_names)
        rel_path = f"./results/benchmark/grid_{grid_name}_{timestamp}.json"
        stx.io.save(combined_results, rel_path)

    else:
        # Single benchmark mode
        print("Running single benchmark with baseline parameters")
        run_benchmarks(base_params, signals, args.output_dir, args.n_trials)

    print("\nBenchmarking complete!")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark gPAC against Tensorpac"
    )

    # Basic options
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/exp_01/synthetic_pac_signals.pt",
        help="Path to synthetic data",
    )
    parser.add_argument(
        "--param_set",
        type=str,
        default="BASELINE",
        choices=["BASELINE", "VARIATIONS", "ALL"],
        help="Parameter set to use (from config/PARAMS.yaml)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of trials to run for each benchmark",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./scripts/exp_02_tensorpac_comparison/results/benchmark",
        help="Directory to save benchmark results",
    )

    # Parameter sweep options
    parser.add_argument(
        "--param_sweep", action="store_true", help="Run parameter sweep"
    )
    parser.add_argument(
        "--param_name",
        type=str,
        default=None,
        help="Parameter to sweep (for --param_sweep)",
    )

    # Grid search options
    parser.add_argument(
        "--param_grid",
        action="store_true",
        help="Run grid search on multiple parameters",
    )
    parser.add_argument(
        "--param_names",
        type=str,
        default=None,
        help="Comma-separated list of parameters for grid search (for --param_grid)",
    )
    parser.add_argument(
        "--param_grid_file",
        type=str,
        default=None,
        help="JSON file with parameter grid configuration",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# EOF
