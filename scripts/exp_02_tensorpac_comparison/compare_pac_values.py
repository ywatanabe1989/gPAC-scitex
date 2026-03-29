#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 12:45:23 (ywatanabe)"
# File: ./scripts/exp_02_tensorpac_comparison/compare_pac_values.py
# ----------------------------------------
import os

import scitex as stx

__FILE__ = os.path.abspath(__file__)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Compares PAC values calculated by gPAC and Tensorpac
  - Assesses numerical agreement between the two implementations
  - Generates quantitative metrics for comparison (correlation, RMSE, etc.)
  - Creates visualizations showing the relationship between values

Dependencies:
  - packages:
    - PyTorch
    - Tensorpac
    - NumPy
    - Matplotlib
    - scipy
    - scitex

IO:
  - input-files:
    - ./data/exp_01/synthetic_pac_signals.pt

  - output-files:
    - ./data/exp_02/pac_value_comparison.csv
    - ./data/exp_02/pac_value_comparison.png
    - ./data/exp_02/correlation_metrics.json
"""

"""Imports"""
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon

try:
    import tensorpac
except ImportError:
    print("Warning: Tensorpac not installed. Comparison not possible.")

"""Warnings"""
import warnings
# Ignore specific warnings that could clutter output
warnings.filterwarnings("ignore", category=UserWarning,
                       message="The number of bands in phase and power is unbalanced")
warnings.filterwarnings("ignore", category=FutureWarning)

"""Parameters"""
# Will be loaded via stx.io.load_configs() in main()

"""Functions & Classes"""
def load_synthetic_data(
    data_path: str = "./data/exp_01/synthetic_pac_signals.pt"
) -> torch.Tensor:
    """
    Load synthetic PAC signals for comparison.

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

def prepare_signals(
    signals: torch.Tensor,
    batch_size: int = 4,
    n_chs: int = 2,
    n_segments: int = 1,
    seq_len: int = 1000,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Prepare signals for both gPAC and Tensorpac calculations.

    Args:
        signals: Input signals
        batch_size: Number of samples in batch
        n_chs: Number of channels
        n_segments: Number of segments
        seq_len: Sequence length

    Returns:
        Dictionary containing prepared signals for each framework
    """
    # Ensure we have enough data, otherwise repeat
    if signals.shape[0] < batch_size:
        repeat_factor = (batch_size // signals.shape[0]) + 1
        signals = signals.repeat(repeat_factor, 1, 1, 1)

    # Trim to desired sizes
    signals = signals[:batch_size, :n_chs, :n_segments, :seq_len]

    # Prepare Tensorpac format (if needed)
    tensorpac_signals = None
    try:
        if 'tensorpac' in sys.modules:
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

def compute_gpac_values(
    signals: torch.Tensor,
    fs: float = 1000.0,
    pha_start_hz: float = 2.0,
    pha_end_hz: float = 20.0,
    pha_n_bands: int = 10,
    amp_start_hz: float = 50.0,
    amp_end_hz: float = 200.0,
    amp_n_bands: int = 10,
    device: str = "cpu",
    n_perm: int = 200,
    return_dist: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute PAC values using gPAC.

    Args:
        signals: Input signals
        fs: Sampling frequency
        pha_start_hz: Start frequency for phase bands
        pha_end_hz: End frequency for phase bands
        pha_n_bands: Number of phase bands
        amp_start_hz: Start frequency for amplitude bands
        amp_end_hz: End frequency for amplitude bands
        amp_n_bands: Number of amplitude bands
        device: Device to use ('cpu' or 'cuda')
        n_perm: Number of permutations for surrogate testing
        return_dist: Whether to return surrogate distributions

    Returns:
        When return_dist=False:
            Tuple of (PAC values, phase frequencies, amplitude frequencies)
        When return_dist=True:
            Tuple of (PAC values, surrogate distributions, phase frequencies, amplitude frequencies)
    """
    from src.gpac._pac import calculate_pac

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    device = torch.device(device)
    signals = signals.to(device)

    # Calculate PAC values
    with torch.no_grad():
        # Call calculate_pac with return_dist option
        if return_dist:
            pac_values, surrogate_dist, pha_freqs, amp_freqs = calculate_pac(
                signal=signals,
                fs=fs,
                pha_start_hz=pha_start_hz,
                pha_end_hz=pha_end_hz,
                pha_n_bands=pha_n_bands,
                amp_start_hz=amp_start_hz,
                amp_end_hz=amp_end_hz,
                amp_n_bands=amp_n_bands,
                device=device,
                n_perm=n_perm,
                return_dist=True,
            )
        else:
            pac_values, pha_freqs, amp_freqs = calculate_pac(
                signal=signals,
                fs=fs,
                pha_start_hz=pha_start_hz,
                pha_end_hz=pha_end_hz,
                pha_n_bands=pha_n_bands,
                amp_start_hz=amp_start_hz,
                amp_end_hz=amp_end_hz,
                amp_n_bands=amp_n_bands,
                device=device,
                n_perm=n_perm,
                return_dist=False,
            )

    # Convert to NumPy arrays for comparison
    if isinstance(pac_values, torch.Tensor):
        pac_values = pac_values.cpu().numpy()

    if isinstance(pha_freqs, torch.Tensor):
        pha_freqs = pha_freqs.cpu().numpy()

    if isinstance(amp_freqs, torch.Tensor):
        amp_freqs = amp_freqs.cpu().numpy()

    # Convert surrogate distributions if applicable
    if return_dist and 'surrogate_dist' in locals():
        if isinstance(surrogate_dist, torch.Tensor):
            surrogate_dist = surrogate_dist.cpu().numpy()
        return pac_values, surrogate_dist, pha_freqs, amp_freqs

    return pac_values, pha_freqs, amp_freqs

def compute_tensorpac_values(
    signals: np.ndarray,
    fs: float = 1000.0,
    pha_start_hz: float = 2.0,
    pha_end_hz: float = 20.0,
    pha_n_bands: int = 10,
    amp_start_hz: float = 50.0,
    amp_end_hz: float = 200.0,
    amp_n_bands: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PAC values using Tensorpac.

    Args:
        signals: Input signals in Tensorpac format
        fs: Sampling frequency
        pha_start_hz: Start frequency for phase bands
        pha_end_hz: End frequency for phase bands
        pha_n_bands: Number of phase bands
        amp_start_hz: Start frequency for amplitude bands
        amp_end_hz: End frequency for amplitude bands
        amp_n_bands: Number of amplitude bands

    Returns:
        Tuple of (PAC values, phase frequencies, amplitude frequencies)
    """
    if 'tensorpac' not in sys.modules or signals is None:
        raise ImportError("Tensorpac not available")

    try:
        from tensorpac import Pac
    except ImportError:
        raise ImportError("Failed to import Tensorpac")

    # Create linear phase and amplitude frequency ranges
    # (Tensorpac requires explicit boundaries)
    pha_freqs_array = np.linspace(pha_start_hz, pha_end_hz, pha_n_bands + 1)
    amp_freqs_array = np.linspace(amp_start_hz, amp_end_hz, amp_n_bands + 1)

    # Convert to bands format required by Tensorpac
    p_bands = np.vstack((pha_freqs_array[:-1], pha_freqs_array[1:])).T
    a_bands = np.vstack((amp_freqs_array[:-1], amp_freqs_array[1:])).T

    # Compute midpoints for comparison
    pha_freqs = np.mean(p_bands, axis=1)
    amp_freqs = np.mean(a_bands, axis=1)

    # Set up PAC object
    # Use method (1, 2, 3) which is mean vector length with Hilbert transform
    # to match as closely as possible with gPAC's implementation
    p = Pac(idpac=(1, 2, 3), f_pha=p_bands, f_amp=a_bands, dcomplex='wavelet')

    # Tensorpac requires separate phase/amplitude filtering
    # First, filter the data using Tensorpac's filter method
    # The API has different versions
    try:
        # Try different versions of the API
        try:
            # Version with just ftype parameter
            xpha = p.filter(signals, ftype='phase')
            xamp = p.filter(signals, ftype='amplitude')
        except TypeError:
            # Version with x and ftype parameters
            try:
                xpha = p.filter(x=signals, ftype='phase')
                xamp = p.filter(x=signals, ftype='amplitude')
            except TypeError:
                # Version with x, sf, and ftype parameters
                xpha = p.filter(x=signals, sf=fs, ftype='phase')
                xamp = p.filter(x=signals, sf=fs, ftype='amplitude')
    except Exception as e:
        raise RuntimeError(f"Failed to filter signals with Tensorpac: {e}")

    # Calculate PAC with the filtered signals
    # Different versions of Tensorpac have different APIs
    try:
        # First try direct computation
        pac_values = p.pac(xpha, xamp)
    except AttributeError:
        try:
            # Try computing method
            pac_values = p.compute(xpha, xamp)
        except Exception as e:
            raise RuntimeError(f"Failed to compute PAC with Tensorpac: {e}")

    # Average across epochs (equivalent to averaging across segments in gPAC)
    # This will result in shape (n_pha, n_amp) after averaging
    pac_values = np.mean(pac_values, axis=0)

    return pac_values, pha_freqs, amp_freqs

def compare_pac_values(
    gpac_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tensorpac_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict:
    """
    Compare PAC values from gPAC and Tensorpac.

    Args:
        gpac_data: Tuple of (PAC values, phase frequencies, amplitude frequencies) from gPAC
        tensorpac_data: Tuple of (PAC values, phase frequencies, amplitude frequencies) from Tensorpac

    Returns:
        Dictionary of comparison metrics
    """
    gpac_pac, gpac_pha, gpac_amp = gpac_data
    tensorpac_pac, tensorpac_pha, tensorpac_amp = tensorpac_data

    # Validate frequency arrays match (approximately)
    freq_match_pha = np.allclose(gpac_pha, tensorpac_pha, rtol=0.1, atol=1.0)
    freq_match_amp = np.allclose(gpac_amp, tensorpac_amp, rtol=0.1, atol=1.0)

    if not (freq_match_pha and freq_match_amp):
        print("Warning: Frequency bands don't match between gPAC and Tensorpac")
        print(f"Phase bands - gPAC: {gpac_pha[:5]}..., Tensorpac: {tensorpac_pha[:5]}...")
        print(f"Amplitude bands - gPAC: {gpac_amp[:5]}..., Tensorpac: {tensorpac_amp[:5]}...")

    # Ensure PAC values are properly shaped for comparison
    # If gPAC has batch/channel dims, average over them
    if gpac_pac.ndim > 2:
        if gpac_pac.ndim == 4:  # (batch, channel, pha, amp)
            gpac_pac = gpac_pac.mean(axis=(0, 1))
        elif gpac_pac.ndim == 3:  # (batch, pha, amp) or (channel, pha, amp)
            gpac_pac = gpac_pac.mean(axis=0)

    # Ensure Tensorpac values are properly shaped
    if tensorpac_pac.ndim > 2:
        tensorpac_pac = tensorpac_pac.mean(axis=tuple(range(tensorpac_pac.ndim - 2)))

    # Verify shapes after processing
    if gpac_pac.shape != tensorpac_pac.shape:
        print(f"Warning: PAC values have different shapes after processing: gPAC {gpac_pac.shape}, Tensorpac {tensorpac_pac.shape}")
        min_pha = min(gpac_pac.shape[0], tensorpac_pac.shape[0])
        min_amp = min(gpac_pac.shape[1], tensorpac_pac.shape[1])
        gpac_pac = gpac_pac[:min_pha, :min_amp]
        tensorpac_pac = tensorpac_pac[:min_pha, :min_amp]

    # Normalize PAC values for comparison
    # This is important because gPAC and Tensorpac may use different scales
    gpac_pac_norm = gpac_pac / np.max(gpac_pac) if np.max(gpac_pac) > 0 else gpac_pac
    tensorpac_pac_norm = tensorpac_pac / np.max(tensorpac_pac) if np.max(tensorpac_pac) > 0 else tensorpac_pac

    # Calculate differences
    abs_diff = np.abs(gpac_pac_norm - tensorpac_pac_norm)
    rel_diff = abs_diff / (np.maximum(np.abs(gpac_pac_norm), np.abs(tensorpac_pac_norm)) + 1e-10)

    # Calculate similarity metrics
    pearson_r, pearson_p = stats.pearsonr(gpac_pac_norm.flatten(), tensorpac_pac_norm.flatten())
    spearman_r, spearman_p = stats.spearmanr(gpac_pac_norm.flatten(), tensorpac_pac_norm.flatten())

    # Calculate RMSE
    rmse = np.sqrt(np.mean((gpac_pac_norm - tensorpac_pac_norm) ** 2))

    # Calculate cosine similarity
    cos_sim = np.dot(gpac_pac_norm.flatten(), tensorpac_pac_norm.flatten()) / (
        np.linalg.norm(gpac_pac_norm.flatten()) * np.linalg.norm(tensorpac_pac_norm.flatten())
    )

    # Calculate Jensen-Shannon divergence
    # First, ensure PAC values are non-negative (they should be for PAC)
    gpac_pac_prob = np.maximum(gpac_pac_norm, 0)
    tensorpac_pac_prob = np.maximum(tensorpac_pac_norm, 0)

    # Normalize to create probability distributions
    gpac_pac_prob = gpac_pac_prob / np.sum(gpac_pac_prob) if np.sum(gpac_pac_prob) > 0 else gpac_pac_prob
    tensorpac_pac_prob = tensorpac_pac_prob / np.sum(tensorpac_pac_prob) if np.sum(tensorpac_pac_prob) > 0 else tensorpac_pac_prob

    # Calculate JS divergence
    js_div = jensenshannon(gpac_pac_prob.flatten(), tensorpac_pac_prob.flatten())

    # Calculate rank correlation of peak PAC values
    # Find indices of top N values in each
    n_top = min(10, gpac_pac.size)
    gpac_top_idx = np.argsort(gpac_pac.flatten())[-n_top:]
    tensorpac_top_idx = np.argsort(tensorpac_pac.flatten())[-n_top:]

    # Convert flat indices to 2D indices
    pha_dim, amp_dim = gpac_pac.shape
    gpac_top_pha = [idx // amp_dim for idx in gpac_top_idx]
    gpac_top_amp = [idx % amp_dim for idx in gpac_top_idx]

    tensorpac_top_pha = [idx // amp_dim for idx in tensorpac_top_idx]
    tensorpac_top_amp = [idx % amp_dim for idx in tensorpac_top_idx]

    # Calculate overlap ratio of top indices
    gpac_top_set = set(zip(gpac_top_pha, gpac_top_amp))
    tensorpac_top_set = set(zip(tensorpac_top_pha, tensorpac_top_amp))
    overlap = len(gpac_top_set.intersection(tensorpac_top_set))
    overlap_ratio = overlap / n_top

    # Return all metrics
    metrics = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "rmse": float(rmse),
        "cosine_similarity": float(cos_sim),
        "jensen_shannon_divergence": float(js_div),
        "top_peak_overlap_ratio": float(overlap_ratio),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
    }

    return {
        "metrics": metrics,
        "gpac_pac": gpac_pac,
        "tensorpac_pac": tensorpac_pac,
        "gpac_pac_norm": gpac_pac_norm,
        "tensorpac_pac_norm": tensorpac_pac_norm,
        "pha_freqs": gpac_pha,
        "amp_freqs": gpac_amp,
    }

def visualize_comparison(
    comparison_data: Dict,
    title: str = "PAC Value Comparison: gPAC vs Tensorpac",
) -> plt.Figure:
    """
    Visualize PAC value comparison.

    Args:
        comparison_data: Dictionary of comparison data
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    # Extract data
    gpac_pac = comparison_data["gpac_pac_norm"]
    tensorpac_pac = comparison_data["tensorpac_pac_norm"]
    pha_freqs = comparison_data["pha_freqs"]
    amp_freqs = comparison_data["amp_freqs"]

    # Extract metrics
    metrics = comparison_data["metrics"]
    pearson_r = metrics["pearson_r"]
    rmse = metrics["rmse"]
    cos_sim = metrics["cosine_similarity"]
    js_div = metrics["jensen_shannon_divergence"]

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    # Plot gPAC heatmap
    ax1 = plt.subplot(grid[0, 0])
    im1 = ax1.pcolormesh(pha_freqs, amp_freqs, gpac_pac.T, cmap="viridis", shading="auto")
    ax1.set_title("gPAC Values")
    ax1.set_xlabel("Phase Frequency (Hz)")
    ax1.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im1, ax=ax1, label="Normalized PAC Value")

    # Plot Tensorpac heatmap
    ax2 = plt.subplot(grid[0, 1])
    im2 = ax2.pcolormesh(pha_freqs, amp_freqs, tensorpac_pac.T, cmap="viridis", shading="auto")
    ax2.set_title("Tensorpac Values")
    ax2.set_xlabel("Phase Frequency (Hz)")
    ax2.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im2, ax=ax2, label="Normalized PAC Value")

    # Plot absolute difference heatmap
    abs_diff = np.abs(gpac_pac - tensorpac_pac)
    ax3 = plt.subplot(grid[0, 2])
    im3 = ax3.pcolormesh(pha_freqs, amp_freqs, abs_diff.T, cmap="OrRd", shading="auto")
    ax3.set_title("Absolute Difference")
    ax3.set_xlabel("Phase Frequency (Hz)")
    ax3.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im3, ax=ax3, label="Difference")

    # Scatter plot comparison
    ax4 = plt.subplot(grid[1, 0:2])
    ax4.scatter(gpac_pac.flatten(), tensorpac_pac.flatten(), alpha=0.5)
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.3)  # Unity line
    ax4.set_title("gPAC vs Tensorpac Values")
    ax4.set_xlabel("gPAC PAC Value")
    ax4.set_ylabel("Tensorpac PAC Value")

    # Add text metrics on the scatter plot
    metrics_text = (
        f"Pearson r: {pearson_r:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"Cosine Similarity: {cos_sim:.4f}\n"
        f"Jensen-Shannon Div: {js_div:.4f}"
    )
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

    # Plot peak locations overlay
    ax5 = plt.subplot(grid[1, 2])
    # Create mask for top values
    n_top = min(5, gpac_pac.size // 10)
    gpac_mask = np.zeros_like(gpac_pac)
    tensorpac_mask = np.zeros_like(tensorpac_pac)

    # Flatten, get indices of top values, then unflatten
    gpac_flat_idx = np.argsort(gpac_pac.flatten())[-n_top:]
    tensorpac_flat_idx = np.argsort(tensorpac_pac.flatten())[-n_top:]

    for idx in gpac_flat_idx:
        i, j = np.unravel_index(idx, gpac_pac.shape)
        gpac_mask[i, j] = 1

    for idx in tensorpac_flat_idx:
        i, j = np.unravel_index(idx, tensorpac_pac.shape)
        tensorpac_mask[i, j] = 1

    # Create RGB image for overlay
    # Red: gPAC only, Green: Tensorpac only, Yellow: Both
    overlay = np.zeros((*gpac_pac.shape, 3))
    overlay[:, :, 0] = gpac_mask  # Red channel: gPAC
    overlay[:, :, 1] = tensorpac_mask  # Green channel: Tensorpac

    # Plot the overlay
    ax5.pcolormesh(pha_freqs, amp_freqs, overlay.transpose(1, 0, 2), shading="auto")
    ax5.set_title("Peak PAC Locations")
    ax5.set_xlabel("Phase Frequency (Hz)")
    ax5.set_ylabel("Amplitude Frequency (Hz)")

    # Add legend for the overlay plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='w', label='gPAC Peaks'),
        Patch(facecolor='green', edgecolor='w', label='Tensorpac Peaks'),
        Patch(facecolor='yellow', edgecolor='w', label='Overlapping Peaks')
    ]
    ax5.legend(handles=legend_elements, loc='upper right')

    # Set main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def run_comparison(args):
    """
    Run PAC comparison between gPAC and Tensorpac.

    Args:
        args: Command line arguments

    Returns:
        Dictionary of comparison results
    """
    # Load signals
    print("Loading synthetic data...")
    if os.path.exists(args.data_path):
        signals_data = load_synthetic_data(args.data_path)
    else:
        print(f"Warning: Could not find {args.data_path}")
        print("Creating random signals for demonstration.")
        signals_data = torch.randn(10, 2, 1, 2000)

    # Prepare signals for both frameworks
    print("Preparing signals...")
    signals = prepare_signals(
        signals_data,
        batch_size=args.batch_size,
        n_chs=args.n_chs,
        n_segments=args.n_segments,
        seq_len=args.seq_len,
    )

    # Set frequency parameters
    pha_start_hz = args.pha_start_hz
    pha_end_hz = args.pha_end_hz
    pha_n_bands = args.pha_n_bands
    amp_start_hz = args.amp_start_hz
    amp_end_hz = args.amp_end_hz
    amp_n_bands = args.amp_n_bands

    # Compute PAC values with gPAC
    print("Computing PAC values with gPAC...")
    if args.return_dist:
        print("Including surrogate distributions in output...")
        gpac_result = compute_gpac_values(
            signals["gpac_signals"],
            fs=args.fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            device=args.device,
            n_perm=args.n_perm,
            return_dist=True,
        )
        gpac_values, gpac_surrogate_dist, gpac_pha, gpac_amp = gpac_result
    else:
        gpac_values, gpac_pha, gpac_amp = compute_gpac_values(
            signals["gpac_signals"],
            fs=args.fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            device=args.device,
            n_perm=args.n_perm,
            return_dist=False,
        )
        gpac_surrogate_dist = None

    # Compute PAC values with Tensorpac
    print("Computing PAC values with Tensorpac...")
    tensorpac_values, tensorpac_pha, tensorpac_amp = compute_tensorpac_values(
        signals["tensorpac_signals"],
        fs=args.fs,
        pha_start_hz=pha_start_hz,
        pha_end_hz=pha_end_hz,
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_start_hz,
        amp_end_hz=amp_end_hz,
        amp_n_bands=amp_n_bands,
    )

    # Compare PAC values
    print("Comparing PAC values...")
    comparison_data = compare_pac_values(
        (gpac_values, gpac_pha, gpac_amp),
        (tensorpac_values, tensorpac_pha, tensorpac_amp),
    )

    # Add surrogate distribution if available
    if args.return_dist and gpac_surrogate_dist is not None:
        comparison_data['gpac_surrogate_dist'] = gpac_surrogate_dist

    return comparison_data

@stx.session
def main(args):
    """Main function for comparing gPAC and Tensorpac PAC values."""
    # Check if Tensorpac is available
    if 'tensorpac' not in sys.modules:
        print("Error: Tensorpac is not available. Comparison can't be performed.")
        print("Please install Tensorpac: pip install tensorpac")
        return 1

    # Run comparison for specified number of trials
    print(f"Running comparison for {args.n_trials} trials...")
    all_metrics = []
    all_surrogate_dists = []

    for trial in range(args.n_trials):
        print(f"\nTrial {trial + 1}/{args.n_trials}")
        comparison_data = run_comparison(args)
        all_metrics.append(comparison_data["metrics"])

        # Store surrogate distributions if available
        if args.return_dist and 'gpac_surrogate_dist' in comparison_data:
            all_surrogate_dists.append(comparison_data['gpac_surrogate_dist'])

    # Average metrics across trials
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    # Create visualization from last trial
    print("Creating visualization...")
    fig = visualize_comparison(comparison_data)

    import pandas as pd

    # Save visualization
    stx.io.save(
        fig,
        "./results/exp_02/pac_values/pac_value_comparison.png",
    )

    # Save metrics
    stx.io.save(
        avg_metrics,
        "./results/exp_02/pac_values/correlation_metrics.json",
    )

    # Convert PAC matrices to DataFrame
    pha_freqs = comparison_data["pha_freqs"]
    amp_freqs = comparison_data["amp_freqs"]
    gpac_pac = comparison_data["gpac_pac"]
    tensorpac_pac = comparison_data["tensorpac_pac"]

    data = []
    for i, pha in enumerate(pha_freqs):
        for j, amp in enumerate(amp_freqs):
            data.append({
                "pha_freq": pha,
                "amp_freq": amp,
                "gpac_pac": gpac_pac[i, j],
                "tensorpac_pac": tensorpac_pac[i, j],
                "abs_diff": np.abs(gpac_pac[i, j] - tensorpac_pac[i, j]),
                "rel_diff": np.abs(gpac_pac[i, j] - tensorpac_pac[i, j]) / (max(abs(gpac_pac[i, j]), abs(tensorpac_pac[i, j])) + 1e-10),
            })

    df = pd.DataFrame(data)
    stx.io.save(
        df,
        "./results/exp_02/pac_values/pac_value_comparison.csv",
    )

    # Save all raw data for future reference
    full_results = {
        'gpac_pac': gpac_pac,
        'tensorpac_pac': tensorpac_pac,
        'gpac_pac_norm': comparison_data["gpac_pac_norm"],
        'tensorpac_pac_norm': comparison_data["tensorpac_pac_norm"],
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'metrics': avg_metrics,
        'parameters': {
            'batch_size': args.batch_size,
            'n_chs': args.n_chs,
            'n_segments': args.n_segments,
            'seq_len': args.seq_len,
            'fs': args.fs,
            'pha_bands': [args.pha_start_hz, args.pha_end_hz, args.pha_n_bands],
            'amp_bands': [args.amp_start_hz, args.amp_end_hz, args.amp_n_bands],
            'device': args.device,
            'n_trials': args.n_trials,
            'return_dist': args.return_dist,
            'n_perm': args.n_perm
        }
    }

    # Add surrogate distributions if available
    if args.return_dist and all_surrogate_dists:
        # Convert to numpy if needed
        if isinstance(all_surrogate_dists[0], torch.Tensor):
            all_surrogate_dists = [dist.cpu().numpy() for dist in all_surrogate_dists]

        # Stack along a new dimension (trials)
        surrogate_stack = np.stack(all_surrogate_dists, axis=0)
        full_results['gpac_surrogate_dists'] = surrogate_stack

        # Save surrogate distributions separately to avoid large files
        stx.io.save(
            surrogate_stack,
            "./results/exp_02/pac_values/gpac_surrogate_distributions.npz",
        )

    stx.io.save(
        full_results,
        "./results/exp_02/pac_values/complete_comparison_data.npz",
    )

    # Print key metrics
    print("\nAverage Comparison Metrics (across trials):")
    print(f"  Pearson correlation: {avg_metrics['pearson_r']:.4f}")
    print(f"  Spearman correlation: {avg_metrics['spearman_r']:.4f}")
    print(f"  RMSE: {avg_metrics['rmse']:.4f}")
    print(f"  Cosine similarity: {avg_metrics['cosine_similarity']:.4f}")
    print(f"  Jensen-Shannon divergence: {avg_metrics['jensen_shannon_divergence']:.4f}")
    print(f"  Top peak location overlap: {avg_metrics['top_peak_overlap_ratio']:.4f}")

    if args.return_dist:
        print("\nSurrogate distribution data has been saved")
        if len(all_surrogate_dists) > 0:
            print(f"  Shape: {all_surrogate_dists[0].shape}")

    plt.close(fig)  # Close figure to free memory
    return 0

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PAC values from gPAC and Tensorpac"
    )

    # Data parameters
    parser.add_argument(
        "--data_path", type=str, default="./data/exp_01/synthetic_pac_signals.pt",
        help="Path to synthetic data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of samples in batch"
    )
    parser.add_argument(
        "--n_chs", type=int, default=2,
        help="Number of channels"
    )
    parser.add_argument(
        "--n_segments", type=int, default=1,
        help="Number of segments"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2000,
        help="Sequence length"
    )
    parser.add_argument(
        "--fs", type=float, default=1000.0,
        help="Sampling frequency"
    )

    # Frequency parameters
    parser.add_argument(
        "--pha_start_hz", type=float, default=2.0,
        help="Phase start frequency"
    )
    parser.add_argument(
        "--pha_end_hz", type=float, default=20.0,
        help="Phase end frequency"
    )
    parser.add_argument(
        "--pha_n_bands", type=int, default=10,
        help="Number of phase bands"
    )
    parser.add_argument(
        "--amp_start_hz", type=float, default=50.0,
        help="Amplitude start frequency"
    )
    parser.add_argument(
        "--amp_end_hz", type=float, default=200.0,
        help="Amplitude end frequency"
    )
    parser.add_argument(
        "--amp_n_bands", type=int, default=10,
        help="Number of amplitude bands"
    )

    # PAC calculation parameters
    parser.add_argument(
        "--n_perm", type=int, default=200,
        help="Number of permutations for surrogate testing"
    )
    parser.add_argument(
        "--return_dist", action="store_true",
        help="Return surrogate distributions with PAC values"
    )

    # Computation parameters
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device to use for gPAC computation"
    )

    # Trial parameters
    parser.add_argument(
        "--n_trials", type=int, default=3,
        help="Number of trials for each test"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

# EOF
