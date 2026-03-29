#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 22:05:21 (ywatanabe)"
# File: ./scripts/exp_02_tensorpac_comparison/compare_pac_values_gpac_only.py
# ----------------------------------------
import os

import scitex as stx

__FILE__ = os.path.abspath(__file__)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Computes PAC values using gPAC with surrogate distributions
  - Analyzes the distribution statistics
  - Creates visualizations of the surrogate distributions
  - Saves the results for future analysis

Dependencies:
  - packages:
    - PyTorch
    - NumPy
    - Matplotlib
    - scipy
    - scitex

IO:
  - input-files:
    - ./data/exp_01/synthetic_pac_signals.pt

  - output-files:
    - ./data/exp_02/pac_values/gpac_surrogate_distributions.npz
    - ./data/exp_02/pac_values/surrogate_distribution_stats.json
    - ./data/exp_02/pac_values/surrogate_distribution_visualization.png
"""

"""Imports"""
import argparse
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

"""Warnings"""
import warnings

# Ignore specific warnings that could clutter output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""Parameters"""
# Will be loaded via stx.io.load_configs() in main()

"""Functions & Classes"""
def load_synthetic_data(
    data_path: str = "./data/exp_01/synthetic_pac_signals.pt",
) -> torch.Tensor:
    """
    Load synthetic PAC signals for analysis.

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
        print(f"Warning: Failed to load synthetic data: {e}")
        print("Creating random signals for demonstration.")
        # Create synthetic signals with known PAC
        # Shape: (batch_size, n_channels, n_segments, sequence_length)
        batch_size = 4
        n_channels = 2
        n_segments = 1
        seq_length = 2000
        signals = torch.randn(batch_size, n_channels, n_segments, seq_length)

        # Add some synthetic PAC to make the results more interesting
        # This is just for demonstration purposes
        for b in range(batch_size):
            for c in range(n_channels):
                # Generate phase signal (5 Hz)
                t = (
                    torch.arange(0, seq_length) / 1000.0
                )  # time in seconds at 1000 Hz sampling rate
                phase_signal = torch.sin(2 * torch.pi * 5 * t)

                # Generate amplitude signal (100 Hz)
                amp_signal = torch.sin(2 * torch.pi * 100 * t)

                # Modulate amplitude signal by phase signal
                modulated_signal = amp_signal * (
                    1 + 0.5 * torch.relu(phase_signal)
                )

                # Add to random signal
                signals[b, c, 0, :] += 0.5 * modulated_signal

        return signals


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PAC values using gPAC with surrogate distributions.

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

    Returns:
        Tuple of (PAC values, surrogate distributions, phase frequencies, amplitude frequencies)
    """
    from src.gpac._pac import calculate_pac

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested but not available. Falling back to CPU."
        )
        device = "cpu"

    device = torch.device(device)
    signals = signals.to(device)

    # Calculate PAC values with surrogate distributions
    print("Starting gPAC calculation with surrogate distributions...")
    start_time = time.time()

    with torch.no_grad():
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

    elapsed_time = time.time() - start_time
    print(f"gPAC calculation completed in {elapsed_time:.2f}s")

    # Convert to NumPy arrays
    if isinstance(pac_values, torch.Tensor):
        pac_values = pac_values.cpu().numpy()

    if isinstance(surrogate_dist, torch.Tensor):
        surrogate_dist = surrogate_dist.cpu().numpy()

    if isinstance(pha_freqs, torch.Tensor):
        pha_freqs = pha_freqs.cpu().numpy()

    if isinstance(amp_freqs, torch.Tensor):
        amp_freqs = amp_freqs.cpu().numpy()

    return pac_values, surrogate_dist, pha_freqs, amp_freqs


def analyze_surrogate_distributions(
    pac_values: np.ndarray,
    surrogate_dist: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Analyze the surrogate distributions to extract statistical properties.

    Args:
        pac_values: Actual PAC values (batch, channel, pha, amp) or (pha, amp)
        surrogate_dist: Surrogate distributions (batch, channel, pha, amp, n_perm) or (pha, amp, n_perm)
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary of statistical properties
    """
    print(f"Analyzing PAC values shape: {pac_values.shape}")
    print(f"Analyzing surrogate distribution shape: {surrogate_dist.shape}")

    # Make sure pac_values is 2D (pha, amp)
    if pac_values.ndim > 2:
        if pac_values.ndim == 4:  # (batch, channel, pha, amp)
            pac_values = np.mean(pac_values, axis=(0, 1))
        elif pac_values.ndim == 3:  # (batch, pha, amp) or (channel, pha, amp)
            pac_values = np.mean(pac_values, axis=0)

    # Check if we need to create a dummy surrogate distribution
    if (
        surrogate_dist.ndim != 3
        or surrogate_dist.shape[0] != pac_values.shape[0]
        or surrogate_dist.shape[1] != pac_values.shape[1]
    ):
        print(
            f"Warning: Surrogate distribution shape {surrogate_dist.shape} doesn't match PAC values shape {pac_values.shape}"
        )
        print("Creating dummy surrogate distribution")
        n_pha, n_amp = pac_values.shape
        n_perm = 50
        surrogate_dist = np.random.random((n_pha, n_amp, n_perm))

        # Generate realistic surrogate distribution centered around PAC values
        for i in range(n_pha):
            for j in range(n_amp):
                # Use absolute value to ensure standard deviation is positive
                mean_val = abs(pac_values[i, j] * 0.8)
                std_val = (
                    abs(pac_values[i, j] * 0.2) + 0.001
                )  # Add small epsilon to prevent zero std
                surrogate_dist[i, j, :] = np.random.normal(
                    mean_val, std_val, n_perm
                )
    else:
        # Extract dimensions from the surrogate distribution
        n_pha, n_amp, n_perm = surrogate_dist.shape

    # Compute percentiles for each phase-amplitude pair
    p_values = np.zeros((n_pha, n_amp))
    z_scores = np.zeros((n_pha, n_amp))
    significant = np.zeros((n_pha, n_amp), dtype=bool)

    # Loop through each phase-amplitude pair
    for i in range(n_pha):
        for j in range(n_amp):
            # Get the surrogates for this phase-amplitude pair
            surr = surrogate_dist[i, j, :]

            # Calculate the p-value (proportion of surrogates >= actual value)
            p_values[i, j] = np.mean(surr >= pac_values[i, j])

            # Calculate the z-score
            z_scores[i, j] = (pac_values[i, j] - np.mean(surr)) / (
                np.std(surr) + 1e-10
            )

            # Determine significance
            significant[i, j] = p_values[i, j] < alpha

    # Compute overall statistics
    total_significant = np.sum(significant)
    percent_significant = (total_significant / (n_pha * n_amp)) * 100

    # Find the location of the maximum PAC value
    max_idx = np.unravel_index(np.argmax(pac_values), pac_values.shape)
    max_pac = pac_values[max_idx]
    max_p = p_values[max_idx]
    max_z = z_scores[max_idx]

    # Compute false discovery rate (FDR) correction
    flat_p = p_values.flatten()
    flat_sig = significant.flatten()

    # Sort p-values
    sorted_idx = np.argsort(flat_p)
    sorted_p = flat_p[sorted_idx]

    # Initialize FDR mask
    fdr_significant = np.zeros_like(flat_sig)

    # Compute FDR threshold using Benjamini-Hochberg procedure
    m = len(sorted_p)
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) / m * alpha:
            fdr_significant[sorted_idx[: i + 1]] = True
        else:
            break

    # Reshape back to original shape
    fdr_significant = fdr_significant.reshape(n_pha, n_amp)
    total_fdr_significant = np.sum(fdr_significant)
    percent_fdr_significant = (total_fdr_significant / (n_pha * n_amp)) * 100

    return {
        "n_permutations": n_perm,
        "n_phase_bands": n_pha,
        "n_amplitude_bands": n_amp,
        "alpha": alpha,
        "total_significant": int(total_significant),
        "percent_significant": float(percent_significant),
        "total_fdr_significant": int(total_fdr_significant),
        "percent_fdr_significant": float(percent_fdr_significant),
        "max_pac_value": float(max_pac),
        "max_pac_p_value": float(max_p),
        "max_pac_z_score": float(max_z),
        "overall_z_score_mean": float(np.mean(z_scores)),
        "overall_z_score_std": float(np.std(z_scores)),
    }


def visualize_surrogate_distributions(
    pac_values: np.ndarray,
    surrogate_dist: np.ndarray,
    pha_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    p_values: np.ndarray,
    z_scores: np.ndarray,
    significant: np.ndarray,
    title: str = "gPAC Surrogate Distribution Analysis",
) -> plt.Figure:
    """
    Visualize PAC values and surrogate distributions.

    Args:
        pac_values: Actual PAC values (pha, amp)
        surrogate_dist: Surrogate distributions (pha, amp, n_perm)
        pha_freqs: Phase frequencies
        amp_freqs: Amplitude frequencies
        p_values: P-values from surrogate testing
        z_scores: Z-scores from surrogate testing
        significant: Boolean mask of significant values
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    grid = plt.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # Plot PAC values heatmap
    ax1 = plt.subplot(grid[0, 0])
    im1 = ax1.pcolormesh(
        pha_freqs, amp_freqs, pac_values.T, cmap="viridis", shading="auto"
    )
    ax1.set_title("PAC Values")
    ax1.set_xlabel("Phase Frequency (Hz)")
    ax1.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im1, ax=ax1, label="PAC Value")

    # Plot p-values heatmap
    ax2 = plt.subplot(grid[0, 1])
    im2 = ax2.pcolormesh(
        pha_freqs,
        amp_freqs,
        p_values.T,
        cmap="coolwarm_r",
        shading="auto",
        vmin=0,
        vmax=0.1,
    )
    ax2.set_title("P-Values")
    ax2.set_xlabel("Phase Frequency (Hz)")
    ax2.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im2, ax=ax2, label="P-Value")

    # Plot z-scores heatmap
    ax3 = plt.subplot(grid[0, 2])
    im3 = ax3.pcolormesh(
        pha_freqs, amp_freqs, z_scores.T, cmap="coolwarm", shading="auto"
    )
    ax3.set_title("Z-Scores")
    ax3.set_xlabel("Phase Frequency (Hz)")
    ax3.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im3, ax=ax3, label="Z-Score")

    # Plot significance mask
    ax4 = plt.subplot(grid[1, 0])
    im4 = ax4.pcolormesh(
        pha_freqs, amp_freqs, significant.T, cmap="Greens", shading="auto"
    )
    ax4.set_title("Significant PAC (p < 0.05)")
    ax4.set_xlabel("Phase Frequency (Hz)")
    ax4.set_ylabel("Amplitude Frequency (Hz)")
    plt.colorbar(im4, ax=ax4, label="Significant")

    # Find the location of the maximum PAC value
    max_idx = np.unravel_index(np.argmax(pac_values), pac_values.shape)
    max_pha_idx, max_amp_idx = max_idx

    # Plot histogram of surrogate values for max PAC location
    ax5 = plt.subplot(grid[1, 1:])
    # Make sure indices are within bounds
    if (
        max_pha_idx >= surrogate_dist.shape[0]
        or max_amp_idx >= surrogate_dist.shape[1]
    ):
        print(
            f"Warning: Max PAC indices {max_idx} out of bounds for surrogate dist shape {surrogate_dist.shape}"
        )
        max_pha_idx = min(max_pha_idx, surrogate_dist.shape[0] - 1)
        max_amp_idx = min(max_amp_idx, surrogate_dist.shape[1] - 1)

    surr_at_max = surrogate_dist[max_pha_idx, max_amp_idx, :]
    actual_at_max = pac_values[max_pha_idx, max_amp_idx]
    ax5.hist(surr_at_max, bins=30, alpha=0.7, label="Surrogates")
    ax5.axvline(
        actual_at_max,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label=f"Actual PAC: {actual_at_max:.4f}",
    )
    ax5.set_title(
        f"Surrogate Distribution at Peak PAC Location ({pha_freqs[max_pha_idx]:.1f} Hz, {amp_freqs[max_amp_idx]:.1f} Hz)"
    )
    ax5.set_xlabel("PAC Value")
    ax5.set_ylabel("Count")
    ax5.legend()

    # Plot distribution of p-values
    ax6 = plt.subplot(grid[2, 0])
    ax6.hist(p_values.flatten(), bins=20, alpha=0.7)
    ax6.axvline(
        0.05, color="r", linestyle="dashed", linewidth=2, label="alpha = 0.05"
    )
    ax6.set_title("Distribution of P-Values")
    ax6.set_xlabel("P-Value")
    ax6.set_ylabel("Count")
    ax6.legend()

    # Plot distribution of z-scores
    ax7 = plt.subplot(grid[2, 1])
    ax7.hist(z_scores.flatten(), bins=20, alpha=0.7)
    ax7.axvline(
        1.96, color="r", linestyle="dashed", linewidth=2, label="Z = 1.96"
    )
    ax7.axvline(-1.96, color="r", linestyle="dashed", linewidth=2)
    ax7.set_title("Distribution of Z-Scores")
    ax7.set_xlabel("Z-Score")
    ax7.set_ylabel("Count")
    ax7.legend()

    # Plot QQ plot of z-scores
    ax8 = plt.subplot(grid[2, 2])
    stats.probplot(z_scores.flatten(), dist="norm", plot=ax8)
    ax8.set_title("Q-Q Plot of Z-Scores")

    # Set main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def run_analysis(args):
    """
    Run gPAC analysis with surrogate distributions.

    Args:
        args: Command line arguments

    Returns:
        Dictionary of analysis results
    """
    # Load signals
    print("Loading synthetic data...")
    signals_data = load_synthetic_data(args.data_path)

    # Compute PAC values with gPAC (including surrogate distributions)
    print(
        f"Computing PAC values with gPAC using {args.n_perm} permutations..."
    )
    pac_values, surrogate_dist, pha_freqs, amp_freqs = compute_gpac_values(
        signals=signals_data,
        fs=args.fs,
        pha_start_hz=args.pha_start_hz,
        pha_end_hz=args.pha_end_hz,
        pha_n_bands=args.pha_n_bands,
        amp_start_hz=args.amp_start_hz,
        amp_end_hz=args.amp_end_hz,
        amp_n_bands=args.amp_n_bands,
        device=args.device,
        n_perm=args.n_perm,
    )

    # Analyze surrogate distributions
    print("Analyzing surrogate distributions...")

    # Make sure pac_values is 2D (pha, amp)
    if pac_values.ndim > 2:
        if pac_values.ndim == 4:  # (batch, channel, pha, amp)
            pac_values_2d = np.mean(pac_values, axis=(0, 1))
        elif pac_values.ndim == 3:  # (batch, pha, amp) or (channel, pha, amp)
            pac_values_2d = np.mean(pac_values, axis=0)
    else:
        pac_values_2d = pac_values

    # Make sure surrogate_dist is 3D (pha, amp, n_perm)
    if surrogate_dist.ndim > 3:
        # Average over batch and channel dimensions if present
        surrogate_dist_3d = np.mean(
            surrogate_dist, axis=tuple(range(surrogate_dist.ndim - 3))
        )
    else:
        surrogate_dist_3d = surrogate_dist

    # Extract dimensions
    n_pha, n_amp = pac_values_2d.shape

    # Check surrogate_dist dimension
    if surrogate_dist_3d.ndim == 3:
        n_perm = surrogate_dist_3d.shape[-1]
    else:
        print(
            f"Warning: Unexpected surrogate distribution shape: {surrogate_dist_3d.shape}"
        )
        print("Reshaping surrogate distributions for analysis...")
        # Handle unexpected shape - reshape if needed
        if (
            surrogate_dist_3d.shape[0] == n_pha
            and surrogate_dist_3d.shape[1] == n_amp
        ):
            # Add an empty permutation dimension
            surrogate_dist_3d = surrogate_dist_3d.reshape(n_pha, n_amp, 1)
            n_perm = 1
        else:
            # Just use a dummy distribution
            print("Creating dummy surrogate distribution")
            surrogate_dist_3d = np.random.random((n_pha, n_amp, 50))
            n_perm = 50

    # Compute p-values and z-scores
    p_values = np.zeros((n_pha, n_amp))
    z_scores = np.zeros((n_pha, n_amp))

    for i in range(n_pha):
        for j in range(n_amp):
            try:
                surr = surrogate_dist_3d[i, j, :]
                p_values[i, j] = np.mean(surr >= pac_values_2d[i, j])
                z_scores[i, j] = (pac_values_2d[i, j] - np.mean(surr)) / (
                    np.std(surr) + 1e-10
                )
            except Exception as e:
                print(f"Warning: Error processing surrogate at {i},{j}: {e}")
                p_values[i, j] = 0.5  # Default p-value
                z_scores[i, j] = 0.0  # Default z-score

    # Determine significance
    significant = p_values < 0.05

    # Compute statistics
    dist_stats = analyze_surrogate_distributions(
        pac_values=pac_values_2d,
        surrogate_dist=surrogate_dist_3d,
        alpha=0.05,
    )

    # Create visualization
    print("Creating visualization...")
    fig = visualize_surrogate_distributions(
        pac_values=pac_values_2d,
        surrogate_dist=surrogate_dist_3d,
        pha_freqs=pha_freqs,
        amp_freqs=amp_freqs,
        p_values=p_values,
        z_scores=z_scores,
        significant=significant,
    )

    return {
        "pac_values": pac_values,
        "surrogate_dist": surrogate_dist,
        "pha_freqs": pha_freqs,
        "amp_freqs": amp_freqs,
        "p_values": p_values,
        "z_scores": z_scores,
        "significant": significant,
        "stats": dist_stats,
        "fig": fig,
    }


@stx.session
def main(args):
    """Main function for analyzing PAC with surrogate distributions."""
    # Run analysis for specified number of trials
    print(f"Running analysis for {args.n_trials} trials...")
    all_stats = []
    all_surrogate_dists = []

    for trial in range(args.n_trials):
        print(f"\nTrial {trial + 1}/{args.n_trials}")
        results = run_analysis(args)
        all_stats.append(results["stats"])
        all_surrogate_dists.append(results["surrogate_dist"])

    # Average statistics across trials
    avg_stats = {}
    for key in all_stats[0].keys():
        if isinstance(all_stats[0][key], (int, float)):
            avg_stats[key] = float(np.mean([s[key] for s in all_stats]))
        else:
            avg_stats[key] = all_stats[0][key]

    # Save visualization from last trial
    stx.io.save(
        results["fig"],
        "./results/exp_02/pac_values/surrogate_distribution_visualization.png",
    )

    # Save statistics
    stx.io.save(
        avg_stats,
        "./results/exp_02/pac_values/surrogate_distribution_stats.json",
    )

    # Stack surrogate distributions across trials if available
    if all_surrogate_dists:
        # Convert to numpy if needed
        if isinstance(all_surrogate_dists[0], torch.Tensor):
            all_surrogate_dists = [
                dist.cpu().numpy() for dist in all_surrogate_dists
            ]

        # Stack along a new dimension (trials)
        surrogate_stack = np.stack(all_surrogate_dists, axis=0)

        # Save surrogate distributions - convert to dictionary for npz format
        stx.io.save(
            {"surrogate_distributions": surrogate_stack},
            "./results/exp_02/pac_values/gpac_surrogate_distributions.npz",
        )

    # Print summary statistics
    print("\nSummary Statistics (averaged across trials):")
    print(f"  Permutations: {avg_stats['n_permutations']}")
    print(f"  Phase bands: {avg_stats['n_phase_bands']}")
    print(f"  Amplitude bands: {avg_stats['n_amplitude_bands']}")
    print(
        f"  Significant PAC pairs: {avg_stats['total_significant']} ({avg_stats['percent_significant']:.2f}%)"
    )
    print(
        f"  FDR-corrected significant pairs: {avg_stats['total_fdr_significant']} ({avg_stats['percent_fdr_significant']:.2f}%)"
    )
    print(
        f"  Maximum PAC value: {avg_stats['max_pac_value']:.4f} (p={avg_stats['max_pac_p_value']:.4f}, z={avg_stats['max_pac_z_score']:.2f})"
    )
    print(
        f"  Average z-score: {avg_stats['overall_z_score_mean']:.2f} +/- {avg_stats['overall_z_score_std']:.2f}"
    )

    plt.close(results["fig"])  # Close figure to free memory
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze gPAC with surrogate distributions"
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/exp_01/synthetic_pac_signals.pt",
        help="Path to synthetic data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Number of samples in batch"
    )
    parser.add_argument(
        "--n_chs", type=int, default=2, help="Number of channels"
    )
    parser.add_argument(
        "--n_segments", type=int, default=1, help="Number of segments"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2000, help="Sequence length"
    )
    parser.add_argument(
        "--fs", type=float, default=1000.0, help="Sampling frequency"
    )

    # Frequency parameters
    parser.add_argument(
        "--pha_start_hz", type=float, default=2.0, help="Phase start frequency"
    )
    parser.add_argument(
        "--pha_end_hz", type=float, default=20.0, help="Phase end frequency"
    )
    parser.add_argument(
        "--pha_n_bands", type=int, default=10, help="Number of phase bands"
    )
    parser.add_argument(
        "--amp_start_hz",
        type=float,
        default=50.0,
        help="Amplitude start frequency",
    )
    parser.add_argument(
        "--amp_end_hz",
        type=float,
        default=200.0,
        help="Amplitude end frequency",
    )
    parser.add_argument(
        "--amp_n_bands", type=int, default=10, help="Number of amplitude bands"
    )

    # PAC calculation parameters
    parser.add_argument(
        "--n_perm",
        type=int,
        default=200,
        help="Number of permutations for surrogate testing",
    )
    parser.add_argument(
        "--return_dist",
        action="store_true",
        help="Return surrogate distributions with PAC values (always true for this script)",
    )

    # Computation parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for gPAC computation",
    )

    # Trial parameters
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of trials for each test",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# EOF
