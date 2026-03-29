#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 11:55:32 (ywatanabe)"
# File: ./scripts/exp_03_advanced_statistics/return_pac_distributions.py
# ----------------------------------------
import os

import scitex as stx

__FILE__ = os.path.abspath(__file__)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Extends gPAC to return full permutation test distributions
  - Provides functions for statistical analysis of PAC distributions
  - Visualizes PAC null distributions for significance testing
  - Supports custom statistical threshold definitions
  - Includes examples of advanced statistical analyses on PAC data

Dependencies:
  - packages:
    - scitex
    - numpy
    - torch
    - matplotlib
    - scipy
    - gpac

IO:
  - input-files:
    - None (interfaces with gPAC core functionality)
  - output-files:
    - ./results/exp_03/distribution_examples.png
    - ./results/exp_03/statistical_threshold_comparison.png
"""

"""Imports"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

# Get the project root directory
project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))

# Import PAC modules from gpac package
from src.gpac._pac import PAC as CorePAC
from src.gpac._pac import calculate_pac


class PACDistributionAnalyzer:
    """
    Analyzes permutation distributions from Phase-Amplitude Coupling (PAC) calculations.
    Provides statistical methods and visualizations for permutation-based significance testing.
    """

    def __init__(self, n_perm: int = 200, alpha: float = 0.05):
        self.n_perm = n_perm
        self.alpha = alpha

    def calculate_thresholds(self, distributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate statistical thresholds from PAC permutation distributions.

        Parameters
        ----------
        distributions : torch.Tensor
            The permutation distributions returned by PAC calculation
            Shape: (n_perm, batch, channels, phase_freqs, amp_freqs)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing different threshold calculations:
            - 'percentile': Threshold based on percentile of distribution
            - 'zscore': Threshold based on Z-score (mean + Z*std)
            - 'fdr': Threshold with False Discovery Rate correction
        """
        # Get dimensions
        n_perm, batch_size, n_channels, n_pha_bands, n_amp_bands = distributions.shape
        device = distributions.device

        # Calculate simple percentile threshold
        percentile_threshold = torch.quantile(
            distributions, 1 - self.alpha, dim=0
        )

        # Calculate Z-score threshold (mean + Z*std)
        z_critical = stats.norm.ppf(1 - self.alpha)
        mean_dist = distributions.mean(dim=0)
        std_dist = distributions.std(dim=0)
        zscore_threshold = mean_dist + z_critical * std_dist

        # FDR correction (Benjamini-Hochberg procedure)
        # Reshape for easier processing
        dist_flat = distributions.reshape(n_perm, -1)  # (n_perm, batch*channels*pha*amp)
        n_tests = dist_flat.shape[1]

        # Get p-values for each position by comparing observed to distribution
        # For demonstration, using random values as "observed" here
        # In real usage, this would be the actual PAC values
        observed = torch.rand(n_tests, device=device)

        # For each test position, count how many permutation values are >= observed
        p_values = torch.zeros(n_tests, device=device)
        for i in range(n_tests):
            p_values[i] = (dist_flat[:, i] >= observed[i]).float().mean()

        # Sort p-values and get threshold with FDR correction
        sorted_p, sorted_indices = torch.sort(p_values)
        thresholds = torch.arange(1, n_tests + 1, device=device) * self.alpha / n_tests

        # Find largest p-value that satisfies p <= (i/m)alpha
        passing_indices = sorted_p <= thresholds
        if passing_indices.any():
            max_idx = torch.argmax(passing_indices.long() * torch.arange(1, n_tests + 1, device=device))
            fdr_threshold = sorted_p[max_idx]
        else:
            fdr_threshold = torch.tensor(self.alpha, device=device)

        # Reshape FDR threshold to match dimensions of original data
        fdr_threshold = fdr_threshold.expand(batch_size, n_channels, n_pha_bands, n_amp_bands)

        return {
            'percentile': percentile_threshold,
            'zscore': zscore_threshold,
            'fdr': fdr_threshold
        }

    def visualize_distributions(
        self,
        pac_values: torch.Tensor,
        distributions: torch.Tensor,
        pha_freqs: np.ndarray,
        amp_freqs: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize PAC distributions and significance thresholds.

        Parameters
        ----------
        pac_values : torch.Tensor
            PAC Z-scores from calculation
        distributions : torch.Tensor
            Permutation distributions from PAC calculation
        pha_freqs : np.ndarray
            Phase frequencies used in PAC calculation
        amp_freqs : np.ndarray
            Amplitude frequencies used in PAC calculation
        save_path : Optional[str]
            Path to save the figure, if provided

        Returns
        -------
        plt.Figure
            Matplotlib Figure object with distribution visualizations
        """
        # Move tensors to CPU and convert to numpy for plotting
        if isinstance(pac_values, torch.Tensor):
            pac_values = pac_values.detach().cpu().numpy()
        if isinstance(distributions, torch.Tensor):
            distributions = distributions.detach().cpu().numpy()

        # Get dimensions and select example from first batch and channel
        batch_idx, chan_idx = 0, 0
        pac_example = pac_values[batch_idx, chan_idx]

        # Find maximum PAC value position
        max_pos = np.unravel_index(np.argmax(pac_example), pac_example.shape)
        pha_idx, amp_idx = max_pos

        # Extract distribution for this position
        dist = distributions[:, batch_idx, chan_idx, pha_idx, amp_idx]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. PAC Matrix Plot
        im = axes[0, 0].imshow(
            pac_example,
            aspect='auto',
            origin='lower',
            extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]]
        )
        axes[0, 0].set_title("PAC Z-scores")
        axes[0, 0].set_xlabel("Amplitude Frequency (Hz)")
        axes[0, 0].set_ylabel("Phase Frequency (Hz)")
        plt.colorbar(im, ax=axes[0, 0])
        axes[0, 0].plot(amp_freqs[amp_idx], pha_freqs[pha_idx], 'rx', markersize=10)

        # 2. Histogram of selected distribution
        axes[0, 1].hist(dist, bins=30, alpha=0.7, density=True)
        axes[0, 1].axvline(pac_example[pha_idx, amp_idx], color='r', linestyle='--',
                          label=f'Observed Z={pac_example[pha_idx, amp_idx]:.2f}')

        # Add percentile lines
        percentiles = [90, 95, 99]
        colors = ['green', 'orange', 'red']
        for p, c in zip(percentiles, colors):
            thresh = np.percentile(dist, p)
            axes[0, 1].axvline(thresh, color=c, linestyle='-',
                              label=f'{p}th percentile: {thresh:.2f}')

        axes[0, 1].set_title(f"Null Distribution at ({pha_freqs[pha_idx]:.1f}Hz, {amp_freqs[amp_idx]:.1f}Hz)")
        axes[0, 1].set_xlabel("PAC Value")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()

        # 3. QQ Plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(dist)))
        sorted_data = np.sort(dist)
        axes[1, 0].scatter(theoretical_quantiles, sorted_data)

        # Add reference line
        min_val = min(np.min(theoretical_quantiles), np.min(sorted_data))
        max_val = max(np.max(theoretical_quantiles), np.max(sorted_data))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')

        axes[1, 0].set_title("Q-Q Plot of Null Distribution")
        axes[1, 0].set_xlabel("Theoretical Quantiles")
        axes[1, 0].set_ylabel("Sample Quantiles")

        # 4. Threshold Comparison
        thresholds = self.calculate_thresholds(torch.tensor(distributions))
        threshold_names = ['percentile', 'zscore', 'fdr']
        threshold_labels = ['Percentile', 'Z-score', 'FDR']

        bar_data = []
        for name in threshold_names:
            thresh = thresholds[name].detach().cpu().numpy()[batch_idx, chan_idx, pha_idx, amp_idx]
            bar_data.append(thresh)

        axes[1, 1].bar(threshold_labels, bar_data, alpha=0.7)
        axes[1, 1].axhline(pac_example[pha_idx, amp_idx], color='r', linestyle='--',
                          label=f'Observed Z={pac_example[pha_idx, amp_idx]:.2f}')

        axes[1, 1].set_title("Threshold Comparison")
        axes[1, 1].set_ylabel("Threshold Value")
        axes[1, 1].legend()

        plt.tight_layout()

        # Save figure if path is provided
        if save_path:
            stx.io.save(fig, save_path)
            print(f"Figure saved to {save_path}")

        return fig


class EnhancedPAC(CorePAC):
    """
    Enhanced PAC module that extends the core PAC implementation to support
    returning the full permutation distribution.
    """

    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_start_hz: float = 2.0,
        pha_end_hz: float = 20.0,
        pha_n_bands: int = 50,
        amp_start_hz: float = 60.0,
        amp_end_hz: float = 160.0,
        amp_n_bands: int = 30,
        n_perm: Optional[int] = None,
        trainable: bool = False,
        fp16: bool = False,
        amp_prob: bool = False,
        mi_n_bins: int = 18,
        filter_cycle: int = 3,
        return_dist: bool = False,
    ):
        # Initialize with parent class
        super().__init__(
            seq_len=seq_len,
            fs=fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            trainable=trainable,
            fp16=fp16,
            amp_prob=amp_prob,
            mi_n_bins=mi_n_bins,
            filter_cycle=filter_cycle,
            return_dist=return_dist,
        )


def calculate_enhanced_pac(
    signal: torch.Tensor | np.ndarray,
    fs: float,
    pha_start_hz: float = 2.0,
    pha_end_hz: float = 20.0,
    pha_n_bands: int = 50,
    amp_start_hz: float = 60.0,
    amp_end_hz: float = 160.0,
    amp_n_bands: int = 30,
    n_perm: Optional[int] = None,
    trainable: bool = False,
    fp16: bool = False,
    amp_prob: bool = False,
    mi_n_bins: int = 18,
    filter_cycle: int = 3,
    device: Optional[str | torch.device] = None,
    chunk_size: Optional[int] = None,
    average_channels: bool = False,
    return_dist: bool = False,
) -> Union[
    Tuple[torch.Tensor, np.ndarray, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
]:
    """
    High-level function to calculate Phase-Amplitude Coupling (PAC) with the option
    to return the full permutation distribution.

    This is a wrapper around the core calculate_pac function with the return_dist option.

    Parameters
    ----------
    signal : torch.Tensor | np.ndarray
        Input signal as tensor or numpy array
    fs : float
        Sampling frequency in Hz
    pha_start_hz : float, optional
        Lowest phase frequency to analyze
    pha_end_hz : float, optional
        Highest phase frequency to analyze
    pha_n_bands : int, optional
        Number of phase frequency bands
    amp_start_hz : float, optional
        Lowest amplitude frequency to analyze
    amp_end_hz : float, optional
        Highest amplitude frequency to analyze
    amp_n_bands : int, optional
        Number of amplitude frequency bands
    n_perm : Optional[int], optional
        Number of permutations for surrogate testing (None to skip)
    trainable : bool, optional
        Whether to use trainable frequency bands
    fp16 : bool, optional
        Use half precision (float16)
    amp_prob : bool, optional
        Calculate amplitude probability instead of modulation index
    mi_n_bins : int, optional
        Number of bins for modulation index calculation
    filter_cycle : int, optional
        Number of cycles for filter design
    device : Optional[str | torch.device], optional
        Computation device ("cuda", "cpu", or torch.device)
    chunk_size : Optional[int], optional
        Process in chunks of this size (None for no chunking)
    average_channels : bool, optional
        Whether to average across channels in the output
    return_dist : bool, optional
        Whether to return the full distribution of surrogate PAC values

    Returns
    -------
    Union[Tuple[torch.Tensor, np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]]
        When return_dist=False:
            - PAC values tensor with shape (B, C, F_pha, F_amp) or (B, F_pha, F_amp)
              if average_channels=True
            - Phase frequencies as numpy array
            - Amplitude frequencies as numpy array

        When return_dist=True and n_perm is not None:
            - PAC values tensor with shape (B, C, F_pha, F_amp) or (B, F_pha, F_amp)
            - Surrogate distribution tensor with shape (n_perm, B, C, F_pha, F_amp)
              or (n_perm, B, F_pha, F_amp) if average_channels=True
            - Phase frequencies as numpy array
            - Amplitude frequencies as numpy array
    """
    # Use the core calculate_pac function with return_dist option
    return calculate_pac(
        signal=signal,
        fs=fs,
        pha_start_hz=pha_start_hz,
        pha_end_hz=pha_end_hz,
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_start_hz,
        amp_end_hz=amp_end_hz,
        amp_n_bands=amp_n_bands,
        n_perm=n_perm,
        trainable=trainable,
        fp16=fp16,
        amp_prob=amp_prob,
        mi_n_bins=mi_n_bins,
        filter_cycle=filter_cycle,
        device=device,
        chunk_size=chunk_size,
        average_channels=average_channels,
        return_dist=return_dist,
    )


def run_example(output_dir: str = './results') -> None:
    """
    Run an example demonstration of PAC distribution analysis.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save output figures
    """
    # Create synthetic signal for demonstration
    print("Generating synthetic signal...")
    fs = 1000
    duration = 10  # seconds
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)

    # Create two components with PAC
    pha_freq = 8.0  # Hz
    amp_freq = 80.0  # Hz

    # Phase component
    pha_comp = np.sin(2 * np.pi * pha_freq * t)

    # Amplitude modulation
    amp_env = 1 + 0.8 * np.sin(2 * np.pi * pha_freq * t + np.pi/2)  # Shift to cosine for coupling
    amp_comp = amp_env * np.sin(2 * np.pi * amp_freq * t)

    # Combine and add noise
    signal = pha_comp + amp_comp
    signal += 0.2 * np.random.randn(n_samples)

    # Reshape for PAC calculation: [batch, channels, segments, time]
    signal_tensor = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)

    # Calculate PAC with distribution
    print("Calculating PAC with permutation distributions...")
    result = calculate_enhanced_pac(
        signal=signal_tensor,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=10,
        amp_start_hz=60.0,
        amp_end_hz=120.0,
        amp_n_bands=10,
        n_perm=200,
        trainable=False,
        return_dist=True
    )

    # Check if we got distribution back
    if len(result) == 4:
        pac_values, surrogate_dist, pha_freqs, amp_freqs = result

        # Analyze distributions
        print("Analyzing PAC distributions...")
        analyzer = PACDistributionAnalyzer(n_perm=200)

        # Calculate thresholds
        thresholds = analyzer.calculate_thresholds(surrogate_dist)

        # Create a dictionary of results to save
        results_data = {
            'pac_values': pac_values.cpu().numpy(),
            'surrogate_dist': surrogate_dist.cpu().numpy(),
            'pha_freqs': pha_freqs,
            'amp_freqs': amp_freqs,
            'thresholds': {k: v.cpu().numpy() for k, v in thresholds.items()},
            'parameters': {
                'fs': fs,
                'duration': duration,
                'pha_freq': pha_freq,
                'amp_freq': amp_freq,
                'n_perm': 200
            }
        }

        # Save numerical results using stx framework
        stx.io.save(
            results_data,
            "./results/exp_03/pac_distribution_analysis.npz",
        )

        # Visualize distributions
        fig = analyzer.visualize_distributions(
            pac_values=pac_values,
            distributions=surrogate_dist,
            pha_freqs=pha_freqs,
            amp_freqs=amp_freqs,
            save_path=None  # Don't save directly, use stx.io.save instead
        )

        # Save figure using stx framework
        stx.io.save(
            fig,
            "./results/exp_03/distribution_examples.png",
        )

        # Create a statistical threshold comparison figure
        fig_thresh = plt.figure(figsize=(12, 8))
        ax = fig_thresh.add_subplot(111)

        # Extract data for visualization
        batch_idx, chan_idx = 0, 0
        pac_example = pac_values[batch_idx, chan_idx].cpu().numpy()

        # Find maximum PAC value position
        max_pos = np.unravel_index(np.argmax(pac_example), pac_example.shape)
        pha_idx, amp_idx = max_pos

        # Get threshold values at peak position
        threshold_values = {
            k: float(v[batch_idx, chan_idx, pha_idx, amp_idx].cpu().numpy())
            for k, v in thresholds.items()
        }

        # Plot thresholds
        threshold_labels = ['Percentile', 'Z-score', 'FDR']
        threshold_values_list = [threshold_values[k.lower()] for k in ['percentile', 'zscore', 'fdr']]

        ax.bar(threshold_labels, threshold_values_list, alpha=0.7)
        ax.axhline(
            pac_example[pha_idx, amp_idx],
            color='r',
            linestyle='--',
            label=f'Observed Z={pac_example[pha_idx, amp_idx]:.2f}'
        )

        ax.set_title("Statistical Threshold Comparison")
        ax.set_ylabel("Threshold Value")
        ax.legend()

        plt.tight_layout()

        # Save second figure using stx
        stx.io.save(
            fig_thresh,
            "./results/exp_03/statistical_threshold_comparison.png",
        )

        # Close figures to free memory
        plt.close(fig)
        plt.close(fig_thresh)

        print("Analysis complete. Results saved using stx framework.")
    else:
        print("Error: Distribution was not returned. Check n_perm and return_dist parameters.")


@stx.session
def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main entry point for the script.

    Parameters
    ----------
    args : Optional[argparse.Namespace]
        Command-line arguments

    Returns
    -------
    int
        Exit status (0 for success)
    """
    # Handle arguments
    if args is None:
        parser = argparse.ArgumentParser(description='PAC Distribution Analysis')
        parser.add_argument('--output_dir', type=str, default='./results',
                            help='Directory to save output figures')
        args = parser.parse_args()

    # Run the example
    try:
        run_example(output_dir=args.output_dir)
        return 0
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PAC Distribution Analysis')
    parser.add_argument('--output_dir', type=str,
                        default='./results/exp_03',
                        help='Directory to save output figures')
    args = parser.parse_args()
    main(args)
