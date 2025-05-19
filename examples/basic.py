#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-17 17:40:25 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/basic.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/basic.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates basic usage of the gPAC package
  - Generates synthetic signals with Phase-Amplitude Coupling
  - Calculates PAC values for different frequency combinations
  - Visualizes the results with heatmaps and statistics
Dependencies:
  - packages:
    - gpac
    - numpy
    - torch
    - matplotlib
    - mngs
IO:
  - output-files:
    - ./results/basic_example/pac_analysis.png
    - ./results/basic_example/pac_values.npy
"""
"""Imports"""
import argparse

import gpac
import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch

"""Parameters"""
DEFAULT_PARAMS = {
    "fs": 1000,  # Sampling frequency in Hz
    "duration": 5.0,  # Signal duration in seconds
    "pha_freq": 5.0,  # Phase modulating frequency in Hz
    "amp_freq": 80.0,  # Amplitude carrier frequency in Hz
    "coupling": 0.8,  # Coupling strength (0-1)
    "noise": 0.2,  # Noise level
    "pha_bands": 20,  # Number of phase frequency bands to analyze
    "amp_bands": 20,  # Number of amplitude frequency bands to analyze
    "n_permutations": 200,  # Number of permutations for statistical testing
}

"""Functions & Classes"""
def generate_pac_signal(args):
    """Generate a synthetic signal with phase-amplitude coupling."""
    # Create time vector
    time = np.arange(0, args.duration, 1 / args.fs)
    seq_len = len(time)

    # Generate phase signal (slow oscillation)
    phase_signal = np.sin(2 * np.pi * args.pha_freq * time)

    # Create amplitude modulation based on phase
    modulation = (
        1 + args.coupling * np.cos(2 * np.pi * args.pha_freq * time)
    ) / 2

    # Create carrier signal (fast oscillation)
    carrier = np.sin(2 * np.pi * args.amp_freq * time)

    # Apply amplitude modulation to carrier
    modulated_carrier = modulation * carrier

    # Create final signal with both components
    pac_signal = phase_signal + modulated_carrier

    # Add noise
    noise = np.random.normal(0, args.noise, len(time))
    signal = pac_signal + noise

    # Reshape to match expected dimensions (batch, channels, segments, time)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).reshape(
        1, 1, 1, -1
    )

    return signal_tensor, time


def analyze_pac(signal, args):
    """Calculate PAC values for the input signal."""
    print(f"Calculating PAC using gPAC v{gpac.__version__}...")

    # Calculate PAC values with permutation testing
    pac_values, surrogate_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal=signal,
        fs=args.fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=args.pha_bands,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=args.amp_bands,
        n_perm=args.n_permutations,
        return_dist=True,
    )

    return pac_values, surrogate_dist, pha_freqs, amp_freqs


def visualize_results(
    signal, pac_values, surrogate_dist, pha_freqs, amp_freqs, args
):
    """Create visualization of the PAC analysis results."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Plot the original signal
    ax1 = fig.add_subplot(2, 2, 1)
    time = np.arange(0, args.duration, 1 / args.fs)
    ax1.plot(time, signal[0, 0, 0, :].numpy())
    ax1.set_title("Original Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    # 2. Plot the PAC values as a heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(
        pac_values[0, 0].numpy(),
        origin="lower",
        aspect="auto",
        extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
    )
    ax2.set_title("PAC Z-Scores")
    ax2.set_xlabel("Amplitude Frequency (Hz)")
    ax2.set_ylabel("Phase Frequency (Hz)")
    plt.colorbar(im, ax=ax2, label="Z-score")

    # 3. Find the maximum PAC value and highlight it
    max_idx = pac_values[0, 0].argmax()
    max_pha_idx, max_amp_idx = np.unravel_index(
        max_idx.item(), pac_values[0, 0].shape
    )
    max_pha = pha_freqs[max_pha_idx]
    max_amp = amp_freqs[max_amp_idx]

    # Add marker for maximum value
    ax2.plot(max_amp, max_pha, "ro", markersize=10)
    ax2.annotate(
        f"Max: ({max_pha:.1f}, {max_amp:.1f})",
        (max_amp, max_pha),
        xytext=(10, 10),
        textcoords="offset points",
        color="white",
        bbox=dict(boxstyle="round", fc="0.8", alpha=0.7),
    )

    # 4. Plot the surrogate distribution for the maximum PAC
    ax3 = fig.add_subplot(2, 2, 3)
    surrogate_values = surrogate_dist[
        :, 0, 0, max_pha_idx, max_amp_idx
    ].numpy()
    observed_value = pac_values[0, 0, max_pha_idx, max_amp_idx].item()

    ax3.hist(surrogate_values, bins=20, alpha=0.8)
    ax3.axvline(
        observed_value,
        color="r",
        linestyle="--",
        label=f"Observed: {observed_value:.2f}",
    )
    ax3.set_xlabel("PAC Value")
    ax3.set_ylabel("Count")
    ax3.set_title("Surrogate Distribution at Max PAC")
    ax3.legend()

    # 5. Calculate and display p-value
    p_value = (surrogate_values >= observed_value).mean()

    # 6. Plot line profiles at maximum coupling
    ax4 = fig.add_subplot(2, 2, 4)

    # Plot horizontal profile (amplitude frequency axis)
    amp_profile = pac_values[0, 0, max_pha_idx, :].numpy()
    ax4.plot(amp_freqs, amp_profile, "b-", label=f"Phase={max_pha:.1f} Hz")

    # Plot vertical profile (phase frequency axis)
    pha_profile = pac_values[0, 0, :, max_amp_idx].numpy()
    ax4.plot(pha_freqs, pha_profile, "r-", label=f"Amplitude={max_amp:.1f} Hz")

    ax4.set_title("PAC Profiles at Maximum Coupling")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("PAC Z-Score")
    ax4.legend()

    # Add overall title with analysis info
    plt.suptitle(
        f"Phase-Amplitude Coupling Analysis\n"
        + f"True Coupling: Phase={args.pha_freq}Hz, Amplitude={args.amp_freq}Hz\n"
        + f"Detected Max: Phase={max_pha:.1f}Hz, Amplitude={max_amp:.1f}Hz, p={p_value:.4f}",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save results
    pac_data = {
        "pac_values": pac_values[0, 0].numpy(),
        "pha_freqs": pha_freqs,
        "amp_freqs": amp_freqs,
        "max_pha": max_pha,
        "max_amp": max_amp,
        "p_value": p_value,
    }

    return fig, pac_data


def main(args):
    """Main function to run the PAC example."""
    print("Running gPAC basic example...")

    # Generate synthetic PAC signal
    signal, time = generate_pac_signal(args)
    print(f"Generated signal with shape: {signal.shape}")

    __import__("ipdb").set_trace()

    # Calculate PAC
    pac_values, surrogate_dist, pha_freqs, amp_freqs = analyze_pac(
        signal, args
    )
    print(f"PAC calculation complete. Shape: {pac_values.shape}")

    # Visualize results
    fig, pac_data = visualize_results(
        signal, pac_values, surrogate_dist, pha_freqs, amp_freqs, args
    )

    # Save results using mngs
    mngs.io.save(fig, "./data/pac_analysis.png", symlink_from_cwd=True)
    mngs.io.save(pac_data, "./data/pac_values.npz", symlink_from_cwd=True)

    print(
        f"Analysis complete! Results saved to {os.path.join(CONFIG['DIR_SAVE'], 'pac_analysis.png')}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    parser = argparse.ArgumentParser(description="Basic gPAC example")

    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_PARAMS["fs"],
        help=f"Sampling frequency in Hz (default: {DEFAULT_PARAMS['fs']})",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_PARAMS["duration"],
        help=f"Signal duration in seconds (default: {DEFAULT_PARAMS['duration']})",
    )

    parser.add_argument(
        "--pha_freq",
        type=float,
        default=DEFAULT_PARAMS["pha_freq"],
        help=f"Phase modulating frequency in Hz (default: {DEFAULT_PARAMS['pha_freq']})",
    )

    parser.add_argument(
        "--amp_freq",
        type=float,
        default=DEFAULT_PARAMS["amp_freq"],
        help=f"Amplitude carrier frequency in Hz (default: {DEFAULT_PARAMS['amp_freq']})",
    )

    parser.add_argument(
        "--coupling",
        type=float,
        default=DEFAULT_PARAMS["coupling"],
        help=f"Coupling strength, 0-1 (default: {DEFAULT_PARAMS['coupling']})",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=DEFAULT_PARAMS["noise"],
        help=f"Noise level (default: {DEFAULT_PARAMS['noise']})",
    )

    parser.add_argument(
        "--pha_bands",
        type=int,
        default=DEFAULT_PARAMS["pha_bands"],
        help=f"Number of phase frequency bands (default: {DEFAULT_PARAMS['pha_bands']})",
    )

    parser.add_argument(
        "--amp_bands",
        type=int,
        default=DEFAULT_PARAMS["amp_bands"],
        help=f"Number of amplitude frequency bands (default: {DEFAULT_PARAMS['amp_bands']})",
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=DEFAULT_PARAMS["n_permutations"],
        help=f"Number of permutations for statistical testing (default: {DEFAULT_PARAMS['n_permutations']})",
    )

    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
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
