#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 08:23:18 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/01_basic_pac_visualization.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/01_basic_pac_visualization.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Timestamp: "2025-05-16 10:58:12"

"""
Basic PAC visualization example with surrogate distributions.

This example demonstrates:
1. Creating a synthetic signal with known PAC
2. Calculating PAC with surrogate distributions
3. Visualizing PAC values with a heatmap
4. Visualizing surrogate distributions for specific frequency pairs
5. Saving the visualizations with high-quality settings

Requirements:
- gpac
- numpy
- torch
- matplotlib
- seaborn (for better visualizations)
"""

import gpac
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

# Set plotting style for better visualizations
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

# Create output directory for figures under the examples directory
FIGURE_DIR = os.path.join(os.path.dirname(__FILE__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Parameters
FS = 1000  # Sampling frequency in Hz
DURATION = 10  # Signal duration in seconds
PHA_FREQ = 5  # Target phase frequency in Hz
AMP_FREQ = 80  # Target amplitude frequency in Hz


def create_synthetic_signal_with_pac(
    fs, duration, pha_freq, amp_freq, modulation_strength=0.8
):
    """Create a synthetic signal with known PAC."""
    # Time vector
    t = np.arange(0, duration, 1 / fs)

    # Phase signal (slow oscillation)
    phase_signal = np.sin(2 * np.pi * pha_freq * t)

    # Amplitude signal (fast oscillation)
    amp_carrier = np.sin(2 * np.pi * amp_freq * t)

    # Modulate amplitude by phase
    amplitude_modulation = (1 + modulation_strength * phase_signal) / 2
    pac_signal = amplitude_modulation * amp_carrier

    # Add some noise and the phase signal itself
    noise = np.random.normal(0, 0.1, len(t))
    signal = phase_signal * 0.5 + pac_signal + noise

    # Reshape to expected format (batch, channel, segment, time)
    tensor_signal = torch.from_numpy(signal.astype(np.float32)).reshape(
        1, 1, 1, -1
    )

    return tensor_signal, t


def plot_signal_and_pac(signal, t, fs, pha_freq, amp_freq):
    """Plot the input signal and highlight PAC components."""
    plt.figure(figsize=(12, 8))

    # Plot the full signal
    plt.subplot(411)
    plt.plot(t, signal[0, 0, 0, :].numpy(), "k", alpha=0.7, linewidth=1.5)
    plt.title("Synthetic Signal with PAC")
    plt.xlabel("Time (s)")
    plt.xlim(0, 1)  # Show first second only for clarity

    # Apply bandpass filter to extract phase component
    from scipy.signal import butter, filtfilt

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    # Extract phase component
    pha_low = pha_freq - 2
    pha_high = pha_freq + 2
    phase_component = bandpass_filter(
        signal[0, 0, 0, :].numpy(), pha_low, pha_high, fs, order=3
    )

    # Extract amplitude component
    amp_low = amp_freq - 10
    amp_high = amp_freq + 10
    amplitude_component = bandpass_filter(
        signal[0, 0, 0, :].numpy(), amp_low, amp_high, fs, order=3
    )

    # Plot phase component
    plt.subplot(412)
    plt.plot(t, phase_component, "b", linewidth=1.5)
    plt.title(f"Phase Component ({pha_freq} Hz)")
    plt.xlabel("Time (s)")
    plt.xlim(0, 1)

    # Plot amplitude component
    plt.subplot(413)
    plt.plot(t, amplitude_component, "r", linewidth=1.5)
    plt.title(f"Amplitude Component ({amp_freq} Hz)")
    plt.xlabel("Time (s)")
    plt.xlim(0, 1)

    # Plot amplitude envelope and phase
    from scipy.signal import hilbert

    # Extract envelope from amplitude component
    amplitude_analytic = hilbert(amplitude_component)
    amplitude_envelope = np.abs(amplitude_analytic)

    # Calculate instantaneous phase from phase component
    phase_analytic = hilbert(phase_component)
    instantaneous_phase = np.angle(phase_analytic)

    # Normalize phase to [0, 1] for color mapping
    norm_phase = (instantaneous_phase + np.pi) / (2 * np.pi)

    plt.subplot(414)
    plt.plot(t, amplitude_envelope, "r", linewidth=1.5, alpha=0.7)

    # Create color array based on phase
    phase_colors = plt.cm.hsv(norm_phase)

    # Plot colored points to represent phase
    for i in range(0, len(t), 20):  # Plot every 20th point for clarity
        plt.plot(
            t[i],
            amplitude_envelope[i],
            "o",
            color=phase_colors[i],
            markersize=5,
        )

    plt.title("Amplitude Envelope Colored by Phase")
    plt.xlabel("Time (s)")
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "01_signal_components.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_pac_heatmap(pac_values, pha_freqs, amp_freqs):
    """Plot PAC values as a heatmap."""
    plt.figure(figsize=(10, 8))

    # Create a custom colormap that highlights significant values
    colors = [(0.95, 0.95, 0.95), (1, 1, 0.7), (1, 0.7, 0.5), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("pac_cmap", colors, N=100)

    # Plot heatmap
    plt.imshow(
        pac_values[0, 0].cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="none",
    )

    # Customize axis ticks and labels
    # Add ticks at every 2nd position
    plt.xticks(
        np.arange(0, len(amp_freqs), 2),
        [f"{f:.1f}" for f in amp_freqs[::2]],
        rotation=45,
    )
    plt.yticks(
        np.arange(0, len(pha_freqs), 2), [f"{f:.1f}" for f in pha_freqs[::2]]
    )

    plt.xlabel("Amplitude Frequency (Hz)")
    plt.ylabel("Phase Frequency (Hz)")
    plt.title("Phase-Amplitude Coupling Z-Scores")

    # Add colorbar
    cbar = plt.colorbar(label="Z-score")
    cbar.ax.tick_params(labelsize=10)

    # Add markers for target frequencies
    target_pha_idx = np.argmin(np.abs(pha_freqs - PHA_FREQ))
    target_amp_idx = np.argmin(np.abs(amp_freqs - AMP_FREQ))

    plt.plot(
        target_amp_idx,
        target_pha_idx,
        "o",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=12,
        markeredgewidth=2,
    )

    # Annotate target with actual frequency values
    plt.annotate(
        f"Target: ({pha_freqs[target_pha_idx]:.1f}, {amp_freqs[target_amp_idx]:.1f}) Hz",
        xy=(target_amp_idx, target_pha_idx),
        xytext=(target_amp_idx - 3, target_pha_idx - 3),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "01_pac_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_surrogate_distribution(
    pac_values, surrogate_dist, pha_freqs, amp_freqs
):
    """Plot surrogate distributions for specific frequency pairs."""
    plt.figure(figsize=(15, 10))

    # Find target frequencies
    target_pha_idx = np.argmin(np.abs(pha_freqs - PHA_FREQ))
    target_amp_idx = np.argmin(np.abs(amp_freqs - AMP_FREQ))

    # Get maximum PAC value - ensure it's on CPU before converting to NumPy
    pac_cpu = pac_values[0, 0].cpu()
    max_idx = pac_cpu.argmax()
    max_pha_idx, max_amp_idx = np.unravel_index(
        max_idx.item(), pac_cpu.shape
    )

    # Define frequency pairs to plot
    pairs = [
        (target_pha_idx, target_amp_idx, "Target Frequency Pair"),
        (max_pha_idx, max_amp_idx, "Maximum PAC Value"),
        (0, 0, "Low Frequency Control"),
        (
            pha_freqs.shape[0] - 1,
            amp_freqs.shape[0] - 1,
            "High Frequency Control",
        ),
    ]

    for i, (pha_idx, amp_idx, title_suffix) in enumerate(pairs):
        plt.subplot(2, 2, i + 1)

        # Get observed and surrogate values - ensure CPU before conversion
        observed_pac = pac_values[0, 0, pha_idx, amp_idx].cpu().item()
        # Check and handle if surrogate_dist is on CUDA
        surrogate_values = (
            surrogate_dist[:, 0, 0, pha_idx, amp_idx].cpu().numpy()
        )

        # Calculate p-value and statistics
        p_value = np.mean(surrogate_values >= observed_pac)
        mean_surr = np.mean(surrogate_values)
        std_surr = np.std(surrogate_values)
        z_score = (observed_pac - mean_surr) / std_surr if std_surr > 0 else 0

        # Plot surrogate distribution
        sns.histplot(
            surrogate_values, bins=20, kde=True, color="skyblue", alpha=0.7
        )

        # Add observed value as vertical line
        plt.axvline(
            observed_pac,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed: {observed_pac:.2f}\nZ-score: {z_score:.2f}",
        )

        # Add mean of surrogates as vertical line
        plt.axvline(
            mean_surr,
            color="black",
            linestyle="-",
            linewidth=1,
            label=f"Mean of surrogates: {mean_surr:.2f}",
        )

        # Plot title with frequency information
        plt.title(
            f"{title_suffix}\n"
            f"Phase: {pha_freqs[pha_idx]:.1f} Hz, Amp: {amp_freqs[amp_idx]:.1f} Hz\n"
            f"p-value: {p_value:.4f}"
        )

        plt.xlabel("PAC Value")
        plt.ylabel("Count")
        plt.legend(loc="upper right")

        # Highlight significant results
        if p_value < 0.05:
            plt.gca().set_facecolor((1, 0.95, 0.95))
            plt.gca().patch.set_alpha(0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "01_surrogate_distributions.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """Main function to run the example."""
    # Create synthetic signal with known PAC
    print("Creating synthetic signal with PAC...")
    signal, t = create_synthetic_signal_with_pac(
        FS, DURATION, PHA_FREQ, AMP_FREQ
    )

    # Plot signal and its components
    print("Plotting signal components...")
    plot_signal_and_pac(signal, t, FS, PHA_FREQ, AMP_FREQ)

    # Calculate PAC with surrogate distribution
    print("Calculating PAC with surrogate distribution...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pac_values, surrogate_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal=signal.to(device) if device == "cuda" else signal,
        fs=FS,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=15,  # Use 15 phase bands for better visualization
        amp_start_hz=40.0,
        amp_end_hz=160.0,
        amp_n_bands=12,  # Use 12 amplitude bands for better visualization
        n_perm=200,  # Number of permutations
        return_dist=True,
        device=device,
        fp16=False,  # Use fp32 for better precision
    )

    # Convert frequencies to NumPy arrays
    pha_freqs = pha_freqs.astype(np.float32)
    amp_freqs = amp_freqs.astype(np.float32)

    # Plot PAC heatmap
    print("Plotting PAC heatmap...")
    plot_pac_heatmap(pac_values, pha_freqs, amp_freqs)

    # Plot surrogate distributions
    print("Plotting surrogate distributions...")
    plot_surrogate_distribution(
        pac_values, surrogate_dist, pha_freqs, amp_freqs
    )

    print("Done! Figures saved in the 'figures' directory:")
    print("  - figures/01_signal_components.png")
    print("  - figures/01_pac_heatmap.png")
    print("  - figures/01_surrogate_distributions.png")


if __name__ == "__main__":
    main()

# EOF
