#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 00:27:54 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/learnability/.old/demonstrate_learnability.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/.old/demonstrate_learnability.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""
PAC Learnability Demonstration Script

This script demonstrates the learnability of the Phase-Amplitude Coupling (PAC) module
by conducting synthetic experiments where ground truth parameters are known, and the
module learns to recover them.

This can be used as evidence for a research paper section on learnability.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from gpac._pac import PAC

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def create_output_dir():
    """Create output directory for results."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def create_synthetic_pac_signal(
    fs=1000.0,
    duration=2.0,
    pha_freq=8.0,  # Hz, for phase component
    amp_freq=100.0,  # Hz, for amplitude component
    batch_size=2,
    n_channels=3,
    n_segments=1,
    modulation_strength=0.8,  # 0 to 1, higher = stronger coupling
):
    """Create a synthetic signal with known PAC."""
    # Create time vector
    t = np.arange(0, duration, 1 / fs)
    seq_len = len(t)

    # Create phase signal (slow oscillation)
    phase_signal = np.sin(2 * np.pi * pha_freq * t)

    # Create amplitude modulation based on phase
    # Here, amplitude peaks at phase = 0 (cosine)
    modulation = (
        1 + modulation_strength * np.cos(2 * np.pi * pha_freq * t)
    ) / 2

    # Create carrier signal (fast oscillation)
    carrier = np.sin(2 * np.pi * amp_freq * t)

    # Apply amplitude modulation to carrier
    modulated_carrier = modulation * carrier

    # Create final signal with both components
    pac_signal = phase_signal + modulated_carrier

    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    signal = pac_signal + noise

    # Create tensor with batch and channel dimensions
    # Shape: (batch_size, n_channels, n_segments, seq_len)
    signal_tensor = np.zeros((batch_size, n_channels, n_segments, seq_len))

    # Fill all batches and channels with the same signal (for simplicity)
    for b in range(batch_size):
        for c in range(n_channels):
            for s in range(n_segments):
                signal_tensor[b, c, s, :] = signal

    signal_tensor = torch.from_numpy(signal_tensor.astype(np.float32))

    # Return the tensor and the frequencies used
    return signal_tensor, pha_freq, amp_freq


def learn_pac_parameters(
    signal,
    true_pha_freq,
    true_amp_freq,
    num_epochs=100,
    learning_rate=0.01,
    device="cpu",
):
    """
    Train PAC module to learn optimal frequency parameters.

    Args:
        signal: Input signal tensor (B, C, Seg, T)
        true_pha_freq: True phase frequency used to generate the signal
        true_amp_freq: True amplitude frequency used to generate the signal
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run computation on

    Returns:
        dict: Results including losses, parameter trajectories, etc.
    """
    batch_size, n_channels, n_segments, seq_len = signal.shape
    fs = 1000.0  # Sampling rate in Hz

    # Move signal to device
    signal = signal.to(device)

    # Set up PAC module with trainable parameters
    # Create a range of frequencies around the true values
    pha_start_hz = max(1.0, true_pha_freq * 0.5)
    pha_end_hz = min(30.0, true_pha_freq * 2.0)
    pha_n_bands = 10  # More bands for higher resolution

    amp_start_hz = max(50.0, true_amp_freq * 0.5)
    amp_end_hz = min(200.0, true_amp_freq * 2.0)
    amp_n_bands = 10

    print(f"Initializing PAC module with frequency ranges:")
    print(f"  Phase: {pha_start_hz} - {pha_end_hz} Hz ({pha_n_bands} bands)")
    print(
        f"  Amplitude: {amp_start_hz} - {amp_end_hz} Hz ({amp_n_bands} bands)"
    )
    print(
        f"True frequencies to recover: Phase {true_pha_freq} Hz, Amplitude {true_amp_freq} Hz"
    )

    # Create PAC module
    model = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=pha_start_hz,
        pha_end_hz=pha_end_hz,
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_start_hz,
        amp_end_hz=amp_end_hz,
        amp_n_bands=amp_n_bands,
        trainable=True,
    ).to(device)

    # Store initial parameters
    initial_pha_mids = model.PHA_MIDS_HZ.clone().detach().cpu().numpy()
    initial_amp_mids = model.AMP_MIDS_HZ.clone().detach().cpu().numpy()

    # Set up optimizer
    optimizer = optim.Adam(
        [model.PHA_MIDS_HZ, model.AMP_MIDS_HZ], lr=learning_rate
    )

    # Track metrics during training
    losses = []
    pha_distances = []  # Distance from closest band to true phase frequency
    amp_distances = (
        []
    )  # Distance from closest band to true amplitude frequency
    pha_trajectories = np.zeros((num_epochs + 1, len(initial_pha_mids)))
    amp_trajectories = np.zeros((num_epochs + 1, len(initial_amp_mids)))

    # Store initial trajectories
    pha_trajectories[0] = initial_pha_mids
    amp_trajectories[0] = initial_amp_mids

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    # Function to calculate losses
    def calc_losses():
        # Calculate PAC values
        pac_values = model(signal)

        # Find band indices closest to true frequencies
        pha_idx = np.argmin(
            np.abs(model.PHA_MIDS_HZ.detach().cpu().numpy() - true_pha_freq)
        )
        amp_idx = np.argmin(
            np.abs(model.AMP_MIDS_HZ.detach().cpu().numpy() - true_amp_freq)
        )

        # Calculate loss: negated PAC value at the closest bands to encourage high PAC
        # We look at all combinations because we don't know which band will be closest
        loss = -pac_values.mean()

        # Calculate distance metrics
        pha_dist = np.min(
            np.abs(model.PHA_MIDS_HZ.detach().cpu().numpy() - true_pha_freq)
        )
        amp_dist = np.min(
            np.abs(model.AMP_MIDS_HZ.detach().cpu().numpy() - true_amp_freq)
        )

        return loss, pha_dist, amp_dist

    # Initial loss calculation
    with torch.no_grad():
        initial_loss, initial_pha_dist, initial_amp_dist = calc_losses()
        losses.append(initial_loss.item())
        pha_distances.append(initial_pha_dist)
        amp_distances.append(initial_amp_dist)

    print(
        f"Initial - Loss: {initial_loss:.4f}, Phase Dist: {initial_pha_dist:.2f} Hz, Amp Dist: {initial_amp_dist:.2f} Hz"
    )

    # Training epochs
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # Forward pass and loss calculation
        loss, pha_dist, amp_dist = calc_losses()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Store metrics
        losses.append(loss.item())
        pha_distances.append(pha_dist)
        amp_distances.append(amp_dist)
        pha_trajectories[epoch] = model.PHA_MIDS_HZ.detach().cpu().numpy()
        amp_trajectories[epoch] = model.AMP_MIDS_HZ.detach().cpu().numpy()

        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f}, Phase Dist: {pha_dist:.2f} Hz, Amp Dist: {amp_dist:.2f} Hz"
            )

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # Final parameter values
    final_pha_mids = model.PHA_MIDS_HZ.detach().cpu().numpy()
    final_amp_mids = model.AMP_MIDS_HZ.detach().cpu().numpy()

    # Calculate best matching bands
    best_pha_idx = np.argmin(np.abs(final_pha_mids - true_pha_freq))
    best_amp_idx = np.argmin(np.abs(final_amp_mids - true_amp_freq))

    best_pha_freq = final_pha_mids[best_pha_idx]
    best_amp_freq = final_amp_mids[best_amp_idx]

    print(
        f"\nBest matching phase band: {best_pha_freq:.2f} Hz (true: {true_pha_freq:.2f} Hz)"
    )
    print(
        f"Best matching amplitude band: {best_amp_freq:.2f} Hz (true: {true_amp_freq:.2f} Hz)"
    )
    print(
        f"Phase frequency error: {abs(best_pha_freq - true_pha_freq):.2f} Hz"
    )
    print(
        f"Amplitude frequency error: {abs(best_amp_freq - true_amp_freq):.2f} Hz"
    )

    # Return results
    results = {
        "losses": losses,
        "pha_distances": pha_distances,
        "amp_distances": amp_distances,
        "pha_trajectories": pha_trajectories,
        "amp_trajectories": amp_trajectories,
        "initial_pha_mids": initial_pha_mids,
        "initial_amp_mids": initial_amp_mids,
        "final_pha_mids": final_pha_mids,
        "final_amp_mids": final_amp_mids,
        "true_pha_freq": true_pha_freq,
        "true_amp_freq": true_amp_freq,
        "best_pha_freq": best_pha_freq,
        "best_amp_freq": best_amp_freq,
        "train_time": train_time,
    }

    return results


def plot_results(results, output_dir):
    """Create plots visualizing the learning results."""
    # Unpack results
    losses = results["losses"]
    pha_distances = results["pha_distances"]
    amp_distances = results["amp_distances"]
    pha_trajectories = results["pha_trajectories"]
    amp_trajectories = results["amp_trajectories"]
    true_pha_freq = results["true_pha_freq"]
    true_amp_freq = results["true_amp_freq"]

    # Create figures directory
    figures_dir = output_dir
    figures_dir.mkdir(exist_ok=True)

    # 1. Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("PAC Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(figures_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 2. Distance to true frequencies
    plt.figure(figsize=(10, 6))
    plt.plot(pha_distances, label=f"Phase Distance to {true_pha_freq} Hz")
    plt.plot(amp_distances, label=f"Amplitude Distance to {true_amp_freq} Hz")
    plt.title("Distance to True Frequencies During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Distance (Hz)")
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_dir / "frequency_distances.png", dpi=300)
    plt.close()

    # 3. Parameter trajectories for phase bands
    plt.figure(figsize=(12, 8))
    epochs = range(pha_trajectories.shape[0])
    for i in range(pha_trajectories.shape[1]):
        plt.plot(epochs, pha_trajectories[:, i], label=f"Band {i+1}")

    plt.axhline(
        y=true_pha_freq,
        color="r",
        linestyle="--",
        label=f"True: {true_pha_freq} Hz",
    )
    plt.title("Phase Frequency Bands Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_dir / "phase_trajectories.png", dpi=300)
    plt.close()

    # 4. Parameter trajectories for amplitude bands
    plt.figure(figsize=(12, 8))
    for i in range(amp_trajectories.shape[1]):
        plt.plot(epochs, amp_trajectories[:, i], label=f"Band {i+1}")

    plt.axhline(
        y=true_amp_freq,
        color="r",
        linestyle="--",
        label=f"True: {true_amp_freq} Hz",
    )
    plt.title("Amplitude Frequency Bands Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_dir / "amplitude_trajectories.png", dpi=300)
    plt.close()

    # 5. Comparison of initial and final frequency bands
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Phase bands
    initial_pha = results["initial_pha_mids"]
    final_pha = results["final_pha_mids"]
    x_pha = np.arange(len(initial_pha))

    ax1.bar(x_pha - 0.2, initial_pha, width=0.4, label="Initial")
    ax1.bar(x_pha + 0.2, final_pha, width=0.4, label="Final")
    ax1.axhline(
        y=true_pha_freq,
        color="r",
        linestyle="--",
        label=f"True: {true_pha_freq} Hz",
    )
    ax1.set_title("Phase Frequency Bands Before and After Training")
    ax1.set_xlabel("Band Index")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.legend()
    ax1.grid(True)

    # Amplitude bands
    initial_amp = results["initial_amp_mids"]
    final_amp = results["final_amp_mids"]
    x_amp = np.arange(len(initial_amp))

    ax2.bar(x_amp - 0.2, initial_amp, width=0.4, label="Initial")
    ax2.bar(x_amp + 0.2, final_amp, width=0.4, label="Final")
    ax2.axhline(
        y=true_amp_freq,
        color="r",
        linestyle="--",
        label=f"True: {true_amp_freq} Hz",
    )
    ax2.set_title("Amplitude Frequency Bands Before and After Training")
    ax2.set_xlabel("Band Index")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(figures_dir / "frequency_bands_comparison.png", dpi=300)
    plt.close()


def save_results(results, output_dir):
    """Save numerical results to files."""
    import json

    import pandas as pd

    # Convert numerical results to lists for JSON serialization
    json_results = {
        "losses": [float(x) for x in results["losses"]],
        "pha_distances": [float(x) for x in results["pha_distances"]],
        "amp_distances": [float(x) for x in results["amp_distances"]],
        "true_pha_freq": float(results["true_pha_freq"]),
        "true_amp_freq": float(results["true_amp_freq"]),
        "best_pha_freq": float(results["best_pha_freq"]),
        "best_amp_freq": float(results["best_amp_freq"]),
        "train_time": float(results["train_time"]),
    }

    # Save summary metrics as JSON
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Save frequency trajectories as CSV
    pha_df = pd.DataFrame(
        results["pha_trajectories"],
        columns=[
            f"band_{i}" for i in range(results["pha_trajectories"].shape[1])
        ],
    )
    pha_df.to_csv(output_dir / "phase_trajectories.csv", index_label="epoch")

    amp_df = pd.DataFrame(
        results["amp_trajectories"],
        columns=[
            f"band_{i}" for i in range(results["amp_trajectories"].shape[1])
        ],
    )
    amp_df.to_csv(
        output_dir / "amplitude_trajectories.csv", index_label="epoch"
    )

    # Create a detailed report
    with open(output_dir / "learnability_report.md", "w") as f:
        f.write("# PAC Module Learnability Report\n\n")
        f.write(f"## Experiment Configuration\n\n")
        f.write(f"- True Phase Frequency: {results['true_pha_freq']:.2f} Hz\n")
        f.write(
            f"- True Amplitude Frequency: {results['true_amp_freq']:.2f} Hz\n"
        )
        f.write(
            f"- Initial Phase Bands: {', '.join([f'{x:.2f}' for x in results['initial_pha_mids']])} Hz\n"
        )
        f.write(
            f"- Initial Amplitude Bands: {', '.join([f'{x:.2f}' for x in results['initial_amp_mids']])} Hz\n\n"
        )

        f.write(f"## Results\n\n")
        f.write(
            f"- Best Matching Phase Band: {results['best_pha_freq']:.2f} Hz (error: {abs(results['best_pha_freq'] - results['true_pha_freq']):.2f} Hz)\n"
        )
        f.write(
            f"- Best Matching Amplitude Band: {results['best_amp_freq']:.2f} Hz (error: {abs(results['best_amp_freq'] - results['true_amp_freq']):.2f} Hz)\n"
        )
        f.write(f"- Final Loss: {results['losses'][-1]:.4f}\n")
        f.write(f"- Training Time: {results['train_time']:.2f} seconds\n\n")

        f.write(f"## Parameter Change\n\n")
        f.write("### Phase Bands\n\n")
        f.write("| Band | Initial (Hz) | Final (Hz) | Change (Hz) |\n")
        f.write("|------|--------------|------------|-------------|\n")
        for i, (init, final) in enumerate(
            zip(results["initial_pha_mids"], results["final_pha_mids"])
        ):
            f.write(
                f"| {i+1} | {init:.2f} | {final:.2f} | {final-init:.2f} |\n"
            )

        f.write("\n### Amplitude Bands\n\n")
        f.write("| Band | Initial (Hz) | Final (Hz) | Change (Hz) |\n")
        f.write("|------|--------------|------------|-------------|\n")
        for i, (init, final) in enumerate(
            zip(results["initial_amp_mids"], results["final_amp_mids"])
        ):
            f.write(
                f"| {i+1} | {init:.2f} | {final:.2f} | {final-init:.2f} |\n"
            )


def run_learnability_experiment():
    """Run a complete learnability experiment."""
    print("=" * 80)
    print("PAC MODULE LEARNABILITY EXPERIMENT")
    print("=" * 80)

    # Create output directory
    output_dir = create_output_dir()
    print(f"Results will be saved to: {output_dir}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic data with known PAC frequencies
    true_pha_freq = 8.0
    true_amp_freq = 100.0

    print(f"\nGenerating synthetic data with:")
    print(f"  Phase Frequency: {true_pha_freq} Hz")
    print(f"  Amplitude Frequency: {true_amp_freq} Hz")

    signal, pha_freq, amp_freq = create_synthetic_pac_signal(
        pha_freq=true_pha_freq, amp_freq=true_amp_freq, fs=1000.0, duration=2.0
    )

    # Train PAC module to learn parameters
    print("\nStarting parameter learning experiment...")
    results = learn_pac_parameters(
        signal=signal,
        true_pha_freq=pha_freq,
        true_amp_freq=amp_freq,
        num_epochs=100,
        learning_rate=0.01,
        device=device,
    )

    # Save and visualize results
    print("\nSaving results...")
    save_results(results, output_dir)
    plot_results(results, output_dir)

    print(
        f"\nExperiment completed successfully. Results saved to {output_dir}"
    )
    print("\nSummary:")
    print(f"  Initial Phase Error: {results['pha_distances'][0]:.2f} Hz")
    print(f"  Final Phase Error: {results['pha_distances'][-1]:.2f} Hz")
    print(f"  Initial Amplitude Error: {results['amp_distances'][0]:.2f} Hz")
    print(f"  Final Amplitude Error: {results['amp_distances'][-1]:.2f} Hz")

    return results


if __name__ == "__main__":
    run_learnability_experiment()

# EOF
