#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 02:18:07 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/learnability/learnability_multitask_loss/classify_binary_pac_multitask.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/learnability_multitask_loss/classify_binary_pac_multitask.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Classification of binary PAC signals using MultiTaskLoss to balance phase and amplitude learning.

This script:
1. Loads synthetic binary PAC signals
2. Implements a MultiTaskLoss approach to balance learning for phase and amplitude bands
3. Trains PAC classifiers with trainable frequency bands
4. Visualizes the convergence of phase and amplitude bands
5. Compares results with regular loss approaches

Dependencies:
- PyTorch
- Matplotlib
- NumPy
- gPAC
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root to Python path
project_root = Path(__FILE__).resolve().parents[3]
sys.path.append(str(project_root))

# Import gPAC modules
from gpac._pac import PAC


class MultiTaskLoss(nn.Module):
    """Uncertainty weighting for multitask learning from Kendall et al. (2017)."""

    def __init__(self, is_regression, reduction="none"):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression + 1) * (stds**2))
        multi_task_losses = coeffs * losses + torch.log(stds)

        if self.reduction == "sum":
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == "mean":
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


class PACDataset(Dataset):
    """Dataset for PAC signals with frequency labels."""

    def __init__(self, signals, pha_freqs, amp_freqs, noise_levels=None):
        self.signals = signals
        self.pha_freqs = pha_freqs
        self.amp_freqs = amp_freqs
        self.noise_levels = (
            noise_levels
            if noise_levels is not None
            else np.zeros(len(signals))
        )

        # Create unique frequency class labels
        self.unique_pha_freqs = np.unique(pha_freqs)
        self.unique_amp_freqs = np.unique(amp_freqs)

        # Map frequencies to class indices
        self.pha_class_map = {
            freq: i for i, freq in enumerate(self.unique_pha_freqs)
        }
        self.amp_class_map = {
            freq: i for i, freq in enumerate(self.unique_amp_freqs)
        }

        # Create class labels
        self.pha_classes = np.array(
            [self.pha_class_map[freq] for freq in pha_freqs]
        )
        self.amp_classes = np.array(
            [self.amp_class_map[freq] for freq in amp_freqs]
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]

        # Convert labels to torch tensors
        pha_class = torch.tensor(self.pha_classes[idx], dtype=torch.long)
        amp_class = torch.tensor(self.amp_classes[idx], dtype=torch.long)
        noise_level = torch.tensor(self.noise_levels[idx], dtype=torch.float32)

        return signal, pha_class, amp_class, noise_level


class PACClassifier(nn.Module):
    """Neural network for classifying PAC signals by frequency using MultiTaskLoss."""

    def __init__(self, pac_module, n_pha_classes, n_amp_classes):
        super().__init__()

        # PAC module for feature extraction
        self.pac_module = pac_module

        # Get output dimensions from PAC module
        self.pha_n_bands = pac_module.PHA_MIDS_HZ.shape[0]
        self.amp_n_bands = pac_module.AMP_MIDS_HZ.shape[0]

        # Total number of PAC values (pha_bands × amp_bands)
        n_features = self.pha_n_bands * self.amp_n_bands

        # Classification heads
        self.pha_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_pha_classes),
        )

        self.amp_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_amp_classes),
        )

        # Add multi-task loss - both tasks are classification (False)
        self.multi_task_loss = MultiTaskLoss(
            is_regression=torch.tensor([False, False]), reduction="mean"
        )

    def forward(self, x):
        # Extract PAC features - shape (B, C, F_pha, F_amp)
        pac_values = self.pac_module(x)

        # Check shape and ensure it's as expected
        batch_size = pac_values.shape[0]

        # Handle channel dimension if present
        if len(pac_values.shape) == 4:  # (B, C, F_pha, F_amp)
            # Average across channels if there are multiple channels
            if pac_values.shape[1] > 1:
                pac_values = pac_values.mean(dim=1)  # (B, F_pha, F_amp)
            else:
                pac_values = pac_values.squeeze(
                    1
                )  # Remove channel dim if only one channel

        # Reshape PAC values to match the expected input for the classifiers
        # From (B, F_pha, F_amp) to (B, F_pha*F_amp)
        pac_values_flat = pac_values.reshape(batch_size, -1)

        # Verify the flattened shape matches our expected n_features
        expected_features = self.pha_n_bands * self.amp_n_bands
        if pac_values_flat.shape[1] != expected_features:
            raise ValueError(
                f"Flattened PAC values shape mismatch. Got {pac_values_flat.shape[1]} features, "
                f"expected {expected_features} (pha_bands={self.pha_n_bands} × amp_bands={self.amp_n_bands})"
            )

        # Classify phase and amplitude frequencies
        pha_logits = self.pha_classifier(pac_values_flat)
        amp_logits = self.amp_classifier(pac_values_flat)

        return pha_logits, amp_logits

    def get_frequency_bands(self):
        """Get current frequency band values."""
        return {
            "pha_bands": self.pac_module.PHA_MIDS_HZ.detach().cpu().numpy(),
            "amp_bands": self.pac_module.AMP_MIDS_HZ.detach().cpu().numpy(),
        }


def load_synthetic_data(data_path):
    """Load synthetic PAC data."""
    print(f"Loading data from {data_path}")

    # Check if file exists
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data with weights_only=False to handle numpy arrays
    data = torch.load(data_path, weights_only=False)

    signals = data["signals"]
    metadata = data["metadata"]
    params = data["params"]

    # Print information about the loaded data
    print(f"Loaded {len(signals)} signals with shape {signals.shape}")
    print(f"Phase frequencies: {np.unique(metadata['pha_freqs'])}")
    print(f"Amplitude frequencies: {np.unique(metadata['amp_freqs'])}")
    print(f"Noise levels: {np.unique(metadata['noise_levels'])}")

    return signals, metadata, params


def split_train_test(signals, metadata, test_ratio=0.2):
    """Split data into training and testing sets."""
    # Get size
    n_samples = len(signals)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    # Create random indices for splitting
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
    train_signals = signals[train_indices]
    train_pha_freqs = metadata["pha_freqs"][train_indices]
    train_amp_freqs = metadata["amp_freqs"][train_indices]
    train_noise_levels = metadata["noise_levels"][train_indices]

    test_signals = signals[test_indices]
    test_pha_freqs = metadata["pha_freqs"][test_indices]
    test_amp_freqs = metadata["amp_freqs"][test_indices]
    test_noise_levels = metadata["noise_levels"][test_indices]

    # Create datasets
    train_dataset = PACDataset(
        train_signals, train_pha_freqs, train_amp_freqs, train_noise_levels
    )
    test_dataset = PACDataset(
        test_signals, test_pha_freqs, test_amp_freqs, test_noise_levels
    )

    print(f"Data split: {n_train} training samples, {n_test} testing samples")

    return train_dataset, test_dataset


def create_model(
    seq_len,
    fs,
    pha_n_bands,
    amp_n_bands,
    n_pha_classes,
    n_amp_classes,
    trainable=True,
    device="cpu",
):
    """Create PAC classifier model."""
    # Create PAC module
    pac_module = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=pha_n_bands,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=amp_n_bands,
        trainable=trainable,
    ).to(device)

    # Create classifier
    return PACClassifier(
        pac_module=pac_module,
        n_pha_classes=n_pha_classes,
        n_amp_classes=n_amp_classes,
    ).to(device)


def track_band_convergence(
    model, train_loader, n_epochs, learning_rate, device, track_interval=5
):
    """Train model while tracking frequency band evolution and task weights."""
    # Move model to device
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Include MultiTaskLoss parameters in the optimizer
    params = list(model.parameters()) + list(
        model.multi_task_loss.parameters()
    )
    optimizer = optim.Adam(params, lr=learning_rate)

    # Initialize tracking data
    history = {
        "train_loss": [],
        "pha_acc": [],
        "amp_acc": [],
        "task_weights": [],
        "pha_bands": [],
        "amp_bands": [],
    }

    # Store initial band values
    bands = model.get_frequency_bands()
    history["pha_bands"].append(bands["pha_bands"])
    history["amp_bands"].append(bands["amp_bands"])

    # Main training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct_pha = 0
        correct_amp = 0
        total = 0

        for signals, pha_classes, amp_classes, _ in train_loader:
            # Move data to device
            signals = signals.to(device)
            pha_classes = pha_classes.to(device)
            amp_classes = amp_classes.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pha_logits, amp_logits = model(signals)

            # Calculate individual losses (reduce=none to get per-sample losses)
            batch_size = signals.size(0)
            pha_losses = criterion(pha_logits, pha_classes)
            amp_losses = criterion(amp_logits, amp_classes)

            # Mean over batch dimension to get scalar loss for each task
            pha_loss = pha_losses.mean()
            amp_loss = amp_losses.mean()

            # Stack losses for multi-task loss
            losses = torch.stack([pha_loss, amp_loss])

            # Apply multi-task loss
            loss = model.multi_task_loss(losses)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * batch_size
            _, pha_preds = torch.max(pha_logits, 1)
            _, amp_preds = torch.max(amp_logits, 1)
            correct_pha += (pha_preds == pha_classes).sum().item()
            correct_amp += (amp_preds == amp_classes).sum().item()
            total += batch_size

        # Epoch statistics
        epoch_loss = running_loss / total
        epoch_pha_acc = correct_pha / total
        epoch_amp_acc = correct_amp / total

        # Record metrics
        history["train_loss"].append(epoch_loss)
        history["pha_acc"].append(epoch_pha_acc)
        history["amp_acc"].append(epoch_amp_acc)

        # Calculate and record task weights
        log_vars = model.multi_task_loss.log_vars.detach().cpu()
        stds = torch.exp(log_vars) ** (1 / 2)
        weights = 1 / (stds**2)
        history["task_weights"].append(weights.numpy())

        # Print progress
        print(
            f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, "
            f"Phase Acc: {epoch_pha_acc:.4f}, Amp Acc: {epoch_amp_acc:.4f}"
        )
        print(
            f"Task weights - Phase: {weights[0]:.4f}, Amplitude: {weights[1]:.4f}"
        )

        # Track parameter evolution at intervals
        if epoch % track_interval == 0 or epoch == n_epochs - 1:
            bands = model.get_frequency_bands()
            history["pha_bands"].append(bands["pha_bands"])
            history["amp_bands"].append(bands["amp_bands"])

    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct_pha = 0
    correct_amp = 0
    total = 0

    # Results by noise level
    noise_results = {}

    with torch.no_grad():
        for signals, pha_classes, amp_classes, noise_levels in test_loader:
            signals = signals.to(device)
            pha_classes = pha_classes.to(device)
            amp_classes = amp_classes.to(device)

            # Forward pass
            pha_logits, amp_logits = model(signals)

            # Get predictions
            _, pha_preds = torch.max(pha_logits, 1)
            _, amp_preds = torch.max(amp_logits, 1)

            # Update total correct
            correct_pha += (pha_preds == pha_classes).sum().item()
            correct_amp += (amp_preds == amp_classes).sum().item()
            total += signals.size(0)

            # Track per-noise level results
            for i, noise in enumerate(noise_levels):
                noise_val = float(noise.item())
                if noise_val not in noise_results:
                    noise_results[noise_val] = {
                        "total": 0,
                        "pha_correct": 0,
                        "amp_correct": 0,
                    }

                noise_results[noise_val]["total"] += 1
                noise_results[noise_val]["pha_correct"] += (
                    pha_preds[i] == pha_classes[i]
                ).item()
                noise_results[noise_val]["amp_correct"] += (
                    amp_preds[i] == amp_classes[i]
                ).item()

    # Calculate overall metrics
    pha_acc = correct_pha / total
    amp_acc = correct_amp / total

    # Calculate per-noise level metrics
    noise_accuracies = {}
    for noise, stats in noise_results.items():
        pha_acc_noise = stats["pha_correct"] / stats["total"]
        amp_acc_noise = stats["amp_correct"] / stats["total"]
        noise_accuracies[noise] = {
            "pha_acc": pha_acc_noise,
            "amp_acc": amp_acc_noise,
        }

    return {
        "pha_acc": pha_acc,
        "amp_acc": amp_acc,
        "noise_accuracies": noise_accuracies,
    }


def visualize_convergence(history, ground_truth=None, output_dir=None):
    """Create visualizations showing parameter convergence and task weights."""
    plt.figure(figsize=(18, 12))

    # Plot 1: Accuracy over time
    plt.subplot(2, 2, 1)
    plt.plot(history["pha_acc"], label="Phase Accuracy")
    plt.plot(history["amp_acc"], label="Amplitude Accuracy")
    plt.title("Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Task weights over time
    plt.subplot(2, 2, 2)
    task_weights = np.array(history["task_weights"])
    epochs = np.arange(len(task_weights))
    plt.plot(epochs, task_weights[:, 0], "b-", label="Phase Task Weight")
    plt.plot(epochs, task_weights[:, 1], "r-", label="Amplitude Task Weight")
    plt.title("Multi-Task Loss Weights")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 3: Phase bands evolution
    plt.subplot(2, 2, 3)
    pha_bands = history["pha_bands"]
    colormap = plt.cm.viridis
    # Create color steps based on number of snapshots
    colors = [colormap(i) for i in np.linspace(0, 1, len(pha_bands))]

    for i, bands in enumerate(pha_bands):
        alpha = 0.3 if i < len(pha_bands) - 1 else 1.0
        linewidth = 1 if i < len(pha_bands) - 1 else 2
        label = (
            "Initial"
            if i == 0
            else ("Final" if i == len(pha_bands) - 1 else f"Epoch {i*5}")
        )
        plt.plot(
            np.arange(len(bands)),
            bands,
            "o-",
            color=colors[i],
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    # Add ground truth bands if provided
    if ground_truth and "pha_bands" in ground_truth:
        plt.axhspan(
            ground_truth["pha_bands"][0],
            ground_truth["pha_bands"][1],
            alpha=0.2,
            color="blue",
            label="Ground Truth",
        )

    plt.title("Phase Frequency Band Evolution")
    plt.xlabel("Band Index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 4: Amplitude bands evolution
    plt.subplot(2, 2, 4)
    amp_bands = history["amp_bands"]

    for i, bands in enumerate(amp_bands):
        alpha = 0.3 if i < len(amp_bands) - 1 else 1.0
        linewidth = 1 if i < len(amp_bands) - 1 else 2
        label = (
            "Initial"
            if i == 0
            else ("Final" if i == len(amp_bands) - 1 else f"Epoch {i*5}")
        )
        plt.plot(
            np.arange(len(bands)),
            bands,
            "o-",
            color=colors[i],
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    # Add ground truth bands if provided
    if ground_truth and "amp_bands" in ground_truth:
        plt.axhspan(
            ground_truth["amp_bands"][0],
            ground_truth["amp_bands"][1],
            alpha=0.2,
            color="red",
            label="Ground Truth",
        )

    plt.title("Amplitude Frequency Band Evolution")
    plt.xlabel("Band Index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure if output directory provided
    if output_dir:
        output_path = Path(output_dir) / "convergence_visualization.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Convergence visualization saved to {output_path}")
    else:
        plt.show()

    # Create convergence animation (optional)
    if len(pha_bands) > 3:
        create_band_convergence_animation(
            pha_bands, amp_bands, ground_truth, output_dir
        )


def create_band_convergence_animation(
    pha_bands, amp_bands, ground_truth=None, output_dir=None
):
    """Create animated visualization of band parameter convergence."""
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Setup phase plot
    ax1.set_title("Phase Band Convergence")
    ax1.set_xlabel("Band Index")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.grid(alpha=0.3)

    # Add ground truth for phase
    if ground_truth and "pha_bands" in ground_truth:
        ax1.axhspan(
            ground_truth["pha_bands"][0],
            ground_truth["pha_bands"][1],
            alpha=0.2,
            color="blue",
            label="Target Region",
        )

    # Setup amplitude plot
    ax2.set_title("Amplitude Band Convergence")
    ax2.set_xlabel("Band Index")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.grid(alpha=0.3)

    # Add ground truth for amplitude
    if ground_truth and "amp_bands" in ground_truth:
        ax2.axhspan(
            ground_truth["amp_bands"][0],
            ground_truth["amp_bands"][1],
            alpha=0.2,
            color="red",
            label="Target Region",
        )

    # Initial plot
    (pha_line,) = ax1.plot([], [], "bo-", label="Phase Bands")
    (amp_line,) = ax2.plot([], [], "ro-", label="Amplitude Bands")

    # Add frame counter
    frame_text = fig.text(0.02, 0.02, "", fontsize=10)

    # Set fixed y-axis limits based on min/max values
    pha_min = min([np.min(band) for band in pha_bands]) * 0.9
    pha_max = max([np.max(band) for band in pha_bands]) * 1.1
    amp_min = min([np.min(band) for band in amp_bands]) * 0.9
    amp_max = max([np.max(band) for band in amp_bands]) * 1.1

    ax1.set_ylim(pha_min, pha_max)
    ax2.set_ylim(amp_min, amp_max)

    # Setup x-axis
    pha_indices = np.arange(len(pha_bands[0]))
    amp_indices = np.arange(len(amp_bands[0]))
    ax1.set_xlim(-1, len(pha_indices))
    ax2.set_xlim(-1, len(amp_indices))

    # Add legends
    ax1.legend()
    ax2.legend()

    def init():
        pha_line.set_data([], [])
        amp_line.set_data([], [])
        frame_text.set_text("")
        return pha_line, amp_line, frame_text

    def update(frame):
        # Get snapshot data
        if frame < len(pha_bands):
            pha_data = pha_bands[frame]
            amp_data = amp_bands[frame]
            frame_label = f"Snapshot {frame}"
        else:
            # Show final snapshot if beyond available data
            pha_data = pha_bands[-1]
            amp_data = amp_bands[-1]
            frame_label = "Final State"

        pha_line.set_data(pha_indices, pha_data)
        amp_line.set_data(amp_indices, amp_data)
        frame_text.set_text(frame_label)

        return pha_line, amp_line, frame_text

    # Create animation
    frames = len(pha_bands)
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=500
    )

    # Save animation if output directory provided
    if output_dir:
        output_path = Path(output_dir) / "band_convergence.gif"
        ani.save(output_path, writer="pillow", fps=2)
        print(f"Band convergence animation saved to {output_path}")

    plt.tight_layout()
    plt.close()


def compare_standard_vs_multitask(
    standard_history, multitask_history, output_dir
):
    """Compare standard training with multitask training."""
    plt.figure(figsize=(18, 10))

    # Phase Accuracy Comparison
    plt.subplot(2, 2, 1)
    plt.plot(standard_history["pha_acc"], "b--", label="Standard - Phase")
    plt.plot(multitask_history["pha_acc"], "b-", label="Multitask - Phase")
    plt.title("Phase Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # Amplitude Accuracy Comparison
    plt.subplot(2, 2, 2)
    plt.plot(standard_history["amp_acc"], "r--", label="Standard - Amplitude")
    plt.plot(multitask_history["amp_acc"], "r-", label="Multitask - Amplitude")
    plt.title("Amplitude Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # Phase Band Final States
    plt.subplot(2, 2, 3)
    plt.plot(
        np.arange(len(standard_history["pha_bands"][-1])),
        standard_history["pha_bands"][-1],
        "bo--",
        label="Standard",
    )
    plt.plot(
        np.arange(len(multitask_history["pha_bands"][-1])),
        multitask_history["pha_bands"][-1],
        "bs-",
        label="Multitask",
    )

    plt.title("Final Phase Band States")
    plt.xlabel("Band Index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Amplitude Band Final States
    plt.subplot(2, 2, 4)
    plt.plot(
        np.arange(len(standard_history["amp_bands"][-1])),
        standard_history["amp_bands"][-1],
        "ro--",
        label="Standard",
    )
    plt.plot(
        np.arange(len(multitask_history["amp_bands"][-1])),
        multitask_history["amp_bands"][-1],
        "rs-",
        label="Multitask",
    )

    plt.title("Final Amplitude Band States")
    plt.xlabel("Band Index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save comparison figure
    output_path = Path(output_dir) / "standard_vs_multitask_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Comparison visualization saved to {output_path}")

    # Also generate a report
    report_path = Path(output_dir) / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Standard vs MultiTask Training Comparison\n\n")

        f.write("## Overall Accuracy\n\n")
        f.write("| Method | Phase Accuracy | Amplitude Accuracy |\n")
        f.write("|--------|---------------|-------------------|\n")
        f.write(
            f"| Standard | {standard_history['pha_acc'][-1]:.4f} | {standard_history['amp_acc'][-1]:.4f} |\n"
        )
        f.write(
            f"| Multitask | {multitask_history['pha_acc'][-1]:.4f} | {multitask_history['amp_acc'][-1]:.4f} |\n\n"
        )

        # Calculate improvement
        pha_improvement = (
            (
                multitask_history["pha_acc"][-1]
                - standard_history["pha_acc"][-1]
            )
            / standard_history["pha_acc"][-1]
            * 100
        )
        amp_improvement = (
            (
                multitask_history["amp_acc"][-1]
                - standard_history["amp_acc"][-1]
            )
            / standard_history["amp_acc"][-1]
            * 100
        )

        f.write(f"**Phase accuracy improvement: {pha_improvement:+.2f}%**\n\n")
        f.write(
            f"**Amplitude accuracy improvement: {amp_improvement:+.2f}%**\n\n"
        )

        f.write("## Band Learning Comparison\n\n")

        # Calculate band changes (how much they moved from initial)
        std_pha_change = np.abs(
            standard_history["pha_bands"][-1]
            - standard_history["pha_bands"][0]
        ).mean()
        std_amp_change = np.abs(
            standard_history["amp_bands"][-1]
            - standard_history["amp_bands"][0]
        ).mean()
        mt_pha_change = np.abs(
            multitask_history["pha_bands"][-1]
            - multitask_history["pha_bands"][0]
        ).mean()
        mt_amp_change = np.abs(
            multitask_history["amp_bands"][-1]
            - multitask_history["amp_bands"][0]
        ).mean()

        f.write("### Average Band Parameter Change (Hz)\n\n")
        f.write(
            "| Method | Phase Bands | Amplitude Bands | Ratio (Phase/Amp) |\n"
        )
        f.write(
            "|--------|-------------|----------------|------------------|\n"
        )
        std_ratio = std_pha_change / (
            std_amp_change if std_amp_change > 0 else 1
        )
        mt_ratio = mt_pha_change / (mt_amp_change if mt_amp_change > 0 else 1)
        f.write(
            f"| Standard | {std_pha_change:.2f} Hz | {std_amp_change:.2f} Hz | {std_ratio:.4f} |\n"
        )
        f.write(
            f"| Multitask | {mt_pha_change:.2f} Hz | {mt_amp_change:.2f} Hz | {mt_ratio:.4f} |\n\n"
        )

        ratio_improvement = (mt_ratio - std_ratio) / std_ratio * 100
        f.write(
            f"**Phase/Amplitude ratio improvement: {ratio_improvement:+.2f}%**\n\n"
        )

        f.write("## Task Weights Evolution (MultiTask Only)\n\n")
        f.write("| Epoch | Phase Weight | Amplitude Weight |\n")
        f.write("|-------|--------------|------------------|\n")

        task_weights = multitask_history["task_weights"]
        epochs_to_show = [
            0,
            len(task_weights) // 4,
            len(task_weights) // 2,
            3 * len(task_weights) // 4,
            len(task_weights) - 1,
        ]

        for i in epochs_to_show:
            f.write(
                f"| {i} | {task_weights[i][0]:.4f} | {task_weights[i][1]:.4f} |\n"
            )

        f.write("\n## Conclusion\n\n")
        if mt_ratio > std_ratio:
            f.write(
                "The MultiTaskLoss approach **successfully balanced** the learning between phase and amplitude bands. "
            )
            f.write(
                "Phase bands show more movement compared to the standard approach, leading to improved accuracy and "
            )
            f.write("a more balanced optimization process.")
        else:
            f.write(
                "The MultiTaskLoss approach had an unexpected effect on the learning dynamics. "
            )
            f.write(
                "The standard approach showed a better balance between phase and amplitude learning."
            )

    print(f"Comparison report saved to {report_path}")


def run_standard_training(
    train_loader, test_loader, params, device, output_dir
):
    """Run training with standard loss."""
    print("\n" + "=" * 50)
    print("Standard Training (Cross-Entropy Loss Only)")
    print("=" * 50)

    # Create model
    model = create_model(
        seq_len=params["seq_len"],
        fs=params["fs"],
        pha_n_bands=params["pha_n_bands"],
        amp_n_bands=params["amp_n_bands"],
        n_pha_classes=len(train_loader.dataset.unique_pha_freqs),
        n_amp_classes=len(train_loader.dataset.unique_amp_freqs),
        trainable=True,
        device=device,
    )

    # Remove multitask loss for standard training
    # We'll keep it in the model but redefine the forward pass to not use it
    original_forward = model.forward

    def standard_forward(x):
        pha_logits, amp_logits = original_forward(x)
        return pha_logits, amp_logits

    # Temporarily override forward method
    model.forward = standard_forward

    # Training function for standard loss
    def train_standard_model(
        model, train_loader, n_epochs, learning_rate, device, track_interval=5
    ):
        # Move model to device
        model = model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize tracking data
        history = {
            "train_loss": [],
            "pha_acc": [],
            "amp_acc": [],
            "pha_bands": [],
            "amp_bands": [],
            "task_weights": [],  # Keep for compatibility
        }

        # Store initial band values
        bands = model.get_frequency_bands()
        history["pha_bands"].append(bands["pha_bands"])
        history["amp_bands"].append(bands["amp_bands"])

        # Placeholder for task weights (not used in standard training)
        history["task_weights"] = np.zeros((n_epochs, 2))

        # Main training loop
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            correct_pha = 0
            correct_amp = 0
            total = 0

            for signals, pha_classes, amp_classes, _ in train_loader:
                # Move data to device
                signals = signals.to(device)
                pha_classes = pha_classes.to(device)
                amp_classes = amp_classes.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                pha_logits, amp_logits = model(signals)

                # Calculate loss (standard weighted sum)
                pha_loss = criterion(pha_logits, pha_classes)
                amp_loss = criterion(amp_logits, amp_classes)
                loss = pha_loss + amp_loss  # Equal weighting

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * signals.size(0)
                _, pha_preds = torch.max(pha_logits, 1)
                _, amp_preds = torch.max(amp_logits, 1)
                correct_pha += (pha_preds == pha_classes).sum().item()
                correct_amp += (amp_preds == amp_classes).sum().item()
                total += signals.size(0)

            # Epoch statistics
            epoch_loss = running_loss / total
            epoch_pha_acc = correct_pha / total
            epoch_amp_acc = correct_amp / total

            # Record metrics
            history["train_loss"].append(epoch_loss)
            history["pha_acc"].append(epoch_pha_acc)
            history["amp_acc"].append(epoch_amp_acc)

            # Print progress
            print(
                f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, "
                f"Phase Acc: {epoch_pha_acc:.4f}, Amp Acc: {epoch_amp_acc:.4f}"
            )

            # Track parameter evolution at intervals
            if epoch % track_interval == 0 or epoch == n_epochs - 1:
                bands = model.get_frequency_bands()
                history["pha_bands"].append(bands["pha_bands"])
                history["amp_bands"].append(bands["amp_bands"])

        return history

    # Train with standard approach
    standard_history = train_standard_model(
        model=model,
        train_loader=train_loader,
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        device=device,
        track_interval=params["track_interval"],
    )

    # Evaluate
    standard_results = evaluate_model(model, test_loader, device)
    print("\nStandard Training Results:")
    print(f"Phase Accuracy: {standard_results['pha_acc']:.4f}")
    print(f"Amplitude Accuracy: {standard_results['amp_acc']:.4f}")

    # Visualize
    standard_output_dir = Path(output_dir) / "standard"
    standard_output_dir.mkdir(exist_ok=True, parents=True)

    ground_truth = {
        "pha_bands": (4.0, 14.0),  # Ground truth ranges from synthetic data
        "amp_bands": (80.0, 150.0),
    }

    visualize_convergence(
        standard_history,
        ground_truth=ground_truth,
        output_dir=standard_output_dir,
    )

    return standard_history, standard_results


def run_multitask_training(
    train_loader, test_loader, params, device, output_dir
):
    """Run training with multitask loss."""
    print("\n" + "=" * 50)
    print("MultiTask Training (Uncertainty Weighting)")
    print("=" * 50)

    # Create model
    model = create_model(
        seq_len=params["seq_len"],
        fs=params["fs"],
        pha_n_bands=params["pha_n_bands"],
        amp_n_bands=params["amp_n_bands"],
        n_pha_classes=len(train_loader.dataset.unique_pha_freqs),
        n_amp_classes=len(train_loader.dataset.unique_amp_freqs),
        trainable=True,
        device=device,
    )

    # Train with multitask approach
    multitask_history = track_band_convergence(
        model=model,
        train_loader=train_loader,
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        device=device,
        track_interval=params["track_interval"],
    )

    # Evaluate
    multitask_results = evaluate_model(model, test_loader, device)
    print("\nMultiTask Training Results:")
    print(f"Phase Accuracy: {multitask_results['pha_acc']:.4f}")
    print(f"Amplitude Accuracy: {multitask_results['amp_acc']:.4f}")

    # Visualize
    multitask_output_dir = Path(output_dir) / "multitask"
    multitask_output_dir.mkdir(exist_ok=True, parents=True)

    ground_truth = {
        "pha_bands": (4.0, 14.0),  # Ground truth ranges from synthetic data
        "amp_bands": (80.0, 150.0),
    }

    visualize_convergence(
        multitask_history,
        ground_truth=ground_truth,
        output_dir=multitask_output_dir,
    )

    return multitask_history, multitask_results


def main(args):
    """Main function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(__DIR__) / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    data_path = Path(__DIR__).parent / "data" / "synthetic_pac_signals.pt"
    if args.data_path:
        data_path = Path(args.data_path)

    signals, metadata, data_params = load_synthetic_data(data_path)

    # Parameters
    params = {
        "seq_len": signals.shape[-1],
        "fs": data_params["fs"],
        "pha_n_bands": args.pha_n_bands,
        "amp_n_bands": args.amp_n_bands,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "test_ratio": args.test_ratio,
        "track_interval": args.track_interval,
    }

    # Split data
    train_dataset, test_dataset = split_train_test(
        signals, metadata, test_ratio=params["test_ratio"]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params["batch_size"], shuffle=False
    )

    # Run standard training
    standard_history, standard_results = run_standard_training(
        train_loader, test_loader, params, device, output_dir
    )

    # Run multitask training
    multitask_history, multitask_results = run_multitask_training(
        train_loader, test_loader, params, device, output_dir
    )

    # Compare results
    compare_standard_vs_multitask(
        standard_history, multitask_history, output_dir
    )

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train binary PAC classifiers with MultiTaskLoss"
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to synthetic PAC signals data file"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--pha_n_bands",
        type=int,
        default=10,
        help="Number of phase frequency bands",
    )
    parser.add_argument(
        "--amp_n_bands",
        type=int,
        default=10,
        help="Number of amplitude frequency bands",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Portion of data to use for testing",
    )
    parser.add_argument(
        "--track_interval",
        type=int,
        default=5,
        help="Epoch interval for tracking band changes",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# EOF
