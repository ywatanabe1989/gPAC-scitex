#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 01:35:04 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/visualize_pac_features.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/visualize_pac_features.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Visualizes PAC features and their importance for classification
  - Provides comprehensive analysis of PAC patterns in binary classification
  - Generates publication-quality visualizations of discriminative features
  - Analyzes evolution of PAC parameters during training
  - Compares trainable vs. fixed PAC performance

Dependencies:
  - packages:
    - torch
    - numpy
    - matplotlib
    - seaborn
    - pandas
    - scipy
    - mngs
    - gpac

IO:
  - input-files:
    - ./scripts/learnability/data/binary_pac_signals.pt
    - ./scripts/learnability/results/binary_figures/*
  - output-files:
    - ./scripts/learnability/results/visualizations/
"""

"""Imports"""
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))

from gpac._pac import PAC


"""Parameters"""
DEFAULT_PARAMS = {
    "data_path": "./scripts/learnability/data/binary_pac_signals.pt",
    "results_dir": "./scripts/learnability/results/visualizations",
    "pha_n_bands": 8,
    "amp_n_bands": 8,
    "save_figs": True,
    "dpi": 300,
}


"""Functions & Classes"""
def load_binary_data(data_path: str) -> Tuple[torch.Tensor, np.ndarray, Dict, Dict]:
    """Load binary PAC data."""
    print(f"Loading data from {data_path}")
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = torch.load(data_path, weights_only=False)
    
    signals = data["signals"]
    labels = data["metadata"]["class_labels"]
    metadata = data["metadata"]
    params = data["params"]
    
    print(f"Loaded {len(signals)} signals with shape {signals.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return signals, labels, metadata, params


def create_pac_feature_extractor(
    seq_len: int,
    fs: float,
    pha_n_bands: int,
    amp_n_bands: int,
    trainable: bool = False,
) -> nn.Module:
    """Create a PAC feature extractor."""
    # PAC parameters
    pac_module = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=1.0,
        pha_end_hz=20.0,
        pha_n_bands=pha_n_bands,
        amp_start_hz=60.0,
        amp_end_hz=180.0,
        amp_n_bands=amp_n_bands,
        trainable=trainable,
    )
    
    return pac_module


def extract_pac_features(
    pac_module: nn.Module, 
    signals: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Extract PAC features from signals."""
    # Move to device
    signals = signals.to(device)
    pac_module = pac_module.to(device)
    
    # Process in evaluation mode
    pac_module.eval()
    
    # Extract features with no gradients
    with torch.no_grad():
        pac_values = pac_module(signals)
        
        # Handle channel dimension if present
        if len(pac_values.shape) == 4:  # (B, C, F_pha, F_amp)
            # Average across channels if multiple
            if pac_values.shape[1] > 1:
                pac_values = pac_values.mean(dim=1)  # (B, F_pha, F_amp)
            else:
                pac_values = pac_values.squeeze(1)  # Remove channel dim if only one
    
    return pac_values


def visualize_class_examples(
    signals: torch.Tensor, 
    labels: np.ndarray,
    metadata: Dict,
    params: Dict,
    output_dir: Path,
    save: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Visualize examples from each class."""
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Example Signals by Class", fontsize=16)
    
    # Get class-specific indices
    class_a_indices = np.where(labels == 0)[0]
    class_b_indices = np.where(labels == 1)[0]
    
    # Select random examples from each class
    np.random.seed(42)
    class_a_examples = np.random.choice(class_a_indices, 3, replace=False)
    class_b_examples = np.random.choice(class_b_indices, 3, replace=False)
    
    # Class A examples (top row)
    for i, ax in enumerate(axes[0]):
        idx = class_a_examples[i]
        signal = signals[idx, 0, 0].cpu().numpy()  # First channel, first segment
        
        # Time vector
        fs = params["fs"]
        duration = signal.shape[0] / fs
        time_vector = np.linspace(0, duration, len(signal))
        
        # Phase and amplitude frequencies
        pha_freq = metadata["pha_freqs"][idx]
        amp_freq = metadata["amp_freqs"][idx]
        noise = metadata["noise_levels"][idx]
        
        # Plot signal
        ax.plot(time_vector, signal)
        ax.set_title(f"Class A\nPhase: {pha_freq:.1f} Hz, Amp: {amp_freq:.1f} Hz\nNoise: {noise:.2f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
    
    # Class B examples (bottom row)
    for i, ax in enumerate(axes[1]):
        idx = class_b_examples[i]
        signal = signals[idx, 0, 0].cpu().numpy()  # First channel, first segment
        
        # Time vector
        fs = params["fs"]
        duration = signal.shape[0] / fs
        time_vector = np.linspace(0, duration, len(signal))
        
        # Phase and amplitude frequencies
        pha_freq = metadata["pha_freqs"][idx]
        amp_freq = metadata["amp_freqs"][idx]
        noise = metadata["noise_levels"][idx]
        
        # Plot signal
        ax.plot(time_vector, signal)
        ax.set_title(f"Class B\nPhase: {pha_freq:.1f} Hz, Amp: {amp_freq:.1f} Hz\nNoise: {noise:.2f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "class_examples.png", dpi=dpi)
    
    return fig


def visualize_pac_features(
    pac_values: torch.Tensor,
    labels: np.ndarray,
    pha_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    output_dir: Path,
    save: bool = True,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Figure]:
    """Visualize average PAC values for each class."""
    # Move to numpy
    pac_values_np = pac_values.cpu().numpy()
    
    # Get class-specific indices
    class_a_indices = np.where(labels == 0)[0]
    class_b_indices = np.where(labels == 1)[0]
    
    # Calculate class-specific means
    class_a_mean = pac_values_np[class_a_indices].mean(axis=0)
    class_b_mean = pac_values_np[class_b_indices].mean(axis=0)
    
    # Calculate difference
    class_diff = class_b_mean - class_a_mean
    
    # Create custom colormap for difference
    colors = [(0, 0, 0.8), (1, 1, 1), (0.8, 0, 0)]  # Blue -> White -> Red
    n_bins = 100
    cmap_diff = plt.cm.RdBu_r
    
    # Create figure for class means
    fig_means, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Class A mean
    im0 = axes[0].imshow(class_a_mean, cmap='viridis', aspect='auto', 
                         extent=[pha_freqs.min(), pha_freqs.max(), 
                                 amp_freqs.min(), amp_freqs.max()])
    axes[0].set_title("Class A - Mean PAC Values", fontsize=14)
    axes[0].set_xlabel("Phase Frequency (Hz)")
    axes[0].set_ylabel("Amplitude Frequency (Hz)")
    fig_means.colorbar(im0, ax=axes[0])
    
    # Class B mean
    im1 = axes[1].imshow(class_b_mean, cmap='viridis', aspect='auto',
                         extent=[pha_freqs.min(), pha_freqs.max(), 
                                 amp_freqs.min(), amp_freqs.max()])
    axes[1].set_title("Class B - Mean PAC Values", fontsize=14)
    axes[1].set_xlabel("Phase Frequency (Hz)")
    axes[1].set_ylabel("Amplitude Frequency (Hz)")
    fig_means.colorbar(im1, ax=axes[1])
    
    # Class difference
    im2 = axes[2].imshow(class_diff, cmap=cmap_diff, aspect='auto',
                         extent=[pha_freqs.min(), pha_freqs.max(), 
                                 amp_freqs.min(), amp_freqs.max()],
                         vmin=-np.abs(class_diff).max(), vmax=np.abs(class_diff).max())
    axes[2].set_title("Class B - Class A (Difference)", fontsize=14)
    axes[2].set_xlabel("Phase Frequency (Hz)")
    axes[2].set_ylabel("Amplitude Frequency (Hz)")
    fig_means.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if save:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "pac_class_means.png", dpi=dpi)
    
    # Create figure for discriminative features
    fig_discrim = plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(class_diff), cmap='hot', aspect='auto',
               extent=[pha_freqs.min(), pha_freqs.max(), 
                       amp_freqs.min(), amp_freqs.max()])
    plt.colorbar(label='|Class Difference|')
    plt.title("Discriminative PAC Features", fontsize=16)
    plt.xlabel("Phase Frequency (Hz)")
    plt.ylabel("Amplitude Frequency (Hz)")
    
    # Add contours to highlight regions
    plt.contour(np.abs(class_diff), levels=5, 
                extent=[pha_freqs.min(), pha_freqs.max(), 
                        amp_freqs.min(), amp_freqs.max()],
                colors='k', alpha=0.5, linewidths=0.5)
    
    # Highlight top discriminative regions
    top_k = 3  # Top-k regions to highlight
    flat_diff = np.abs(class_diff).flatten()
    flat_indices = np.argsort(flat_diff)[::-1][:top_k]
    
    # Convert flat indices to 2D
    pha_indices = flat_indices // len(amp_freqs)
    amp_indices = flat_indices % len(amp_freqs)
    
    # Get actual frequency values
    top_pha_freqs = pha_freqs[pha_indices]
    top_amp_freqs = amp_freqs[amp_indices]
    
    # Plot markers
    plt.scatter(top_pha_freqs, top_amp_freqs, 
                c='white', s=100, marker='*', edgecolor='black', 
                linewidth=1, zorder=10)
    
    # Add text labels
    for i, (px, py) in enumerate(zip(top_pha_freqs, top_amp_freqs)):
        plt.text(px, py + 3, f"#{i+1}", color='white', fontweight='bold', 
                 ha='center', va='center', backgroundcolor='black',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
    
    # Add class regions
    plt.axvspan(4, 8, ymin=0, ymax=0.4, alpha=0.1, color='blue', label='Class A Region')
    plt.axhspan(80, 100, xmin=0.2, xmax=0.4, alpha=0.1, color='blue')
    
    plt.axvspan(10, 14, ymin=0.6, ymax=1.0, alpha=0.1, color='red', label='Class B Region')
    plt.axhspan(120, 150, xmin=0.5, xmax=0.7, alpha=0.1, color='red')
    
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    
    if save:
        plt.savefig(output_dir / "discriminative_features.png", dpi=dpi)
    
    return fig_means, fig_discrim


def visualize_tsne_projection(
    pac_values: torch.Tensor,
    labels: np.ndarray,
    output_dir: Path,
    save: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Visualize t-SNE projection of PAC features colored by class."""
    # Flatten PAC values
    batch_size = pac_values.shape[0]
    pac_features = pac_values.reshape(batch_size, -1).cpu().numpy()
    
    # Apply PCA first to reduce dimensionality (speeds up t-SNE)
    n_components = min(50, pac_features.shape[1])
    pca = PCA(n_components=n_components)
    pac_features_pca = pca.fit_transform(pac_features)
    
    # Apply t-SNE
    print("Computing t-SNE projection (may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    pac_tsne = tsne.fit_transform(pac_features_pca)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Define colors for each class
    class_colors = ['#3498db', '#e74c3c']  # Blue, Red
    
    # Plot t-SNE
    for c in range(2):  # Two classes
        class_indices = np.where(labels == c)[0]
        plt.scatter(pac_tsne[class_indices, 0], pac_tsne[class_indices, 1],
                    c=class_colors[c], s=50, alpha=0.7, label=f"Class {'A' if c == 0 else 'B'}")
    
    plt.title("t-SNE Projection of PAC Features", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(alpha=0.2)
    
    if save:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "tsne_projection.png", dpi=dpi)
    
    return fig


def visualize_feature_importance_3d(
    pac_values: torch.Tensor,
    labels: np.ndarray,
    pha_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    output_dir: Path,
    save: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Visualize PAC feature importance in 3D."""
    # Flatten PAC values and prepare
    pac_values_np = pac_values.cpu().numpy()
    
    # Get class-specific indices
    class_a_indices = np.where(labels == 0)[0]
    class_b_indices = np.where(labels == 1)[0]
    
    # Calculate class means
    class_a_mean = pac_values_np[class_a_indices].mean(axis=0)
    class_b_mean = pac_values_np[class_b_indices].mean(axis=0)
    
    # Calculate importance as absolute difference
    importance = np.abs(class_b_mean - class_a_mean)
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(pha_freqs, amp_freqs)
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface with custom colormap for better visibility
    cmap = plt.cm.viridis
    surf = ax.plot_surface(X, Y, importance.T, cmap=cmap, 
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Feature Importance')
    
    # Add labels
    ax.set_xlabel('Phase Frequency (Hz)')
    ax.set_ylabel('Amplitude Frequency (Hz)')
    ax.set_zlabel('Importance')
    ax.set_title('3D PAC Feature Importance Map', fontsize=16)
    
    # Optimize viewing angle
    ax.view_init(elev=35, azim=-65)
    
    # Add class regions markers
    # Class A region (white sphere in blue region)
    ax.scatter([6], [90], [0], color='white', s=100, 
              marker='o', label='Class A Region')
    
    # Class B region (white sphere in red region)
    ax.scatter([12], [135], [0], color='white', s=100, 
              edgecolor='black', marker='o', label='Class B Region')
    
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "feature_importance_3d.png", dpi=dpi)
    
    return fig


def main(args: argparse.Namespace) -> int:
    """Main function for visualization."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_path = args.data_path
    signals, labels, metadata, params = load_binary_data(data_path)
    
    # Set output directory
    output_dir = Path(args.results_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Visualize class examples
    print("\nCreating signal class examples visualization...")
    visualize_class_examples(
        signals=signals,
        labels=labels,
        metadata=metadata,
        params=params,
        output_dir=output_dir,
        save=args.save_figs,
        dpi=args.dpi,
    )
    
    # Create PAC feature extractor
    print("\nCreating PAC feature extractor...")
    pac_module = create_pac_feature_extractor(
        seq_len=signals.shape[-1],
        fs=params["fs"],
        pha_n_bands=args.pha_n_bands,
        amp_n_bands=args.amp_n_bands,
        trainable=False,
    )
    
    # Extract PAC features
    print("\nExtracting PAC features...")
    pac_values = extract_pac_features(
        pac_module=pac_module,
        signals=signals,
        device=device,
    )
    
    # Get frequency bands
    pha_freqs = pac_module.PHA_MIDS_HZ.cpu().numpy()
    amp_freqs = pac_module.AMP_MIDS_HZ.cpu().numpy()
    
    # Visualize PAC features
    print("\nCreating PAC feature visualizations...")
    visualize_pac_features(
        pac_values=pac_values,
        labels=labels,
        pha_freqs=pha_freqs,
        amp_freqs=amp_freqs,
        output_dir=output_dir,
        save=args.save_figs,
        dpi=args.dpi,
    )
    
    # Visualize t-SNE projection
    print("\nCreating t-SNE projection visualization...")
    visualize_tsne_projection(
        pac_values=pac_values,
        labels=labels,
        output_dir=output_dir,
        save=args.save_figs,
        dpi=args.dpi,
    )
    
    # Visualize 3D feature importance
    print("\nCreating 3D feature importance visualization...")
    visualize_feature_importance_3d(
        pac_values=pac_values,
        labels=labels,
        pha_freqs=pha_freqs,
        amp_freqs=amp_freqs,
        output_dir=output_dir,
        save=args.save_figs,
        dpi=args.dpi,
    )
    
    print(f"\nAll visualizations saved to: {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize PAC features")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_PARAMS["data_path"],
        help="Path to binary PAC signals data file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_PARAMS["results_dir"],
        help="Directory to save results",
    )
    parser.add_argument(
        "--pha_n_bands",
        type=int,
        default=DEFAULT_PARAMS["pha_n_bands"],
        help="Number of phase frequency bands",
    )
    parser.add_argument(
        "--amp_n_bands",
        type=int,
        default=DEFAULT_PARAMS["amp_n_bands"],
        help="Number of amplitude frequency bands",
    )
    parser.add_argument(
        "--save_figs",
        action="store_true",
        default=DEFAULT_PARAMS["save_figs"],
        help="Save figures to disk",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_PARAMS["dpi"],
        help="DPI for saved figures",
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