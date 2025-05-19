#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 01:45:08 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/learnability/visualize_pac_learnability.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/learnability/visualize_pac_learnability.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Visualizes how PAC frequency bands evolve during training
  - Demonstrates the "learnability" aspect of PAC features
  - Tracks bandpass filter parameter changes over training iterations
  - Shows how frequency ranges optimize to discriminate between signal classes
  - Provides insights into which frequency bands are most important for classification

Dependencies:
  - packages:
    - torch
    - numpy
    - matplotlib
    - pandas
    - mngs
    - gpac
    - tqdm

IO:
  - input-files:
    - ./scripts/learnability/data/binary_pac_signals.pt
  - output-files:
    - ./scripts/learnability/results/learnability/
    - ./scripts/learnability/results/learnability/band_evolution.png
    - ./scripts/learnability/results/learnability/optimization_path.png
    - ./scripts/learnability/results/learnability/accuracy_vs_bands.png
    - ./scripts/learnability/results/learnability/feature_convergence.gif
"""

"""Imports"""
import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))

from gpac._pac import PAC


"""Parameters"""
DEFAULT_PARAMS = {
    "data_path": "./scripts/learnability/data/binary_pac_signals.pt",
    "results_dir": "./scripts/learnability/results/learnability",
    "pha_n_bands": 5,
    "amp_n_bands": 5,
    "n_epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.01,
    "track_interval": 5,  # Save intermediate model states every N batches
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


class PACClassifier(nn.Module):
    """
    Binary classifier using PAC features with trainable frequency bands.
    """
    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_n_bands: int,
        amp_n_bands: int,
        hidden_size: int = 32,
        trainable: bool = True
    ):
        super().__init__()
        
        # Initialize PAC module
        self.pac = PAC(
            seq_len=seq_len,
            fs=fs,
            pha_start_hz=1.0,
            pha_end_hz=20.0,
            pha_n_bands=pha_n_bands,
            amp_start_hz=60.0,
            amp_end_hz=180.0,
            amp_n_bands=amp_n_bands,
            trainable=trainable
        )
        
        # Store initial frequency values
        with torch.no_grad():
            self.initial_pha_bands = self.pac.PHA_MIDS_HZ.clone().cpu()
            self.initial_amp_bands = self.pac.AMP_MIDS_HZ.clone().cpu()
        
        # Feature dimensions
        self.feature_dim = pha_n_bands * amp_n_bands
        
        # Simple classifier 
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        # Track frequency bands over time
        self.band_history = {
            'iterations': [],
            'pha_bands': [],
            'amp_bands': [],
            'loss': [],
            'accuracy': []
        }
    
    def forward(self, x):
        # Extract PAC values
        pac_values = self.pac(x)
        
        # Handle different output shapes based on channels
        if len(pac_values.shape) == 4:  # (B, C, F_pha, F_amp)
            # Average across channels if multiple
            if pac_values.shape[1] > 1:
                pac_values = pac_values.mean(dim=1)  # (B, F_pha, F_amp)
            else:
                pac_values = pac_values.squeeze(1)  # Remove channel dim if only one
        
        # Flatten PAC features
        batch_size = pac_values.shape[0]
        features = pac_values.reshape(batch_size, -1)
        
        # Run through classifier
        logits = self.classifier(features)
        
        return logits
    
    def save_band_state(self, iteration, loss, accuracy):
        """Save current frequency band state."""
        self.band_history['iterations'].append(iteration)
        self.band_history['pha_bands'].append(self.pac.PHA_MIDS_HZ.clone().detach().cpu())
        self.band_history['amp_bands'].append(self.pac.AMP_MIDS_HZ.clone().detach().cpu())
        self.band_history['loss'].append(loss)
        self.band_history['accuracy'].append(accuracy)
    
    def get_feature_importance(self):
        """Extract feature importance from classifier weights."""
        # Get weights from the first layer of classifier
        with torch.no_grad():
            weights = self.classifier[0].weight  # (hidden_size, feature_dim)
            
            # Calculate importance as L2 norm of weights
            importance = torch.norm(weights, dim=0)  # (feature_dim,)
            
            # Reshape to PAC grid
            importance = importance.reshape(self.pac.PHA_MIDS_HZ.shape[0], 
                                          self.pac.AMP_MIDS_HZ.shape[0])
            
        return importance


def create_data_loaders(
    signals: torch.Tensor,
    labels: np.ndarray,
    batch_size: int,
    val_split: float = 0.2,
):
    """Create training and validation data loaders."""
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Split into train and validation sets
    n_samples = len(signals)
    n_val = int(n_samples * val_split)
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    # Create datasets
    train_dataset = TensorDataset(signals[train_indices], labels_tensor[train_indices])
    val_dataset = TensorDataset(signals[val_indices], labels_tensor[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    learning_rate: float,
    track_interval: int,
    device: torch.device
):
    """Train the PAC classifier with tracking band evolution."""
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop with tracking
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                batch_correct = (predictions == targets).sum().item()
                batch_total = targets.size(0)
                
                correct += batch_correct
                total += batch_total
                train_loss += loss.item()
                
                # Update progress bar
                accuracy = correct / total
                pbar.set_postfix({
                    'loss': f"{train_loss / (batch_idx + 1):.4f}",
                    'acc': f"{accuracy:.4f}"
                })
                
                # Save model state periodically
                if global_step % track_interval == 0:
                    model.save_band_state(global_step, loss.item(), accuracy)
                
                global_step += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
                val_loss += loss.item()
        
        val_accuracy = val_correct / val_total
        print(f"Validation: Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Save final state for this epoch
        model.save_band_state(global_step, val_loss / len(val_loader), val_accuracy)
    
    return model


def plot_band_evolution(model: PACClassifier, output_dir: Path, save: bool = True, dpi: int = 300):
    """
    Plot the evolution of frequency bands during training.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    history = model.band_history
    iterations = history['iterations']
    
    # Create a figure with two subplots for phase and amplitude bands
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot phase band evolution
    ax_pha = axes[0]
    pha_bands = torch.stack(history['pha_bands'])
    for i in range(pha_bands.shape[1]):
        ax_pha.plot(iterations, pha_bands[:, i], label=f"Band {i+1}")
    
    ax_pha.set_title("Phase Frequency Band Evolution", fontsize=14)
    ax_pha.set_xlabel("Training Iterations", fontsize=12)
    ax_pha.set_ylabel("Frequency (Hz)", fontsize=12)
    ax_pha.grid(alpha=0.3)
    ax_pha.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Highlight the initial ranges for class A and class B
    ax_pha.axhspan(4, 8, color='blue', alpha=0.1, label='Class A Range')
    ax_pha.axhspan(10, 14, color='red', alpha=0.1, label='Class B Range')
    
    # Plot amplitude band evolution
    ax_amp = axes[1]
    amp_bands = torch.stack(history['amp_bands'])
    for i in range(amp_bands.shape[1]):
        ax_amp.plot(iterations, amp_bands[:, i], label=f"Band {i+1}")
    
    ax_amp.set_title("Amplitude Frequency Band Evolution", fontsize=14)
    ax_amp.set_xlabel("Training Iterations", fontsize=12)
    ax_amp.set_ylabel("Frequency (Hz)", fontsize=12)
    ax_amp.grid(alpha=0.3)
    ax_amp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Highlight the initial ranges for class A and class B
    ax_amp.axhspan(80, 100, color='blue', alpha=0.1, label='Class A Range')
    ax_amp.axhspan(120, 150, color='red', alpha=0.1, label='Class B Range')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(output_dir / "band_evolution.png", dpi=dpi)
        plt.close(fig)
    
    return fig


def plot_optimization_path(model: PACClassifier, output_dir: Path, save: bool = True, dpi: int = 300):
    """
    Plot the optimization path of frequency bands in 2D.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    history = model.band_history
    iterations = history['iterations']
    pha_bands = torch.stack(history['pha_bands'])
    amp_bands = torch.stack(history['amp_bands'])
    accuracies = history['accuracy']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate point size based on iteration (later points are larger)
    sizes = np.linspace(20, 100, len(iterations))
    
    # Class regions
    ax.axvspan(4, 8, ymin=0, ymax=0.4, alpha=0.1, color='blue', label='Class A Region')
    ax.axhspan(80, 100, xmin=0.2, xmax=0.4, alpha=0.1, color='blue')
    
    ax.axvspan(10, 14, ymin=0.6, ymax=1.0, alpha=0.1, color='red', label='Class B Region')
    ax.axhspan(120, 150, xmin=0.5, xmax=0.7, alpha=0.1, color='red')
    
    # For each band, plot the optimization path
    pha_n_bands = pha_bands.shape[1]
    amp_n_bands = amp_bands.shape[1]
    
    colormap = plt.cm.rainbow
    colors = [colormap(i / (pha_n_bands * amp_n_bands)) for i in range(pha_n_bands * amp_n_bands)]
    
    band_idx = 0
    for p_idx in range(pha_n_bands):
        for a_idx in range(amp_n_bands):
            # Get path for this band
            pha_path = pha_bands[:, p_idx]
            amp_path = amp_bands[:, a_idx]
            
            # Plot path with gradient color to show direction
            points = np.array([pha_path.numpy(), amp_path.numpy()]).T
            
            for i in range(len(points) - 1):
                ax.plot(points[i:i+2, 0], points[i:i+2, 1], 
                       '-', color=colors[band_idx], alpha=0.7, linewidth=1.5)
                
                # Add arrows to show direction
                if i % 3 == 0:  # Add arrow every few points
                    dx = points[i+1, 0] - points[i, 0]
                    dy = points[i+1, 1] - points[i, 1]
                    ax.arrow(points[i, 0], points[i, 1], dx, dy, 
                            head_width=0.3, head_length=0.5, fc=colors[band_idx], 
                            ec=colors[band_idx], alpha=0.7)
            
            # Mark start point
            ax.scatter(pha_path[0], amp_path[0], s=80, c=[colors[band_idx]], 
                      marker='o', edgecolor='k', alpha=0.9, 
                      label=f"Band ({p_idx+1},{a_idx+1}) Start")
            
            # Mark end point
            ax.scatter(pha_path[-1], amp_path[-1], s=150, c=[colors[band_idx]], 
                      marker='*', edgecolor='k', alpha=0.9, 
                      label=f"Band ({p_idx+1},{a_idx+1}) End")
            
            band_idx += 1
    
    ax.set_title("Frequency Band Optimization Path", fontsize=16)
    ax.set_xlabel("Phase Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Amplitude Frequency (Hz)", fontsize=14)
    ax.grid(alpha=0.3)
    
    # Create custom legend for start and end points
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', 
               markersize=10, label='Start Points'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='grey', 
               markersize=15, label='End Points'),
        Line2D([0], [0], color='blue', lw=4, alpha=0.1, label='Class A Region'),
        Line2D([0], [0], color='red', lw=4, alpha=0.1, label='Class B Region')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(output_dir / "optimization_path.png", dpi=dpi)
        plt.close(fig)
    
    return fig


def plot_accuracy_vs_bands(model: PACClassifier, output_dir: Path, save: bool = True, dpi: int = 300):
    """
    Plot the relationship between accuracy and frequency band evolution.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    history = model.band_history
    iterations = history['iterations']
    accuracies = history['accuracy']
    
    # Calculate average band positions
    pha_bands = torch.stack(history['pha_bands'])
    amp_bands = torch.stack(history['amp_bands'])
    
    # Calculate mean distance to target regions
    pha_class_A_center = 6.0  # Hz, center of Class A phase region
    pha_class_B_center = 12.0  # Hz, center of Class B phase region
    amp_class_A_center = 90.0  # Hz, center of Class A amplitude region
    amp_class_B_center = 135.0  # Hz, center of Class B amplitude region
    
    pha_distances = []
    amp_distances = []
    
    for p_bands, a_bands in zip(pha_bands, amp_bands):
        # Distance from phase bands to correct regions
        p_dist_A = abs(p_bands - pha_class_A_center).min().item()
        p_dist_B = abs(p_bands - pha_class_B_center).min().item()
        pha_distances.append(min(p_dist_A, p_dist_B))
        
        # Distance from amplitude bands to correct regions
        a_dist_A = abs(a_bands - amp_class_A_center).min().item()
        a_dist_B = abs(a_bands - amp_class_B_center).min().item()
        amp_distances.append(min(a_dist_A, a_dist_B))
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot accuracy over iterations
    ax1 = axes[0]
    ax1.plot(iterations, accuracies, 'b-', linewidth=2)
    ax1.set_title("Classification Accuracy", fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Plot phase distance over iterations
    ax2 = axes[1]
    ax2.plot(iterations, pha_distances, 'g-', linewidth=2)
    ax2.set_title("Phase Band Distance to Optimal Region", fontsize=14)
    ax2.set_ylabel("Distance (Hz)", fontsize=12)
    ax2.grid(alpha=0.3)
    
    # Plot amplitude distance over iterations
    ax3 = axes[2]
    ax3.plot(iterations, amp_distances, 'r-', linewidth=2)
    ax3.set_title("Amplitude Band Distance to Optimal Region", fontsize=14)
    ax3.set_xlabel("Training Iterations", fontsize=12)
    ax3.set_ylabel("Distance (Hz)", fontsize=12)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(output_dir / "accuracy_vs_bands.png", dpi=dpi)
        plt.close(fig)
    
    return fig


def create_feature_convergence_animation(model: PACClassifier, output_dir: Path, save: bool = True, dpi: int = 300):
    """
    Create an animation showing how PAC features evolve during training.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    history = model.band_history
    iterations = history['iterations']
    pha_bands = torch.stack(history['pha_bands'])
    amp_bands = torch.stack(history['amp_bands'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        
        # Class regions
        ax.axvspan(4, 8, ymin=0, ymax=0.4, alpha=0.1, color='blue', label='Class A Region')
        ax.axhspan(80, 100, xmin=0.2, xmax=0.4, alpha=0.1, color='blue')
        ax.axvspan(10, 14, ymin=0.6, ymax=1.0, alpha=0.1, color='red', label='Class B Region')
        ax.axhspan(120, 150, xmin=0.5, xmax=0.7, alpha=0.1, color='red')
        
        # Get bands for current frame
        current_pha = pha_bands[frame]
        current_amp = amp_bands[frame]
        
        # Plot each band as a heatmap point
        X, Y = np.meshgrid(current_pha.numpy(), current_amp.numpy())
        
        # Create a grid to represent feature importance (placeholder in this case)
        grid_size = (len(current_pha), len(current_amp))
        Z = np.ones(grid_size)  # Placeholder for feature importance
        
        # Show heatmap
        im = ax.pcolormesh(X, Y, Z.T, cmap='viridis', alpha=0.5)
        
        # Plot bands as scatter points
        for p_idx, p_val in enumerate(current_pha):
            for a_idx, a_val in enumerate(current_amp):
                ax.scatter(p_val.item(), a_val.item(), s=100, 
                          c='red', alpha=0.7, edgecolor='k')
        
        # Label for current iteration
        ax.set_title(f"Frequency Band Evolution - Iteration {iterations[frame]}", fontsize=16)
        ax.set_xlabel("Phase Frequency (Hz)", fontsize=14)
        ax.set_ylabel("Amplitude Frequency (Hz)", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend()
        
        return (im,)
    
    # Create animation
    frames = min(30, len(iterations))  # Limit number of frames for performance
    step = max(1, len(iterations) // frames)
    frame_indices = range(0, len(iterations), step)
    
    anim = FuncAnimation(fig, update, frames=frame_indices, 
                          interval=200, blit=False)
    
    if save:
        anim.save(output_dir / "feature_convergence.gif", 
                 dpi=dpi, writer='pillow', fps=5)
        plt.close(fig)
    
    return anim


def train_and_visualize_learnability(args):
    """Main function to train and visualize PAC learnability."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    signals, labels, metadata, params = load_binary_data(args.data_path)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        signals=signals,
        labels=labels,
        batch_size=args.batch_size,
    )
    
    # Create model
    model = PACClassifier(
        seq_len=signals.shape[-1],
        fs=params["fs"],
        pha_n_bands=args.pha_n_bands,
        amp_n_bands=args.amp_n_bands,
        trainable=True
    )
    
    # Train model with tracking
    print("\nTraining model with trainable PAC frequency bands...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        track_interval=args.track_interval,
        device=device
    )
    
    # Set output directory
    output_dir = Path(args.results_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations
    print("\nCreating band evolution visualization...")
    plot_band_evolution(model, output_dir, save=args.save_figs, dpi=args.dpi)
    
    print("\nCreating optimization path visualization...")
    plot_optimization_path(model, output_dir, save=args.save_figs, dpi=args.dpi)
    
    print("\nCreating accuracy vs. bands visualization...")
    plot_accuracy_vs_bands(model, output_dir, save=args.save_figs, dpi=args.dpi)
    
    print("\nCreating feature convergence animation...")
    create_feature_convergence_animation(model, output_dir, save=args.save_figs, dpi=args.dpi)
    
    # Save trained model for future use
    torch.save({
        'model_state_dict': model.state_dict(),
        'band_history': model.band_history,
        'hyperparams': {
            'pha_n_bands': args.pha_n_bands,
            'amp_n_bands': args.amp_n_bands,
            'n_epochs': args.n_epochs,
            'learning_rate': args.learning_rate,
        }
    }, output_dir / "trained_learnability_model.pt")
    
    print(f"\nTraining and visualization completed. Results saved to: {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize PAC learnability")
    
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
        "--n_epochs",
        type=int,
        default=DEFAULT_PARAMS["n_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_PARAMS["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_PARAMS["learning_rate"],
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--track_interval",
        type=int,
        default=DEFAULT_PARAMS["track_interval"],
        help="Interval for tracking frequency band changes",
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
    
    exit_status = train_and_visualize_learnability(args)
    
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