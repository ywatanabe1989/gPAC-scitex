#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 13:00:45 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/exp_02_tensorpac_comparison/generate_comparison_figures.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/exp_02_tensorpac_comparison/generate_comparison_figures.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Generates publication-ready figures comparing gPAC and Tensorpac
  - Creates Figure 1: PAC calculation workflow diagram
  - Creates Figure 2: Calculation accuracy comparison
  - Creates Figure 3: Performance benchmark comparison across parameters
  - Exports figures in multiple formats (PNG, PDF, SVG)

Dependencies:
  - packages:
    - Matplotlib
    - seaborn
    - numpy
    - pandas
    - mngs

IO:
  - input-files:
    - ./data/exp_02/benchmark_*.json
    - ./data/exp_02/pac_value_comparison.csv
    - ./data/exp_02/correlation_metrics.json

  - output-files:
    - ./data/exp_02/figure_1_pac_workflow.png
    - ./data/exp_02/figure_2_calculation_accuracy.png
    - ./data/exp_02/figure_3_performance_benchmark.png
"""

"""Imports"""
import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

"""Warnings"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

"""Parameters"""
# Will be loaded via mngs.io.load_configs() in run_main()

"""Functions & Classes"""
def find_benchmark_results(data_dir: str = "./data/exp_02") -> List[str]:
    """
    Find all benchmark result files.
    
    Args:
        data_dir: Directory containing benchmark results
        
    Returns:
        List of benchmark result file paths
    """
    pattern = os.path.join(data_dir, "benchmark_*.json")
    grid_pattern = os.path.join(data_dir, "grid_*.json")
    sweep_pattern = os.path.join(data_dir, "sweep_*.json")
    
    # Find all benchmark files
    result_files = glob.glob(pattern)
    result_files.extend(glob.glob(grid_pattern))
    result_files.extend(glob.glob(sweep_pattern))
    
    if not result_files:
        warnings.warn(f"No benchmark files found at {pattern}")
    
    return result_files

def load_benchmark_data(files: List[str]) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Load benchmark data from JSON files.
    
    Args:
        files: List of benchmark result file paths
        
    Returns:
        Tuple of (raw benchmark data, processed DataFrame)
    """
    if not files:
        return [], pd.DataFrame()
    
    # Load raw data
    benchmark_data = []
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                benchmark_data.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Process data into a DataFrame for easier plotting
    records = []
    
    for data in benchmark_data:
        # Handle different data formats
        if "params" in data:
            # Single benchmark
            params = data["params"]
            record = {
                "timestamp": data.get("timestamp", ""),
                **params,
                "gpac_success": data.get("gpac", {}).get("success", False),
                "gpac_calc_time": data.get("gpac", {}).get("calc_time", 0),
                "gpac_memory_cpu": data.get("gpac", {}).get("memory_cpu", 0),
                "gpac_memory_gpu": data.get("gpac", {}).get("memory_gpu", 0),
                "tensorpac_success": data.get("tensorpac", {}).get("success", False),
                "tensorpac_calc_time": data.get("tensorpac", {}).get("calc_time", 0),
                "tensorpac_memory_cpu": data.get("tensorpac", {}).get("memory_cpu", 0),
                "speedup_factor": data.get("gpac", {}).get("calc_time", 0) / 
                                 data.get("tensorpac", {}).get("calc_time", 1) if 
                                 data.get("tensorpac", {}).get("calc_time", 0) > 0 else 0,
            }
            records.append(record)
        elif "param_name" in data:
            # Parameter sweep
            param_name = data["param_name"]
            sweep_results = data.get("results", [])
            
            for result in sweep_results:
                params = result.get("params", {})
                record = {
                    "timestamp": result.get("timestamp", ""),
                    **params,
                    "param_sweep": param_name,
                    "gpac_success": result.get("gpac", {}).get("success", False),
                    "gpac_calc_time": result.get("gpac", {}).get("calc_time", 0),
                    "gpac_memory_cpu": result.get("gpac", {}).get("memory_cpu", 0),
                    "gpac_memory_gpu": result.get("gpac", {}).get("memory_gpu", 0),
                    "tensorpac_success": result.get("tensorpac", {}).get("success", False),
                    "tensorpac_calc_time": result.get("tensorpac", {}).get("calc_time", 0),
                    "tensorpac_memory_cpu": result.get("tensorpac", {}).get("memory_cpu", 0),
                    "speedup_factor": result.get("gpac", {}).get("calc_time", 0) / 
                                     result.get("tensorpac", {}).get("calc_time", 1) if 
                                     result.get("tensorpac", {}).get("calc_time", 0) > 0 else 0,
                }
                records.append(record)
        elif "param_names" in data:
            # Grid search
            param_names = data["param_names"]
            grid_results = data.get("results", [])
            
            for param_values, result in grid_results:
                # param_values is a dict mapping param_names to values
                params = result.get("params", {})
                record = {
                    "timestamp": result.get("timestamp", ""),
                    **params,
                    "param_grid": ",".join(param_names),
                    "gpac_success": result.get("gpac", {}).get("success", False),
                    "gpac_calc_time": result.get("gpac", {}).get("calc_time", 0),
                    "gpac_memory_cpu": result.get("gpac", {}).get("memory_cpu", 0),
                    "gpac_memory_gpu": result.get("gpac", {}).get("memory_gpu", 0),
                    "tensorpac_success": result.get("tensorpac", {}).get("success", False),
                    "tensorpac_calc_time": result.get("tensorpac", {}).get("calc_time", 0),
                    "tensorpac_memory_cpu": result.get("tensorpac", {}).get("memory_cpu", 0),
                    "speedup_factor": result.get("gpac", {}).get("calc_time", 0) / 
                                     result.get("tensorpac", {}).get("calc_time", 1) if 
                                     result.get("tensorpac", {}).get("calc_time", 0) > 0 else 0,
                }
                records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    return benchmark_data, df

def load_pac_comparison_data(data_dir: str = "./data/exp_02") -> Tuple[pd.DataFrame, Dict]:
    """
    Load PAC comparison data.
    
    Args:
        data_dir: Directory containing PAC comparison results
        
    Returns:
        Tuple of (PAC comparison DataFrame, metrics dictionary)
    """
    # PAC value comparison CSV
    csv_file = os.path.join(data_dir, "pac_value_comparison.csv")
    
    if not os.path.exists(csv_file):
        warnings.warn(f"PAC comparison CSV file not found: {csv_file}")
        return pd.DataFrame(), {}
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Load metrics
    metrics_file = os.path.join(data_dir, "correlation_metrics.json")
    
    if not os.path.exists(metrics_file):
        warnings.warn(f"Correlation metrics file not found: {metrics_file}")
        return df, {}
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    return df, metrics

def figure_1_pac_workflow(output_dir: str) -> str:
    """
    Create Figure 1: PAC calculation workflow diagram.
    
    Args:
        output_dir: Directory to save figure
        
    Returns:
        Path to saved figure file
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Define a simplified PAC workflow
    plt.text(0.5, 0.95, "Phase-Amplitude Coupling (PAC) Workflow", ha="center", fontsize=16, weight="bold")
    
    # Input data
    plt.arrow(0.5, 0.9, 0, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    plt.text(0.5, 0.88, "Input Signal (EEG/LFP)", ha="center", fontsize=12)
    
    # Parallel processing box
    rect = plt.Rectangle((0.1, 0.32), 0.8, 0.53, fill=True, color="lightyellow", alpha=0.3)
    plt.gca().add_patch(rect)
    plt.text(0.5, 0.82, "gPAC Processing (GPU-accelerated)", ha="center", fontsize=14, weight="bold")
    
    # Bandpass filtering
    plt.arrow(0.5, 0.85, 0, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    rect1 = plt.Rectangle((0.25, 0.7), 0.5, 0.08, fill=True, color="lightblue", alpha=0.5)
    plt.gca().add_patch(rect1)
    plt.text(0.5, 0.74, "Parallel Bandpass Filtering", ha="center", fontsize=12)
    
    # Phase and Amplitude extraction (Hilbert transform)
    plt.arrow(0.5, 0.7, 0, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    rect2 = plt.Rectangle((0.25, 0.55), 0.5, 0.08, fill=True, color="lightgreen", alpha=0.5)
    plt.gca().add_patch(rect2)
    plt.text(0.5, 0.59, "Parallel Hilbert Transform", ha="center", fontsize=12)
    
    # Split into phase and amplitude
    plt.arrow(0.35, 0.55, -0.1, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    plt.arrow(0.65, 0.55, 0.1, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    
    # Phase
    rect3 = plt.Rectangle((0.15, 0.4), 0.3, 0.08, fill=True, color="lightcoral", alpha=0.5)
    plt.gca().add_patch(rect3)
    plt.text(0.3, 0.44, "Phase", ha="center", fontsize=12)
    
    # Amplitude
    rect4 = plt.Rectangle((0.55, 0.4), 0.3, 0.08, fill=True, color="lightcoral", alpha=0.5)
    plt.gca().add_patch(rect4)
    plt.text(0.7, 0.44, "Amplitude", ha="center", fontsize=12)
    
    # Combine for PAC
    plt.arrow(0.3, 0.4, 0.13, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    plt.arrow(0.7, 0.4, -0.13, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    
    # Modulation Index Calculation
    rect5 = plt.Rectangle((0.3, 0.25), 0.4, 0.08, fill=True, color="lightsalmon", alpha=0.5)
    plt.gca().add_patch(rect5)
    plt.text(0.5, 0.29, "Modulation Index Calculation", ha="center", fontsize=12)
    
    # Optional permutation testing
    plt.arrow(0.5, 0.25, 0, -0.07, head_width=0.03, head_length=0.02, fc='k', ec='k')
    rect6 = plt.Rectangle((0.3, 0.1), 0.4, 0.08, fill=True, color="lightgray", alpha=0.5)
    plt.gca().add_patch(rect6)
    plt.text(0.5, 0.14, "Optional: Statistical Testing", ha="center", fontsize=12)
    
    # Add annotations about gPAC advantages
    advantages = [
        "• Parallel GPU acceleration",
        "• Trainable frequency bands",
        "• Batch processing",
        "• PyTorch integration"
    ]
    for i, adv in enumerate(advantages):
        plt.text(0.85, 0.65 - i*0.05, adv, fontsize=10, ha="left")
    
    # Add a title for the advantages
    plt.text(0.85, 0.7, "gPAC Advantages:", fontsize=11, weight="bold", ha="left")
    
    # Remove axes
    plt.axis("off")
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "figure_1_pac_workflow.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, "figure_1_pac_workflow.pdf")
    plt.savefig(output_file_pdf, format="pdf", bbox_inches="tight")
    
    # Also save as SVG
    output_file_svg = os.path.join(output_dir, "figure_1_pac_workflow.svg")
    plt.savefig(output_file_svg, format="svg", bbox_inches="tight")
    
    # Close figure to free memory
    plt.close(fig)
    
    return output_file

def figure_2_calculation_accuracy(
    pac_df: pd.DataFrame, 
    metrics: Dict,
    output_dir: str
) -> str:
    """
    Create Figure 2: Calculation accuracy comparison.
    
    Args:
        pac_df: PAC comparison DataFrame
        metrics: Correlation metrics dictionary
        output_dir: Directory to save figure
        
    Returns:
        Path to saved figure file
    """
    if pac_df.empty:
        return ""
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    # PLOT 1: Heatmap of gPAC values
    ax1 = plt.subplot(gs[0, 0])
    
    # Get the PAC values and reshape for heatmap
    if "gpac_pac" in pac_df.columns:
        # Already has the values
        gpac_vals = pac_df["gpac_pac"].values
        tensorpac_vals = pac_df["tensorpac_pac"].values
    else:
        # Need to calculate the values
        gpac_vals = pac_df["gpac_pac_norm"].values
        tensorpac_vals = pac_df["tensorpac_pac_norm"].values
    
    # Get frequency values
    pha_freqs = pac_df["pha_freq"].unique()
    amp_freqs = pac_df["amp_freq"].unique()
    
    # Reshape to 2D array for heatmap
    gpac_mat = gpac_vals.reshape(len(pha_freqs), len(amp_freqs))
    
    # Plot gPAC heatmap
    sns.heatmap(gpac_mat, ax=ax1, cmap="viridis",
               xticklabels=np.round(pha_freqs, 1),
               yticklabels=np.round(amp_freqs, 1))
    ax1.set_title("gPAC Values")
    ax1.set_xlabel("Phase Frequency (Hz)")
    ax1.set_ylabel("Amplitude Frequency (Hz)")
    
    # PLOT 2: Heatmap of Tensorpac values
    ax2 = plt.subplot(gs[0, 1])
    
    # Reshape to 2D array for heatmap
    tensorpac_mat = tensorpac_vals.reshape(len(pha_freqs), len(amp_freqs))
    
    # Plot Tensorpac heatmap
    sns.heatmap(tensorpac_mat, ax=ax2, cmap="viridis",
               xticklabels=np.round(pha_freqs, 1),
               yticklabels=np.round(amp_freqs, 1))
    ax2.set_title("Tensorpac Values")
    ax2.set_xlabel("Phase Frequency (Hz)")
    ax2.set_ylabel("Amplitude Frequency (Hz)")
    
    # PLOT 3: Heatmap of absolute differences
    ax3 = plt.subplot(gs[0, 2])
    
    # Calculate differences
    diff_mat = np.abs(gpac_mat - tensorpac_mat)
    
    # Plot difference heatmap
    sns.heatmap(diff_mat, ax=ax3, cmap="OrRd",
               xticklabels=np.round(pha_freqs, 1),
               yticklabels=np.round(amp_freqs, 1))
    ax3.set_title("Absolute Difference")
    ax3.set_xlabel("Phase Frequency (Hz)")
    ax3.set_ylabel("Amplitude Frequency (Hz)")
    
    # PLOT 4: Scatter plot of gPAC vs Tensorpac values
    ax4 = plt.subplot(gs[1, 0:2])
    
    # Plot scatter
    ax4.scatter(gpac_vals, tensorpac_vals, alpha=0.5)
    
    # Plot unity line
    min_val = min(gpac_vals.min(), tensorpac_vals.min())
    max_val = max(gpac_vals.max(), tensorpac_vals.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3)
    
    # Set labels
    ax4.set_title("Comparison: gPAC vs Tensorpac Values")
    ax4.set_xlabel("gPAC PAC Value")
    ax4.set_ylabel("Tensorpac PAC Value")
    
    # Add correlation metrics as text
    if metrics:
        metrics_text = (
            f"Pearson r: {metrics.get('pearson_r', 'N/A'):.4f}\n"
            f"RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
            f"Cosine Similarity: {metrics.get('cosine_similarity', 'N/A'):.4f}\n"
            f"Jensen-Shannon Div: {metrics.get('jensen_shannon_divergence', 'N/A'):.4f}"
        )
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    
    # PLOT 5: Bar plot of correlation metrics
    ax5 = plt.subplot(gs[1, 2])
    
    # Bar plot metrics
    if metrics:
        metric_names = ["Pearson r", "Cosine Sim", "Top Peak Overlap"]
        metric_values = [
            metrics.get("pearson_r", 0),
            metrics.get("cosine_similarity", 0),
            metrics.get("top_peak_overlap_ratio", 0)
        ]
        
        # Create bar colors based on values
        # Green is good (close to 1), red is bad (close to 0)
        colors = ["green" if v > 0.8 else "orange" if v > 0.5 else "red" for v in metric_values]
        
        # Plot bars
        ax5.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add horizontal line at y=1.0 (perfect agreement)
        ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        
        # Set labels
        ax5.set_title("Agreement Metrics")
        ax5.set_ylim(0, 1.1)
        ax5.set_ylabel("Value (higher is better)")
        
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            ax5.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    # Main title
    plt.suptitle("PAC Calculation Accuracy: gPAC vs Tensorpac", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "figure_2_calculation_accuracy.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, "figure_2_calculation_accuracy.pdf")
    plt.savefig(output_file_pdf, format="pdf", bbox_inches="tight")
    
    # Also save as SVG
    output_file_svg = os.path.join(output_dir, "figure_2_calculation_accuracy.svg")
    plt.savefig(output_file_svg, format="svg", bbox_inches="tight")
    
    # Close figure to free memory
    plt.close(fig)
    
    return output_file

def figure_3_performance_benchmark(
    benchmark_df: pd.DataFrame, 
    output_dir: str
) -> str:
    """
    Create Figure 3: Performance benchmark comparison.
    
    Args:
        benchmark_df: Benchmark results DataFrame
        output_dir: Directory to save figure
        
    Returns:
        Path to saved figure file
    """
    if benchmark_df.empty:
        return ""
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # PLOT 1: Bar plot of computation time comparison
    ax1 = plt.subplot(gs[0, 0:2])
    
    # Filter for baseline parameters
    baseline = benchmark_df.query('batch_size == 2 and n_chs == 2')
    
    if not baseline.empty:
        # Create bar data
        packages = ["gPAC", "Tensorpac"]
        times = [baseline["gpac_calc_time"].mean(), baseline["tensorpac_calc_time"].mean()]
        
        # Calculate speedup
        speedup = times[1] / times[0] if times[0] > 0 else 0
        
        # Plot bars
        bars = ax1.bar(packages, times, color=["steelblue", "salmon"], alpha=0.7)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f"{times[i]:.3f}s", ha='center', va='bottom')
        
        # Add speedup annotation
        ax1.text(0.5, 0.9, f"gPAC Speedup: {speedup:.1f}x", transform=ax1.transAxes,
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set labels
        ax1.set_title("Computation Time Comparison (Lower is Better)")
        ax1.set_ylabel("Time (seconds)")
    else:
        ax1.text(0.5, 0.5, "No baseline data available", ha='center', va='center', fontsize=12)
        ax1.set_title("Computation Time Comparison")
    
    # PLOT 2: Memory usage comparison
    ax2 = plt.subplot(gs[0, 2])
    
    if not baseline.empty:
        # Create bar data for memory
        memory_data = {
            "CPU (MB)": [baseline["gpac_memory_cpu"].mean(), baseline["tensorpac_memory_cpu"].mean()],
            "GPU (MB)": [baseline["gpac_memory_gpu"].mean() / (1024*1024), 0]  # Convert bytes to MB
        }
        
        # Create DataFrame for easy plotting
        memory_df = pd.DataFrame(memory_data, index=packages)
        
        # Plot bars
        memory_df.plot(kind="bar", ax=ax2, rot=0, alpha=0.7)
        
        # Set labels
        ax2.set_title("Memory Usage Comparison")
        ax2.set_ylabel("Memory (MB)")
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "No baseline data available", ha='center', va='center', fontsize=12)
        ax2.set_title("Memory Usage Comparison")
    
    # PLOT 3: Parameter sweep for batch size
    ax3 = plt.subplot(gs[1, 0])
    
    # Filter for batch size parameter sweep
    batch_sweep = benchmark_df[benchmark_df["param_sweep"] == "batch_size"] if "param_sweep" in benchmark_df.columns else pd.DataFrame()
    
    if not batch_sweep.empty:
        # Create line plot
        batch_sweep_pivot = batch_sweep.pivot_table(
            index="batch_size", 
            values=["gpac_calc_time", "tensorpac_calc_time"],
            aggfunc="mean"
        )
        
        # Plot lines
        batch_sweep_pivot.plot(marker="o", ax=ax3)
        
        # Set labels
        ax3.set_title("Computation Time vs Batch Size")
        ax3.set_xlabel("Batch Size")
        ax3.set_ylabel("Time (seconds)")
        ax3.legend(["gPAC", "Tensorpac"])
        ax3.grid(True, alpha=0.3)
    else:
        # Try to find any batch size variation
        batch_sizes = benchmark_df["batch_size"].unique() if "batch_size" in benchmark_df.columns else []
        if len(batch_sizes) > 1:
            # Group by batch size
            by_batch = benchmark_df.groupby("batch_size").agg({
                "gpac_calc_time": "mean",
                "tensorpac_calc_time": "mean"
            })
            
            # Plot lines
            by_batch.plot(marker="o", ax=ax3)
            
            # Set labels
            ax3.set_title("Computation Time vs Batch Size")
            ax3.set_xlabel("Batch Size")
            ax3.set_ylabel("Time (seconds)")
            ax3.legend(["gPAC", "Tensorpac"])
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No batch size sweep data", ha='center', va='center', fontsize=12)
            ax3.set_title("Computation Time vs Batch Size")
    
    # PLOT 4: Parameter sweep for channels
    ax4 = plt.subplot(gs[1, 1])
    
    # Filter for channel parameter sweep
    channel_sweep = benchmark_df[benchmark_df["param_sweep"] == "n_chs"] if "param_sweep" in benchmark_df.columns else pd.DataFrame()
    
    if not channel_sweep.empty:
        # Create line plot
        channel_sweep_pivot = channel_sweep.pivot_table(
            index="n_chs", 
            values=["gpac_calc_time", "tensorpac_calc_time"],
            aggfunc="mean"
        )
        
        # Plot lines
        channel_sweep_pivot.plot(marker="o", ax=ax4)
        
        # Set labels
        ax4.set_title("Computation Time vs Number of Channels")
        ax4.set_xlabel("Number of Channels")
        ax4.set_ylabel("Time (seconds)")
        ax4.legend(["gPAC", "Tensorpac"])
        ax4.grid(True, alpha=0.3)
    else:
        # Try to find any channel variation
        channels = benchmark_df["n_chs"].unique() if "n_chs" in benchmark_df.columns else []
        if len(channels) > 1:
            # Group by channels
            by_chs = benchmark_df.groupby("n_chs").agg({
                "gpac_calc_time": "mean",
                "tensorpac_calc_time": "mean"
            })
            
            # Plot lines
            by_chs.plot(marker="o", ax=ax4)
            
            # Set labels
            ax4.set_title("Computation Time vs Number of Channels")
            ax4.set_xlabel("Number of Channels")
            ax4.set_ylabel("Time (seconds)")
            ax4.legend(["gPAC", "Tensorpac"])
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No channel sweep data", ha='center', va='center', fontsize=12)
            ax4.set_title("Computation Time vs Number of Channels")
    
    # PLOT 5: Parameter sweep for sequence length (t_sec)
    ax5 = plt.subplot(gs[1, 2])
    
    # Filter for sequence length parameter sweep
    seq_sweep = benchmark_df[benchmark_df["param_sweep"] == "t_sec"] if "param_sweep" in benchmark_df.columns else pd.DataFrame()
    
    if not seq_sweep.empty:
        # Create line plot
        seq_sweep_pivot = seq_sweep.pivot_table(
            index="t_sec", 
            values=["gpac_calc_time", "tensorpac_calc_time"],
            aggfunc="mean"
        )
        
        # Plot lines
        seq_sweep_pivot.plot(marker="o", ax=ax5)
        
        # Set labels
        ax5.set_title("Computation Time vs Signal Duration")
        ax5.set_xlabel("Duration (seconds)")
        ax5.set_ylabel("Time (seconds)")
        ax5.legend(["gPAC", "Tensorpac"])
        ax5.grid(True, alpha=0.3)
    else:
        # Try to find any duration variation
        durations = benchmark_df["t_sec"].unique() if "t_sec" in benchmark_df.columns else []
        if len(durations) > 1:
            # Group by duration
            by_t_sec = benchmark_df.groupby("t_sec").agg({
                "gpac_calc_time": "mean",
                "tensorpac_calc_time": "mean"
            })
            
            # Plot lines
            by_t_sec.plot(marker="o", ax=ax5)
            
            # Set labels
            ax5.set_title("Computation Time vs Signal Duration")
            ax5.set_xlabel("Duration (seconds)")
            ax5.set_ylabel("Time (seconds)")
            ax5.legend(["gPAC", "Tensorpac"])
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No duration sweep data", ha='center', va='center', fontsize=12)
            ax5.set_title("Computation Time vs Signal Duration")
    
    # PLOT 6: Parameter sweep for phase bands
    ax6 = plt.subplot(gs[2, 0])
    
    # Filter for phase bands parameter sweep
    pha_sweep = benchmark_df[benchmark_df["param_sweep"] == "pha_n_bands"] if "param_sweep" in benchmark_df.columns else pd.DataFrame()
    
    if not pha_sweep.empty:
        # Create line plot
        pha_sweep_pivot = pha_sweep.pivot_table(
            index="pha_n_bands", 
            values=["gpac_calc_time", "tensorpac_calc_time"],
            aggfunc="mean"
        )
        
        # Plot lines
        pha_sweep_pivot.plot(marker="o", ax=ax6)
        
        # Set labels
        ax6.set_title("Computation Time vs Number of Phase Bands")
        ax6.set_xlabel("Number of Phase Bands")
        ax6.set_ylabel("Time (seconds)")
        ax6.legend(["gPAC", "Tensorpac"])
        ax6.grid(True, alpha=0.3)
    else:
        # Try to find any phase bands variation
        pha_bands = benchmark_df["pha_n_bands"].unique() if "pha_n_bands" in benchmark_df.columns else []
        if len(pha_bands) > 1:
            # Group by phase bands
            by_pha = benchmark_df.groupby("pha_n_bands").agg({
                "gpac_calc_time": "mean",
                "tensorpac_calc_time": "mean"
            })
            
            # Plot lines
            by_pha.plot(marker="o", ax=ax6)
            
            # Set labels
            ax6.set_title("Computation Time vs Number of Phase Bands")
            ax6.set_xlabel("Number of Phase Bands")
            ax6.set_ylabel("Time (seconds)")
            ax6.legend(["gPAC", "Tensorpac"])
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "No phase bands sweep data", ha='center', va='center', fontsize=12)
            ax6.set_title("Computation Time vs Number of Phase Bands")
    
    # PLOT 7: Parameter sweep for amplitude bands
    ax7 = plt.subplot(gs[2, 1])
    
    # Filter for amplitude bands parameter sweep
    amp_sweep = benchmark_df[benchmark_df["param_sweep"] == "amp_n_bands"] if "param_sweep" in benchmark_df.columns else pd.DataFrame()
    
    if not amp_sweep.empty:
        # Create line plot
        amp_sweep_pivot = amp_sweep.pivot_table(
            index="amp_n_bands", 
            values=["gpac_calc_time", "tensorpac_calc_time"],
            aggfunc="mean"
        )
        
        # Plot lines
        amp_sweep_pivot.plot(marker="o", ax=ax7)
        
        # Set labels
        ax7.set_title("Computation Time vs Number of Amplitude Bands")
        ax7.set_xlabel("Number of Amplitude Bands")
        ax7.set_ylabel("Time (seconds)")
        ax7.legend(["gPAC", "Tensorpac"])
        ax7.grid(True, alpha=0.3)
    else:
        # Try to find any amplitude bands variation
        amp_bands = benchmark_df["amp_n_bands"].unique() if "amp_n_bands" in benchmark_df.columns else []
        if len(amp_bands) > 1:
            # Group by amplitude bands
            by_amp = benchmark_df.groupby("amp_n_bands").agg({
                "gpac_calc_time": "mean",
                "tensorpac_calc_time": "mean"
            })
            
            # Plot lines
            by_amp.plot(marker="o", ax=ax7)
            
            # Set labels
            ax7.set_title("Computation Time vs Number of Amplitude Bands")
            ax7.set_xlabel("Number of Amplitude Bands")
            ax7.set_ylabel("Time (seconds)")
            ax7.legend(["gPAC", "Tensorpac"])
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, "No amplitude bands sweep data", ha='center', va='center', fontsize=12)
            ax7.set_title("Computation Time vs Number of Amplitude Bands")
    
    # PLOT 8: Speedup factors
    ax8 = plt.subplot(gs[2, 2])
    
    # Create custom parameter categories for grouping
    params = []
    speedups = []
    
    for param in ["batch_size", "n_chs", "t_sec", "pha_n_bands", "amp_n_bands"]:
        if f"param_sweep" in benchmark_df.columns and any(benchmark_df["param_sweep"] == param):
            sweep_data = benchmark_df[benchmark_df["param_sweep"] == param]
            
            # Calculate average speedup
            param_speedup = sweep_data["speedup_factor"].mean()
            
            params.append(param)
            speedups.append(param_speedup)
        else:
            # Check if we have multiple values for this parameter
            if param in benchmark_df.columns and len(benchmark_df[param].unique()) > 1:
                # Group by this parameter
                by_param = benchmark_df.groupby(param).apply(
                    lambda x: x["gpac_calc_time"].mean() / x["tensorpac_calc_time"].mean()
                    if x["tensorpac_calc_time"].mean() > 0 else 0
                )
                
                # Calculate average speedup
                param_speedup = by_param.mean()
                
                params.append(param)
                speedups.append(param_speedup)
    
    # If we have data, plot it
    if params:
        # Convert parameter names to more readable labels
        param_labels = {
            "batch_size": "Batch Size",
            "n_chs": "Channels",
            "t_sec": "Duration",
            "pha_n_bands": "Phase Bands",
            "amp_n_bands": "Amp. Bands"
        }
        
        # Plot bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
        bars = ax8.bar([param_labels.get(p, p) for p in params], speedups, color=colors, alpha=0.7)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f"{speedups[i]:.1f}x", ha='center', va='bottom')
        
        # Set labels
        ax8.set_title("gPAC Speedup Factors by Parameter")
        ax8.set_ylabel("Speedup Factor (Higher is Better)")
        ax8.grid(True, alpha=0.3, axis="y")
        
        # Add horizontal line at y=1.0 (no speedup)
        ax8.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        
        # Set y-axis to start at 0
        ax8.set_ylim(bottom=0)
        
    else:
        ax8.text(0.5, 0.5, "No parameter sweep data available", 
                ha='center', va='center', fontsize=12)
        ax8.set_title("gPAC Speedup Factors by Parameter")
    
    # Main title
    plt.suptitle("Performance Benchmark: gPAC vs Tensorpac", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "figure_3_performance_benchmark.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, "figure_3_performance_benchmark.pdf")
    plt.savefig(output_file_pdf, format="pdf", bbox_inches="tight")
    
    # Also save as SVG
    output_file_svg = os.path.join(output_dir, "figure_3_performance_benchmark.svg")
    plt.savefig(output_file_svg, format="svg", bbox_inches="tight")
    
    # Close figure to free memory
    plt.close(fig)
    
    return output_file

def main(args):
    """Main function for generating comparison figures."""
    # Setup result directory
    script_name = os.path.splitext(os.path.basename(__FILE__))[0]
    script_dir = os.path.join(__DIR__, f"{script_name}_out")
    os.makedirs(script_dir, exist_ok=True)
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14
    })
    
    # Create output directory in data
    data_dir = os.path.join("data", "exp_02")
    os.makedirs(data_dir, exist_ok=True)
    
    # Find and load benchmark data
    benchmark_files = find_benchmark_results(data_dir)
    print(f"Found {len(benchmark_files)} benchmark result files")
    
    benchmark_data, benchmark_df = load_benchmark_data(benchmark_files)
    print(f"Loaded benchmark data with {len(benchmark_df)} records")
    
    # Load PAC comparison data
    pac_df, metrics = load_pac_comparison_data(data_dir)
    print(f"Loaded PAC comparison data with {len(pac_df)} records")
    
    # Create figures
    print("Creating Figure 1: PAC Workflow Diagram")
    fig1_path = figure_1_pac_workflow(script_dir)
    
    print("Creating Figure 2: Calculation Accuracy")
    fig2_path = figure_2_calculation_accuracy(pac_df, metrics, script_dir)
    
    print("Creating Figure 3: Performance Benchmark")
    fig3_path = figure_3_performance_benchmark(benchmark_df, script_dir)
    
    # Create symlinks in data directory
    print("Creating symlinks in data directory...")
    import mngs
    for file_name in ["figure_1_pac_workflow.png", "figure_2_calculation_accuracy.png", "figure_3_performance_benchmark.png",
                     "figure_1_pac_workflow.pdf", "figure_2_calculation_accuracy.pdf", "figure_3_performance_benchmark.pdf",
                     "figure_1_pac_workflow.svg", "figure_2_calculation_accuracy.svg", "figure_3_performance_benchmark.svg"]:
        source_file = os.path.join(script_dir, file_name)
        target_file = os.path.join(data_dir, file_name)
        # Only create symlink if source file exists
        if os.path.exists(source_file):
            mngs.io.save(None, target_file, symlink_from=source_file)
    
    # Print summary
    print("\nFigure generation complete!")
    if fig1_path:
        print(f"Figure 1 saved to: {fig1_path}")
    if fig2_path:
        print(f"Figure 2 saved to: {fig2_path}")
    if fig3_path:
        print(f"Figure 3 saved to: {fig3_path}")
    
    return 0

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate comparison figures")
    
    parser.add_argument(
        "--data_dir", type=str, default="./data/exp_02",
        help="Directory containing benchmark and PAC comparison data"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save generated figures (defaults to script_out dir)"
    )
    parser.add_argument(
        "--format", type=str, default="all", 
        choices=["png", "pdf", "svg", "all"],
        help="Output format for figures"
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

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

if __name__ == '__main__':
    run_main()

# EOF