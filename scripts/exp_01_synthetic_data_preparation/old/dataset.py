#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 22:12:58 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/exp_01_synthetic_data_preparation/dataset.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/exp_01_synthetic_data_preparation/dataset.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Provides PyTorch dataset implementation for synthetic PAC signals
  - Creates dataset splits for training, validation, and testing
  - Handles metadata for signals with different PAC characteristics

Dependencies:
  - packages:
    - mngs
    - numpy
    - torch
"""

"""Imports"""
from typing import Dict

import mngs
import numpy as np
import torch
from torch.utils.data import Dataset


"""Functions & Classes"""
class SyntheticPACDataset(Dataset):
    """
    PyTorch Dataset for synthetic PAC signals.

    This dataset can be used across different experiments consistently.
    """

    def __init__(self, signals, labels, metadata=None):
        """
        Initialize the dataset with signals and labels.

        Parameters
        ----------
        signals : torch.Tensor
            Tensor of shape (n_samples, n_channels, n_segments, seq_len)
        labels : torch.Tensor
            Tensor of shape (n_samples,) containing class labels
        metadata : dict, optional
            Additional metadata for each sample
        """
        self.signals = signals
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.metadata is not None:
            meta = {k: v[idx] for k, v in self.metadata.items()}
            return signal, label, meta

        return signal, label


def create_pac_dataset_splits(
    data: Dict,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    """
    Create train, validation, and test dataset splits.

    Parameters
    ----------
    data : Dict
        Dictionary containing generated signals and metadata
    output_dir : str
        Directory to save the output files
    train_ratio : float
        Proportion of data to use for training
    val_ratio : float
        Proportion of data to use for validation (rest goes to test)
    """
    # Get total number of samples
    n_samples = len(data["signals"])

    # Handle empty dataset
    if n_samples == 0:
        print("No samples available for dataset splits")
        empty_dataset = SyntheticPACDataset(
            signals=torch.zeros((0, 4, 1, 2000), dtype=torch.float32),
            labels=torch.zeros(0, dtype=torch.long),
            metadata={},
        )
        # Save empty datasets
        mngs.io.save(
            empty_dataset,
            "./data/exp_01/train_dataset.npz",
            symlink_from_cwd=True,
        )
        mngs.io.save(
            empty_dataset,
            "./data/exp_01/val_dataset.npz",
            symlink_from_cwd=True,
        )
        mngs.io.save(
            empty_dataset,
            "./data/exp_01/test_dataset.npz",
            symlink_from_cwd=True,
        )
        mngs.io.save(
            {
                "train_indices": np.array([]),
                "val_indices": np.array([]),
                "test_indices": np.array([]),
            },
            "./data/exp_01/dataset_splits.npz",
            symlink_from_cwd=True,
        )
        return

    # Create stratified split indices
    class_labels = data["class_labels"]
    unique_classes = np.unique(class_labels)
    n_classes = len(unique_classes)

    # Indices for each split by class
    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in unique_classes:
        # Get indices for this class
        class_indices = np.where(class_labels == class_id)[0]
        np.random.shuffle(class_indices)

        # Calculate split sizes
        n_class_samples = len(class_indices)

        # Handle small sample counts
        if n_class_samples <= 1:
            # If only one sample, use it for training
            train_indices.extend(class_indices)
            continue

        if n_class_samples == 2:
            # If two samples, use one for training, one for validation
            train_indices.append(class_indices[0])
            val_indices.append(class_indices[1])
            continue

        # Normal case - calculate proportional splits
        n_train = max(1, int(n_class_samples * train_ratio))
        n_val = max(1, int(n_class_samples * val_ratio))

        # Ensure we have at least one sample for test if there are enough samples
        if n_train + n_val >= n_class_samples and n_class_samples > 2:
            # Reduce validation to ensure at least one test sample
            n_val = max(1, n_class_samples - n_train - 1)

        # Adjust if we would exceed the number of samples
        if n_train + n_val > n_class_samples:
            if n_train > 1:
                n_train -= 1
            else:
                n_val -= 1

        # Split indices
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val :])

    # Shuffle the indices within each split
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Convert signals to torch tensors if they're not already
    signals_tensor = torch.from_numpy(data["signals"].astype(np.float32)) if isinstance(data["signals"], np.ndarray) else data["signals"]
    
    # Create the datasets
    train_data = {
        "signals": data["signals"][train_indices],
        "class_labels": data["class_labels"][train_indices],
        "pha_freqs": data["pha_freqs"][train_indices],
        "amp_freqs": data["amp_freqs"][train_indices],
        "noise_levels": data["noise_levels"][train_indices],
        "coupling_strengths": data["coupling_strengths"][train_indices],
        "sample_ids": data["sample_ids"][train_indices],
    }
    
    val_data = {
        "signals": data["signals"][val_indices],
        "class_labels": data["class_labels"][val_indices],
        "pha_freqs": data["pha_freqs"][val_indices],
        "amp_freqs": data["amp_freqs"][val_indices],
        "noise_levels": data["noise_levels"][val_indices],
        "coupling_strengths": data["coupling_strengths"][val_indices],
        "sample_ids": data["sample_ids"][val_indices],
    }
    
    test_data = {
        "signals": data["signals"][test_indices],
        "class_labels": data["class_labels"][test_indices],
        "pha_freqs": data["pha_freqs"][test_indices],
        "amp_freqs": data["amp_freqs"][test_indices],
        "noise_levels": data["noise_levels"][test_indices],
        "coupling_strengths": data["coupling_strengths"][test_indices],
        "sample_ids": data["sample_ids"][test_indices],
    }

    # Save the splits using mngs.io.save
    mngs.io.save(
        train_data,
        "./data/exp_01/train_dataset.npz",
        symlink_from_cwd=True,
    )
    mngs.io.save(
        val_data,
        "./data/exp_01/val_dataset.npz",
        symlink_from_cwd=True,
    )
    mngs.io.save(
        test_data,
        "./data/exp_01/test_dataset.npz",
        symlink_from_cwd=True,
    )

    # Save indices for reference using mngs.io.save
    mngs.io.save(
        {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        },
        "./data/exp_01/dataset_splits.npz",
        symlink_from_cwd=True,
    )

    print(f"Dataset splits created and saved to {output_dir}")
    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
# EOF