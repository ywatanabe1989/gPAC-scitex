#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-17 15:18:35 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/exp_01_synthetic_data_preparation/SyntheticPACDataset.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/exp_01_synthetic_data_preparation/SyntheticPACDataset.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from torch.utils.data import Dataset

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

# EOF
