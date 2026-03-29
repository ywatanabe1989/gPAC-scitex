#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-17 15:18:21 (ywatanabe)"
# File: ./scripts/exp_01_synthetic_data_preparation/DataGenerator.py
# ----------------------------------------
import os

import scitex as stx

__FILE__ = os.path.abspath(__file__)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Dict, List, Tuple

import numpy as np
import torch


class DataGenerator:
    """
    Class for generating synthetic Phase-Amplitude Coupling (PAC) signals
    with different coupling properties.
    """

    def __init__(self, fs=1000.0, random_seed=None):
        """
        Initialize the data generator.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.fs = fs
        if random_seed is not None:
            np.random.seed(random_seed)

        # Default class frequency ranges
        self.default_class_params = {
            # Class 1: Low phase freq + Low amp freq
            "class1_pha_range": [4.0, 8.0],
            "class1_amp_range": [50.0, 70.0],
            # Class 2: Low phase freq + High amp freq
            "class2_pha_range": [4.0, 8.0],
            "class2_amp_range": [150.0, 170.0],
            # Class 3: Medium phase freq + Medium amp freq
            "class3_pha_range": [9.0, 14.0],
            "class3_amp_range": [100.0, 120.0],
            # Class 4: High phase freq + Low amp freq
            "class4_pha_range": [15.0, 20.0],
            "class4_amp_range": [50.0, 70.0],
            # Class 5: High phase freq + High amp freq
            "class5_pha_range": [15.0, 20.0],
            "class5_amp_range": [150.0, 170.0],
        }

        # Default generation parameters
        self.default_params = {
            "fs": self.fs,
            "duration": 2.0,
            "n_samples": 200,
            "n_channels": 4,
            "n_segments": 1,
            "n_classes": 5,
            "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5],
            "coupling_strengths": [0.6, 0.7, 0.8, 0.9],
        }
        self.default_params.update(self.default_class_params)

    def generate_pac_signal(
        self,
        n_seconds: float,
        pha_mod_freq: float,
        amp_car_freq: float,
        pha_bandwidth: float,
        amp_bandwidth: float,
        coupling_strength: float = 0.8,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Generate a single PAC signal with specified coupling relationship.

        Parameters
        ----------
        n_seconds : float
            Duration of the signal in seconds
        pha_mod_freq : float
            Frequency of the phase modulator in Hz
        amp_car_freq : float
            Frequency of the amplitude carrier in Hz
        pha_bandwidth : float
            Bandwidth of the phase component
        amp_bandwidth : float
            Bandwidth of the amplitude component
        coupling_strength : float
            Strength of the coupling between phase and amplitude (0-1)
        noise_level : float
            Level of random noise to add to the signal

        Returns
        -------
        numpy.ndarray
            The generated PAC signal
        """
        # Create time vector
        t = np.arange(0, n_seconds, 1 / self.fs)

        # Create phase signal (slow oscillation)
        phase_signal = np.sin(2 * np.pi * pha_mod_freq * t)

        # Create amplitude Modulation based on phase
        Modulation = (
            1 + coupling_strength * np.cos(2 * np.pi * pha_mod_freq * t)
        ) / 2

        # Create carrier signal (fast oscillation)
        carrier = np.sin(2 * np.pi * amp_car_freq * t)

        # Apply amplitude Modulation to carrier
        modulated_carrier = Modulation * carrier

        # Create final signal with both components
        pac_signal = phase_signal + modulated_carrier

        # Add noise
        noise = np.random.normal(0, noise_level, len(t))
        signal = pac_signal + noise

        return signal

    def _generate_class_signals(
        self,
        class_id: int,
        pha_range: List[float],
        amp_range: List[float],
        params: Dict,
        start_idx: int,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate signals for a specific class with given phase and amplitude ranges.

        Parameters
        ----------
        class_id : int
            Class identifier (0-4 for 5 classes)
        pha_range : List[float]
            [min, max] range for phase frequency in Hz
        amp_range : List[float]
            [min, max] range for amplitude frequency in Hz
        params : Dict
            Parameters for signal generation
        start_idx : int
            Starting index for this class in the overall array

        Returns
        -------
        Tuple[np.ndarray, Dict]
            Generated signals and corresponding metadata
        """
        # Calculate sizes
        n_noise_levels = len(params["noise_levels"])
        n_coupling = len(params["coupling_strengths"])
        n_samples_per_class = params["n_samples"]

        # Handle special cases for sample sizes
        if n_samples_per_class <= 0:
            # Return empty arrays if no samples requested
            return self._create_empty_class_data(params)

        # Handle very small sample sizes
        if n_samples_per_class < n_noise_levels * n_coupling:
            # Use as many conditions as we have samples
            samples_per_condition = 1
            total_samples = min(
                n_samples_per_class, n_noise_levels * n_coupling
            )
        else:
            # Distribute samples across all conditions
            samples_per_condition = n_samples_per_class // (
                n_noise_levels * n_coupling
            )
            total_samples = samples_per_condition * n_noise_levels * n_coupling

        # Initialize arrays
        seq_len = int(params["duration"] * params["fs"])
        signals = np.zeros(
            (
                total_samples,
                params["n_channels"],
                params["n_segments"],
                seq_len,
            )
        )

        # Initialize metadata
        metadata = {
            "class_labels": np.full(total_samples, class_id, dtype=int),
            "pha_freqs": np.zeros(total_samples),
            "amp_freqs": np.zeros(total_samples),
            "noise_levels": np.zeros(total_samples),
            "coupling_strengths": np.zeros(total_samples),
            "sample_ids": np.zeros(total_samples, dtype=int),
        }

        # Generate signals
        sample_idx = 0
        condition_count = 0

        for noise_idx, noise_level in enumerate(params["noise_levels"]):
            if sample_idx >= total_samples:
                break

            for coupling_idx, coupling_strength in enumerate(
                params["coupling_strengths"]
            ):
                if sample_idx >= total_samples:
                    break

                # For small sample sizes, track conditions used
                if n_samples_per_class < n_noise_levels * n_coupling:
                    if condition_count >= total_samples:
                        break
                    condition_count += 1

                for sample_count in range(samples_per_condition):
                    # Random frequencies from ranges
                    pha_freq = np.random.uniform(pha_range[0], pha_range[1])
                    amp_freq = np.random.uniform(amp_range[0], amp_range[1])

                    # Generate base PAC signal
                    pac_data = self.generate_pac_signal(
                        n_seconds=params["duration"],
                        pha_mod_freq=pha_freq,
                        amp_car_freq=amp_freq,
                        pha_bandwidth=pha_freq / 4,
                        amp_bandwidth=amp_freq / 4,
                        coupling_strength=coupling_strength,
                        noise_level=noise_level,
                    )

                    # Create multiple channels with variations
                    for channel_idx in range(params["n_channels"]):
                        channel_variation = np.random.normal(
                            0, 0.02, len(pac_data)
                        )
                        channel_signal = pac_data + channel_variation

                        # Create multiple segments with variations if needed
                        for segment_idx in range(params["n_segments"]):
                            if params["n_segments"] > 1:
                                segment_variation = np.random.normal(
                                    0, 0.01, len(pac_data)
                                )
                                segment_signal = (
                                    channel_signal + segment_variation
                                )
                            else:
                                segment_signal = channel_signal

                            signals[
                                sample_idx, channel_idx, segment_idx, :
                            ] = segment_signal

                    # Store metadata
                    metadata["pha_freqs"][sample_idx] = pha_freq
                    metadata["amp_freqs"][sample_idx] = amp_freq
                    metadata["noise_levels"][sample_idx] = noise_level
                    metadata["coupling_strengths"][
                        sample_idx
                    ] = coupling_strength
                    metadata["sample_ids"][sample_idx] = sample_count

                    sample_idx += 1

                    # Progress update
                    global_idx = start_idx + sample_idx
                    if global_idx % 50 == 0:
                        print(f"Generated {global_idx} signals")

        return signals, metadata

    def _create_empty_class_data(
        self, params: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Create empty data structure for a class with no samples."""
        seq_len = int(params["duration"] * params["fs"])
        empty_signals = np.zeros(
            (0, params["n_channels"], params["n_segments"], seq_len)
        )
        empty_metadata = {
            "class_labels": np.array([], dtype=int),
            "pha_freqs": np.array([]),
            "amp_freqs": np.array([]),
            "noise_levels": np.array([]),
            "coupling_strengths": np.array([]),
            "sample_ids": np.array([], dtype=int),
        }
        return empty_signals, empty_metadata

    def generate_dataset(self, custom_params: Dict = None) -> Dict:
        """
        Generate a complete synthetic PAC dataset for multiple classes.

        Parameters
        ----------
        custom_params : Dict, optional
            Custom parameters to override defaults

        Returns
        -------
        Dict
            Dictionary containing generated signals and metadata
        """
        # Merge default and custom parameters
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)

        # Calculate total expected samples
        n_classes = params["n_classes"]
        n_samples_per_class = params["n_samples"]
        n_noise_levels = len(params["noise_levels"])
        n_coupling = len(params["coupling_strengths"])

        # Handle empty dataset case
        if n_samples_per_class <= 0:
            empty_signals, empty_metadata = self._create_empty_class_data(
                params
            )
            return {
                "signals": empty_signals,
                "class_labels": empty_metadata["class_labels"],
                "pha_freqs": empty_metadata["pha_freqs"],
                "amp_freqs": empty_metadata["amp_freqs"],
                "noise_levels": empty_metadata["noise_levels"],
                "coupling_strengths": empty_metadata["coupling_strengths"],
                "sample_ids": empty_metadata["sample_ids"],
                "param_info": params,
            }

        # Calculate actual samples per class based on conditions
        if n_samples_per_class < n_noise_levels * n_coupling:
            class_total_samples = min(
                n_samples_per_class, n_noise_levels * n_coupling
            )
            total_samples = class_total_samples * n_classes
        else:
            samples_per_condition = max(
                1, n_samples_per_class // (n_noise_levels * n_coupling)
            )
            total_samples = (
                samples_per_condition * n_noise_levels * n_coupling * n_classes
            )

        print(
            f"Generating {total_samples} total signals ({n_samples_per_class} per class, {n_classes} classes)"
        )

        # Pre-allocate arrays
        seq_len = int(params["duration"] * params["fs"])
        signals_np = np.zeros(
            (
                total_samples,
                params["n_channels"],
                params["n_segments"],
                seq_len,
            )
        )

        # Initialize metadata
        metadata = {
            "class_labels": np.zeros(total_samples, dtype=int),
            "pha_freqs": np.zeros(total_samples),
            "amp_freqs": np.zeros(total_samples),
            "noise_levels": np.zeros(total_samples),
            "coupling_strengths": np.zeros(total_samples),
            "sample_ids": np.zeros(total_samples, dtype=int),
        }

        # Generate signals for each class
        idx_end = 0

        # Generate data for each class
        for class_id in range(n_classes):
            class_name = f"class{class_id+1}"
            pha_key = f"{class_name}_pha_range"
            amp_key = f"{class_name}_amp_range"

            if pha_key in params and amp_key in params:
                print(f"\nGenerating Class {class_id+1}")
                signals_class, metadata_class = self._generate_class_signals(
                    class_id,
                    params[pha_key],
                    params[amp_key],
                    params,
                    idx_end,
                )

                idx_start = idx_end
                idx_end = idx_start + len(signals_class)

                signals_np[idx_start:idx_end] = signals_class
                for key in metadata:
                    metadata[key][idx_start:idx_end] = metadata_class[key]
            else:
                print(
                    f"Skipping Class {class_id+1} - missing frequency range parameters"
                )

        # Shuffle the data
        print("\nShuffling the dataset...")
        indices = np.arange(idx_end)  # Only shuffle the filled portion
        np.random.shuffle(indices)
        signals_np = signals_np[indices]
        for key in metadata:
            metadata[key] = metadata[key][indices]

        # Create final data dictionary
        result = {
            "signals": signals_np,
            "class_labels": metadata["class_labels"],
            "pha_freqs": metadata["pha_freqs"],
            "amp_freqs": metadata["amp_freqs"],
            "noise_levels": metadata["noise_levels"],
            "coupling_strengths": metadata["coupling_strengths"],
            "sample_ids": metadata["sample_ids"],
            "param_info": params,
        }

        return result

    def create_torch_datasets(
        self, data: Dict, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Dict[str, "SyntheticPACDataset"]:
        """
        Split generated data into train/val/test datasets.

        Parameters
        ----------
        data : Dict
            Data dictionary from generate_dataset()
        train_ratio : float
            Proportion of data for training
        val_ratio : float
            Proportion of data for validation

        Returns
        -------
        Dict[str, SyntheticPACDataset]
            Dictionary containing train, val, and test datasets
        """
        from .SyntheticPACDataset import SyntheticPACDataset

        # Convert data to torch tensors if needed
        signals_tensor = torch.tensor(data["signals"], dtype=torch.float32)
        labels_tensor = torch.tensor(data["class_labels"], dtype=torch.long)

        # Initialize metadata tensors
        metadata_tensors = {
            "pha_freqs": torch.tensor(data["pha_freqs"], dtype=torch.float32),
            "amp_freqs": torch.tensor(data["amp_freqs"], dtype=torch.float32),
            "noise_levels": torch.tensor(
                data["noise_levels"], dtype=torch.float32
            ),
            "coupling_strengths": torch.tensor(
                data["coupling_strengths"], dtype=torch.float32
            ),
            "sample_ids": torch.tensor(data["sample_ids"], dtype=torch.long),
        }

        # Get total number of samples
        n_samples = len(signals_tensor)

        # Handle empty dataset
        if n_samples == 0:
            empty_dataset = SyntheticPACDataset(
                signals=torch.zeros((0, 4, 1, 2000), dtype=torch.float32),
                labels=torch.zeros(0, dtype=torch.long),
                metadata={k: torch.zeros(0) for k in metadata_tensors},
            )
            return {
                "train": empty_dataset,
                "val": empty_dataset,
                "test": empty_dataset,
                "indices": {
                    "train": np.array([]),
                    "val": np.array([]),
                    "test": np.array([]),
                },
            }

        # Create stratified split indices
        unique_classes = torch.unique(labels_tensor)
        train_indices = []
        val_indices = []
        test_indices = []

        for class_id in unique_classes:
            # Get indices for this class
            class_mask = labels_tensor == class_id
            class_indices = torch.where(class_mask)[0].numpy()
            np.random.shuffle(class_indices)

            # Calculate split sizes
            n_class_samples = len(class_indices)

            # Handle small sample counts
            if n_class_samples <= 1:
                train_indices.extend(class_indices)
                continue

            if n_class_samples == 2:
                train_indices.append(class_indices[0])
                val_indices.append(class_indices[1])
                continue

            # Normal case - proportional splits
            n_train = max(1, int(n_class_samples * train_ratio))
            n_val = max(1, int(n_class_samples * val_ratio))

            # Ensure at least one test sample if possible
            if n_train + n_val >= n_class_samples and n_class_samples > 2:
                n_val = max(1, n_class_samples - n_train - 1)

            # Adjust if exceeding sample count
            if n_train + n_val > n_class_samples:
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val -= 1

            # Split indices
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train : n_train + n_val])
            test_indices.extend(class_indices[n_train + n_val :])

        # Convert indices to arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Create datasets
        train_dataset = SyntheticPACDataset(
            signals=signals_tensor[train_indices],
            labels=labels_tensor[train_indices],
            metadata={
                k: v[train_indices] for k, v in metadata_tensors.items()
            },
        )

        val_dataset = SyntheticPACDataset(
            signals=signals_tensor[val_indices],
            labels=labels_tensor[val_indices],
            metadata={k: v[val_indices] for k, v in metadata_tensors.items()},
        )

        test_dataset = SyntheticPACDataset(
            signals=signals_tensor[test_indices],
            labels=labels_tensor[test_indices],
            metadata={k: v[test_indices] for k, v in metadata_tensors.items()},
        )

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "indices": {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            },
        }

# EOF
