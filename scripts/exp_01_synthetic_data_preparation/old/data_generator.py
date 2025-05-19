#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-17 15:13:58 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/exp_01_synthetic_data_preparation/data_generator.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/exp_01_synthetic_data_preparation/data_generator.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = os.path.abspath(__file__)

"""
Functionalities:
  - Generates synthetic Phase-Amplitude Coupling (PAC) signals
  - Creates signals with various coupling characteristics
  - Supports customizable signal parameters (frequency ranges, noise, etc.)

Dependencies:
  - packages:
    - mngs
    - numpy
"""

"""Imports"""
from typing import Dict, List, Tuple

import numpy as np


# DataGenerator class for creating synthetic PAC signals
class DataGenerator:
    """
    Class for generating synthetic PAC signals with different coupling properties.
    """

    def __init__(self, fs=1000.0, random_seed=None):
        self.fs = fs
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_pac_with_signal(
        self,
        n_seconds,
        pha_mod_freq,
        amp_car_freq,
        pha_bandwidth,
        amp_bandwidth,
        coupling_strength=0.8,
        noise_level=0.1,
    ):
        """
        Generate a PAC signal with specified coupling relationship.

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


def generate_class_signals(
    data_gen: DataGenerator,
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
    data_gen : DataGenerator
        Instance of DataGenerator to create signals
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

    # Handle small sample sizes
    if n_samples_per_class <= 0:
        # If n_samples is 0 or negative, return empty arrays
        return np.zeros(
            (
                0,
                params["n_channels"],
                params["n_segments"],
                int(params["duration"] * params["fs"]),
            )
        ), {
            "class_labels": np.array([], dtype=int),
            "pha_freqs": np.array([]),
            "amp_freqs": np.array([]),
            "noise_levels": np.array([]),
            "coupling_strengths": np.array([]),
            "sample_ids": np.array([], dtype=int),
        }

    # Special case for very small sample sizes
    if n_samples_per_class < n_noise_levels * n_coupling:
        # Create at most n_samples_per_class samples (one per condition up to the limit)
        samples_per_condition = 1
        # We'll use as many conditions as we have samples
        total_samples = min(n_samples_per_class, n_noise_levels * n_coupling)
    else:
        # Normal case - distribute samples across all conditions
        samples_per_condition = n_samples_per_class // (
            n_noise_levels * n_coupling
        )
        total_samples = samples_per_condition * n_noise_levels * n_coupling

    seq_len = int(params["duration"] * params["fs"])
    signals = np.zeros(
        (total_samples, params["n_channels"], params["n_segments"], seq_len)
    )

    # Metadata storage
    metadata = {
        "class_labels": np.full(total_samples, class_id, dtype=int),
        "pha_freqs": np.zeros(total_samples),
        "amp_freqs": np.zeros(total_samples),
        "noise_levels": np.zeros(total_samples),
        "coupling_strengths": np.zeros(total_samples),
        "sample_ids": np.zeros(total_samples, dtype=int),
    }

    # Generate signals for this class
    sample_idx = 0
    condition_count = 0

    # For small sample sizes, limit the number of conditions we use
    for noise_idx, noise_level in enumerate(params["noise_levels"]):
        if sample_idx >= total_samples:
            break

        for coupling_idx, coupling_strength in enumerate(
            params["coupling_strengths"]
        ):
            if sample_idx >= total_samples:
                break

            # For very small sample sizes, we track how many conditions we've used
            if n_samples_per_class < n_noise_levels * n_coupling:
                if condition_count >= total_samples:
                    break
                condition_count += 1

            for i in range(samples_per_condition):
                # Random frequencies from ranges
                pha_freq = np.random.uniform(pha_range[0], pha_range[1])
                amp_freq = np.random.uniform(amp_range[0], amp_range[1])

                # Generate PAC signal
                pac_data = data_gen.generate_pac_with_signal(
                    n_seconds=params["duration"],
                    pha_mod_freq=pha_freq,
                    amp_car_freq=amp_freq,
                    pha_bandwidth=pha_freq / 4,
                    amp_bandwidth=amp_freq / 4,
                    coupling_strength=coupling_strength,
                    noise_level=noise_level,
                )

                # Reshape to match expected dimensions and create multiple channels/segments
                signal_1d = pac_data.reshape(1, 1, -1)

                # Repeat across channels - make slightly different versions
                for channel_idx in range(params["n_channels"]):
                    # Add small channel-specific variations
                    channel_variation = np.random.normal(
                        0, 0.02, len(pac_data)
                    )
                    channel_signal = pac_data + channel_variation

                    for segment_idx in range(params["n_segments"]):
                        # Add small segment-specific variations if multiple segments
                        if params["n_segments"] > 1:
                            segment_variation = np.random.normal(
                                0, 0.01, len(pac_data)
                            )
                            segment_signal = channel_signal + segment_variation
                        else:
                            segment_signal = channel_signal

                        signals[sample_idx, channel_idx, segment_idx, :] = (
                            segment_signal
                        )

                # Store metadata
                metadata["pha_freqs"][sample_idx] = pha_freq
                metadata["amp_freqs"][sample_idx] = amp_freq
                metadata["noise_levels"][sample_idx] = noise_level
                metadata["coupling_strengths"][sample_idx] = coupling_strength
                metadata["sample_ids"][sample_idx] = i

                sample_idx += 1

                # Progress update
                global_idx = start_idx + sample_idx
                if global_idx % 50 == 0:
                    print(f"Generated {global_idx} signals")

    return signals, metadata


def generate_synthetic_pac_signals(params: Dict) -> Dict:
    """
    Generate synthetic PAC signals for multiple classes.

    Parameters
    ----------
    params : Dict
        Dictionary with signal generation parameters

    Returns
    -------
    Dict
        Dictionary containing generated signals and metadata
    """
    # Set up data generator
    data_gen = DataGenerator(fs=params["fs"], random_seed=42)

    # Calculate total signals with handling for small sample sizes
    n_noise_levels = len(params["noise_levels"])
    n_coupling = len(params["coupling_strengths"])
    n_classes = params["n_classes"]
    n_samples_per_class = params["n_samples"]

    # Handle small sample sizes
    if n_samples_per_class <= 0:
        # Empty dataset case
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
        return {
            "signals_np": empty_signals,
            "metadata": empty_metadata,
            "params": params,
        }

    if n_samples_per_class < n_noise_levels * n_coupling:
        # Each class gets n_samples_per_class samples with one sample per condition used
        samples_per_condition = 1
        # We'll use as many conditions as we have samples
        class_total_samples = min(
            n_samples_per_class, n_noise_levels * n_coupling
        )
        total_samples = class_total_samples * n_classes
    else:
        # Normal case - distribute samples across all conditions
        samples_per_condition = max(
            1, n_samples_per_class // (n_noise_levels * n_coupling)
        )
        total_samples = (
            samples_per_condition * n_noise_levels * n_coupling * n_classes
        )

    print(
        f"Generating {total_samples} total signals "
        + f"({n_samples_per_class} per class, {n_classes} classes)"
    )

    # Pre-allocate final arrays
    seq_len = int(params["duration"] * params["fs"])
    signals_np = np.zeros(
        (total_samples, params["n_channels"], params["n_segments"], seq_len)
    )

    # Metadata dictionary for the entire dataset
    metadata = {
        "class_labels": np.zeros(total_samples, dtype=int),
        "pha_freqs": np.zeros(total_samples),
        "amp_freqs": np.zeros(total_samples),
        "noise_levels": np.zeros(total_samples),
        "coupling_strengths": np.zeros(total_samples),
        "sample_ids": np.zeros(total_samples, dtype=int),
    }

    # Generate signals for each class
    class_size = n_samples_per_class

    # Class 1: Low phase freq + Low amp freq
    print("\nGenerating Class 1: Low phase freq + Low amp freq")
    signals_class1, metadata_class1 = generate_class_signals(
        data_gen,
        0,
        params["class1_pha_range"],
        params["class1_amp_range"],
        params,
        0,
    )
    idx_start = 0
    idx_end = len(signals_class1)
    signals_np[idx_start:idx_end] = signals_class1
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class1[key]

    # Class 2: Low phase freq + High amp freq
    print("\nGenerating Class 2: Low phase freq + High amp freq")
    signals_class2, metadata_class2 = generate_class_signals(
        data_gen,
        1,
        params["class2_pha_range"],
        params["class2_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class2)
    signals_np[idx_start:idx_end] = signals_class2
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class2[key]

    # Class 3: Medium phase freq + Medium amp freq
    print("\nGenerating Class 3: Medium phase freq + Medium amp freq")
    signals_class3, metadata_class3 = generate_class_signals(
        data_gen,
        2,
        params["class3_pha_range"],
        params["class3_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class3)
    signals_np[idx_start:idx_end] = signals_class3
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class3[key]

    # Class 4: High phase freq + Low amp freq
    print("\nGenerating Class 4: High phase freq + Low amp freq")
    signals_class4, metadata_class4 = generate_class_signals(
        data_gen,
        3,
        params["class4_pha_range"],
        params["class4_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class4)
    signals_np[idx_start:idx_end] = signals_class4
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class4[key]

    # Class 5: High phase freq + High amp freq
    print("\nGenerating Class 5: High phase freq + High amp freq")
    signals_class5, metadata_class5 = generate_class_signals(
        data_gen,
        4,
        params["class5_pha_range"],
        params["class5_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class5)
    signals_np[idx_start:idx_end] = signals_class5
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class5[key]

    # Shuffle the data while maintaining correspondence between signals and metadata
    print("\nShuffling the dataset...")
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    signals_np = signals_np[indices]
    for key in metadata:
        metadata[key] = metadata[key][indices]

    # Create a dictionary with all the data to save
    save_data = {
        "signals": signals_np,
        "class_labels": metadata["class_labels"],
        "pha_freqs": metadata["pha_freqs"],
        "amp_freqs": metadata["amp_freqs"],
        "noise_levels": metadata["noise_levels"],
        "coupling_strengths": metadata["coupling_strengths"],
        "sample_ids": metadata["sample_ids"],
    }
    # Add parameters with param_ prefix
    for k, v in params.items():
        save_data[f"param_{k}"] = v

    # Return complete dataset with numpy arrays
    return save_data

# EOF
