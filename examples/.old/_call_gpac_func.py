#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 13:22:15 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/_call_gpac_func.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/_call_gpac_func.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import gpac
import numpy as np
import torch
import matplotlib.pyplot as plt

# Parameters
FS = 1024
SEQ_LEN = FS * 5
BATCH_SIZE = 4
N_CHS = 16
N_SEGMENTS = 1

# Demo Signal
signal_np = np.random.randn(BATCH_SIZE, N_CHS, N_SEGMENTS, SEQ_LEN).astype(
    np.float32
)
signal_gpu = torch.from_numpy(signal_np).cuda()


# Example 1: Basic PAC calculation with permutation testing
try:
    pac_values, freqs_pha, freqs_amp = gpac.calculate_pac(
        signal=signal_gpu,
        fs=FS,
        pha_n_bands=50,
        amp_n_bands=30,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        device="cuda",
        fp16=True,
        n_perm=200,
        # trainable=False,    # Use static filters (default)
        # chunk_size=16       # Process in chunks (optional, if memory is limited)
    )

    print("Basic PAC calculation successful.")
    print(f"Input Signal Shape: {signal_gpu.shape}")
    print(f"Output PAC Tensor Shape: {pac_values.shape}")
    print(f"Phase Frequencies (Num): {len(freqs_pha)}")
    print(f"Amplitude Frequencies (Num): {len(freqs_amp)}")

except Exception as e:
    print(f"An error occurred during basic PAC calculation: {e}")


# Example 2: PAC calculation with surrogate distribution
try:
    # Calculate PAC with return_dist=True to get surrogate distribution
    pac_values, surrogate_dist, freqs_pha, freqs_amp = gpac.calculate_pac(
        signal=signal_gpu,
        fs=FS,
        pha_n_bands=10,              # Fewer bands for clearer visualization
        amp_n_bands=8,               # Fewer bands for clearer visualization
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        device="cuda",
        fp16=True,
        n_perm=200,                  # Number of permutations
        return_dist=True,            # Return surrogate distribution
        average_channels=True,       # Average across channels for simplicity
    )

    print("\nPAC calculation with surrogate distribution successful.")
    print(f"PAC Values Shape: {pac_values.shape}")
    print(f"Surrogate Distribution Shape: {surrogate_dist.shape}")
    
    # Example: Plot histogram of surrogate distribution for a specific frequency pair
    if plt:
        pha_idx, amp_idx = 0, 0  # First frequency pair
        plt.figure(figsize=(10, 6))
        
        # Plot the observed PAC value as a vertical line
        observed_pac = pac_values[0, pha_idx, amp_idx].cpu().numpy()
        plt.axvline(x=observed_pac, color='r', linestyle='-', 
                    label=f'Observed PAC: {observed_pac:.4f}')
        
        # Plot the surrogate distribution histogram
        surrogates = surrogate_dist[:, 0, pha_idx, amp_idx].cpu().numpy()
        plt.hist(surrogates, bins=20, alpha=0.7)
        
        # Calculate empirical p-value
        p_value = (surrogates >= observed_pac).mean()
        
        plt.title(f'PAC Surrogate Distribution\nPhase: {freqs_pha[pha_idx]:.1f}Hz, '
                  f'Amplitude: {freqs_amp[amp_idx]:.1f}Hz, p={p_value:.4f}')
        plt.xlabel('PAC Value')
        plt.ylabel('Count')
        plt.legend()
        
        # Save plot instead of showing (for headless environments)
        plt.savefig('pac_surrogate_distribution.png')
        print(f"Surrogate distribution plot saved to 'pac_surrogate_distribution.png'")
        plt.close()
    
    # Example: Calculate custom p-values from surrogate distribution
    exceeds = (surrogate_dist > pac_values.unsqueeze(0)).float().mean(dim=0)
    p_values = exceeds  # Proportion of surrogates exceeding observed value
    
    # Apply custom statistical threshold (e.g., p < 0.05)
    significant_pac = pac_values * (p_values < 0.05).float()
    
    print(f"Custom p-value calculation complete. Shape: {p_values.shape}")
    print(f"Significant PAC values found: {(significant_pac > 0).sum().item()}")

except Exception as e:
    print(f"An error occurred during PAC with distribution calculation: {e}")

# EOF