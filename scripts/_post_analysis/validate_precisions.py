#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-06 02:36:18 (ywatanabe)"
# File: ./torchPAC/scripts/post_analysis/validate_precisions.py

"""This script does XYZ."""

import sys
from typing import Tuple

import matplotlib.pyplot as plt
import mngs
import numpy as np
import scipy
import tensorpac
from numpy.typing import NDArray


# Functions
def calc_pac_with_tensorpac(
    xx: NDArray,
    fs: int,
    t_sec: float,
    i_batch: int = 0,
    i_ch: int = 0,
    n_perm=200,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Calculates Phase-Amplitude Coupling using tensorpac package.

    Parameters
    ----------
    xx : NDArray
        Input signal array with shape (batch_size, n_channels, time_points)
    fs : int
        Sampling frequency in Hz
    t_sec : float
        Duration of signal in seconds
    i_batch : int
        Batch index to process
    i_ch : int
        Channel index to process

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]
        phases: Phase values from wavelet transform
        amplitudes: Amplitude values from wavelet transform
        freqs_pha: Phase frequencies
        freqs_amp: Amplitude frequencies
        pac: Phase-amplitude coupling values
    """

    # Morlet's Wavelet Transfrmation
    p = tensorpac.Pac(f_pha="hres", f_amp="hres", dcomplex="wavelet")

    # Bandpass Filtering and Hilbert Transformation
    phases = p.filter(
        fs, xx[i_batch, i_ch], ftype="phase", n_jobs=1
    )  # (50, 20, 2048)
    amplitudes = p.filter(
        fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=1
    )  # (50, 20, 2048)

    # Calculates xpac
    k = 2
    p.idpac = (k, 0, 0)
    xpac = p.fit(phases, amplitudes, n_perm=n_perm)  # (50, 50, 20)
    pac = xpac.mean(axis=-1)  # (50, 50)

    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    pac = pac.T  # (amp, pha) -> (pha, amp)

    return phases, amplitudes, freqs_pha, freqs_amp, pac


def plot_PAC_mngs_vs_tensorpac(
    pac_mngs: NDArray,
    pac_mngs_trainable: NDArray,
    pac_tp: NDArray,
    freqs_pha: NDArray,
    freqs_amp: NDArray,
) -> plt.Figure:
    """Creates comparative plots of PAC values from mngs and tensorpac.

    Example
    -------
    >>> pac_mngs = np.random.rand(50, 50)
    >>> pac_tp = np.random.rand(50, 50)
    >>> freqs_pha = np.linspace(1, 50, 50)
    >>> freqs_amp = np.linspace(1, 100, 50)
    >>> fig = plot_PAC_mngs_vs_tensorpac(pac_mngs, pac_tp, freqs_pha, freqs_amp)

    Parameters
    ----------
    pac_mngs : NDArray
        PAC values from mngs package
    pac_tp : NDArray
        PAC values from tensorpac package
    freqs_pha : NDArray
        Phase frequencies
    freqs_amp : NDArray
        Amplitude frequencies

    Returns
    -------
    plt.Figure
        Figure containing three subplots comparing PAC values
    """
    assert pac_mngs.shape == pac_tp.shape

    fig, axes = mngs.plt.subplots(ncols=4, sharex=True, sharey=True)

    # vmin = min(
    #     np.min(pac_mngs),
    #     np.min(pac_mngs_trainable),
    #     np.min(pac_tp),
    #     np.min(pac_mngs - pac_tp),
    # )
    # vmax = max(
    #     np.max(pac_mngs),
    #     np.max(pac_mngs_trainable),
    #     np.max(pac_tp),
    #     np.max(pac_mngs - pac_tp),
    # )

    vmin, vmax = None, None
    cbar = True
    ax = axes[0]
    ax.imshow2d(
        pac_mngs,
        cbar=cbar,
        cbar_shrink=0.5,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title("mngs")

    ax = axes[1]
    ax.imshow2d(
        pac_mngs_trainable,
        cbar=cbar,
        cbar_shrink=0.5,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title("mngs (trainable)")

    ax = axes[2]
    ax.imshow2d(
        pac_tp,
        cbar=cbar,
        cbar_shrink=0.5,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title("tensorpac")

    ax = axes[3]
    ax.imshow2d(
        pac_mngs - pac_tp,
        cbar_label="PAC values",
        cbar_shrink=0.5,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title(f"Diff.")

    fig.suptitle("PAC (MI) values")
    fig.supxlabel("Frequency for phase [Hz]")
    fig.supylabel("Frequency for amplitude [Hz]")

    fig.tight_layout()

    return fig


def compute_pac_differences(
    pac_mngs: NDArray, pac_tp: NDArray
) -> dict[str, float]:
    """Computes RMS of absolute and relative differences between PAC values.

    Parameters
    ----------
    pac_mngs : NDArray
        PAC values from mngs package
    pac_tp : NDArray
        PAC values from tensorpac package

    Returns
    -------
    dict[str, float]
        Dictionary containing RMS metrics:
        - abs_diff_rms: RMS of absolute differences
        - rel_diff_percent: RMS of relative differences in percentage
    """
    abs_diff = np.abs(pac_mngs - pac_tp)
    rel_diff = abs_diff / (np.abs(pac_tp) + np.finfo(float).eps)

    return {
        "abs_diff_rms": float(f"{np.sqrt(np.mean(abs_diff**2)):.3f}"),
        "rel_diff_percent": float(
            f"{100 * np.sqrt(np.mean(rel_diff**2)):.1f}"
        ),  # in %
    }


def compute_pac_correlations(
    pac_mngs: NDArray, pac_tp: NDArray
) -> dict[str, float]:
    """Computes correlations between PAC matrices.

    Parameters
    ----------
    pac_mngs : NDArray
        PAC values from mngs package
    pac_tp : NDArray
        PAC values from tensorpac package

    Returns
    -------
    dict[str, float]
        Dictionary containing correlation metrics:
        - pearson_r: Pearson correlation coefficient
        - spearman_rho: Spearman rank correlation
        - kendall_tau: Kendall's tau correlation
    """
    # Flatten arrays
    mngs_flat = pac_mngs.flatten()
    tp_flat = pac_tp.flatten()

    return {
        "pearson_r": float(f"{np.corrcoef(mngs_flat, tp_flat)[0,1]:.3f}"),
        "spearman_rho": float(
            f"{scipy.stats.spearmanr(mngs_flat, tp_flat)[0]:.3f}"
        ),
        "kendall_tau": float(
            f"{scipy.stats.kendalltau(mngs_flat, tp_flat)[0]:.3f}"
        ),
    }


def compare_pac_mngs_and_pac_tensorpac() -> None:
    """Compares PAC calculations between mngs and tensorpac packages.

    Example
    -------
    >>> compare_pac_mngs_and_pac_tensorpac()
    # Saves comparison plot as pac_value_comparision.jpg
    """
    # Parameters
    params = CONFIG.PARAMS.BASELINE
    params["package"] = "mngs"
    params["fs"] = 512
    params["t_sec"] = 4
    params["n_perm"] = None # ,200
    i_batch, i_ch = 0, 0

    # Signal
    xx, tt, fs = mngs.dsp.demo_sig(
        "pac",
        batch_size=params["batch_size"],
        n_chs=params["n_chs"],
        n_segments=params["n_segments"],
        t_sec=params["t_sec"],
        fs=params["fs"],
    )

    # Calculations
    # Tensorpac
    phases, amplitudes, freqs_pha, freqs_amp, pac_tp = calc_pac_with_tensorpac(
        xx,
        params["fs"],
        params["t_sec"],
        i_batch=i_batch,
        i_ch=0,
        n_perm=params["n_perm"],
    )
    # ipdb> freqs_pha
    # array([ 2.        ,  2.36734694,  2.73469388,  3.10204082,  3.46938776,
    #         3.83673469,  4.20408163,  4.57142857,  4.93877551,  5.30612245,
    #         5.67346939,  6.04081633,  6.40816327,  6.7755102 ,  7.14285714,
    #         7.51020408,  7.87755102,  8.24489796,  8.6122449 ,  8.97959184,
    #         9.34693878,  9.71428571, 10.08163265, 10.44897959, 10.81632653,
    #         11.18367347, 11.55102041, 11.91836735, 12.28571429, 12.65306122,
    #         13.02040816, 13.3877551 , 13.75510204, 14.12244898, 14.48979592,
    #         14.85714286, 15.2244898 , 15.59183673, 15.95918367, 16.32653061,
    #         16.69387755, 17.06122449, 17.42857143, 17.79591837, 18.16326531,
    #         18.53061224, 18.89795918, 19.26530612, 19.63265306, 20.        ])
    # ipdb> freqs_amp
    # array([ 60.        ,  62.04081633,  64.08163265,  66.12244898,
    #         68.16326531,  70.20408163,  72.24489796,  74.28571429,
    #         76.32653061,  78.36734694,  80.40816327,  82.44897959,
    #         84.48979592,  86.53061224,  88.57142857,  90.6122449 ,
    #         92.65306122,  94.69387755,  96.73469388,  98.7755102 ,
    #         100.81632653, 102.85714286, 104.89795918, 106.93877551,
    #         108.97959184, 111.02040816, 113.06122449, 115.10204082,
    #         117.14285714, 119.18367347, 121.2244898 , 123.26530612,
    #         125.30612245, 127.34693878, 129.3877551 , 131.42857143,
    #         133.46938776, 135.51020408, 137.55102041, 139.59183673,
    #         141.63265306, 143.67346939, 145.71428571, 147.75510204,
    #         149.79591837, 151.83673469, 153.87755102, 155.91836735,
    #         157.95918367, 160.        ])
    # mngs
    pac_mngs, freqs_pha, freqs_amp = mngs.dsp.pac(
        xx,
        params["fs"],
        batch_size=params["batch_size"],
        pha_n_bands=pac_tp.shape[0],
        amp_n_bands=pac_tp.shape[1],
        n_perm=params["n_perm"],
    )
    pac_mngs = pac_mngs[i_batch, i_ch]
    # pac_mngs = pac_mngs[i_batch, i_epoch].detach().cpu().numpy()

    # mngs
    pac_mngs_trainable, _, _ = mngs.dsp.pac(
        xx,
        params["fs"],
        batch_size=params["batch_size"],
        pha_n_bands=pac_tp.shape[0],
        amp_n_bands=pac_tp.shape[1],
        n_perm=params["n_perm"],
        trainable=True,
    )
    pac_mngs_trainable = pac_mngs_trainable[i_batch, i_ch]
    # pac_mngs_trainable = pac_mngs_trainable[i_batch, i_epoch].detach().cpu().numpy()

    # Difference metrics
    differences = compute_pac_differences(pac_mngs, pac_tp)
    print("Differences between mngs and tensorpac PAC values:")
    for key, value in differences.items():
        print(f"{key}: {value:.3f}")

    # After difference metrics
    correlations = compute_pac_correlations(pac_mngs, pac_tp)
    print("\nCorrelations between mngs and tensorpac PAC values:")
    for key, value in correlations.items():
        print(f"{key}: {value:.3f}")

    # Plotting
    fig = plot_PAC_mngs_vs_tensorpac(
        pac_mngs, pac_mngs_trainable, pac_tp, freqs_pha, freqs_amp
    )
    fig.suptitle(
        f"PAC (MI) values\nRMS diff: {differences['abs_diff_rms']:.3f} (abs), "
        f"{differences['rel_diff_percent']:.1f}% (rel)"
    )
    mngs.io.save(fig, "pac_value_comparision.jpg")


main = compare_pac_mngs_and_pac_tensorpac

if __name__ == "__main__":
    import torch
    from scripts.utils.prepare_signal import prepare_signal

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, agg=True, verbose=False
    )

    main()

    mngs.gen.close(CONFIG, notify=False)

# EOF
