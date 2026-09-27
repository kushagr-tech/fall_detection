"""
features/extractor.py
=====================
Extracts a flat feature vector from each (window_size × 6) sensor window.

Feature groups
--------------
Time-domain  (per axis):
  mean, std, min, max, range, RMS, peak-to-peak, zero-crossing rate,
  mean absolute deviation, skewness, kurtosis, inter-quartile range,
  Signal Magnitude Area (SMA), correlation between axes

Frequency-domain (per axis via FFT):
  dominant frequency, energy at dominant frequency,
  spectral entropy, spectral centroid, power in 3 sub-bands

Fall-specific:
  SVM (Signal Vector Magnitude) peak, SVM standard-deviation,
  SVM jerk mean/peak, combined acc+gyro resultant magnitude statistics

These 130-ish features are fed to the Random Forest.  The Neural Network
operates on the raw (window × 6) tensor — no feature engineering needed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

# Column order matches SENSOR_COLS in data/loader.py
# [ax, ay, az, gx, gy, gz]
_N_SENSORS = 6
_ACC_IDX   = [0, 1, 2]   # accelerometer axes
_GYR_IDX   = [3, 4, 5]   # gyroscope axes


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    X : np.ndarray of shape (n_windows, window_size, 6)

    Returns
    -------
    np.ndarray of shape (n_windows, n_features)
    """
    return np.vstack([_window_features(X[i]) for i in range(len(X))])


def feature_names() -> list[str]:
    """Return ordered list of feature names matching extract_features output."""
    names: list[str] = []
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]

    # Time-domain per axis
    td_stats = [
        "mean", "std", "min", "max", "range", "rms", "peak2peak",
        "zcr", "mad", "skew", "kurt", "iqr",
    ]
    for ax in axes:
        for stat in td_stats:
            names.append(f"{ax}_{stat}")

    # SMA
    names += ["acc_sma", "gyr_sma"]

    # Cross-axis correlations
    for a, b in [("ax","ay"), ("ax","az"), ("ay","az"),
                  ("gx","gy"), ("gx","gz"), ("gy","gz")]:
        names.append(f"corr_{a}_{b}")

    # Frequency-domain per axis
    fd_stats = ["dom_freq", "dom_energy", "spec_entropy",
                "spec_centroid", "band_low", "band_mid", "band_high"]
    for ax in axes:
        for stat in fd_stats:
            names.append(f"{ax}_{stat}")

    # SVM / jerk features
    names += [
        "acc_svm_mean", "acc_svm_std", "acc_svm_peak",
        "gyr_svm_mean", "gyr_svm_std", "gyr_svm_peak",
        "acc_jerk_mean", "acc_jerk_peak",
        "gyr_jerk_mean", "gyr_jerk_peak",
        "resultant_mean", "resultant_peak",
    ]

    return names


# ── Per-window computation ─────────────────────────────────────────────────────

def _window_features(w: np.ndarray) -> np.ndarray:
    """
    w : (window_size, 6)  float32
    returns 1-D float64 feature vector
    """
    feats: list[float] = []

    # ── Time-domain per axis ───────────────────────────────────────────────────
    for col in range(_N_SENSORS):
        s = w[:, col].astype(np.float64)
        feats += _time_domain(s)

    # ── Signal Magnitude Area ──────────────────────────────────────────────────
    feats.append(np.mean(np.sum(np.abs(w[:, _ACC_IDX]), axis=1)))
    feats.append(np.mean(np.sum(np.abs(w[:, _GYR_IDX]), axis=1)))

    # ── Cross-axis correlations ────────────────────────────────────────────────
    for i, j in [(0,1), (0,2), (1,2), (3,4), (3,5), (4,5)]:
        feats.append(float(np.corrcoef(w[:, i], w[:, j])[0, 1]))

    # ── Frequency-domain per axis ──────────────────────────────────────────────
    fs = 50.0   # assumed sampling rate (Hz) — used only for frequency labelling
    for col in range(_N_SENSORS):
        s = w[:, col].astype(np.float64)
        feats += _freq_domain(s, fs)

    # ── SVM & jerk ────────────────────────────────────────────────────────────
    feats += _svm_jerk_features(w)

    return np.array(feats, dtype=np.float64)


# ── Time-domain statistics ─────────────────────────────────────────────────────

def _time_domain(s: np.ndarray) -> list[float]:
    mn   = float(np.mean(s))
    sd   = float(np.std(s))
    smin = float(np.min(s))
    smax = float(np.max(s))
    rng  = smax - smin
    rms  = float(np.sqrt(np.mean(s ** 2)))
    p2p  = rng

    # Zero-crossing rate
    zcr = float(np.mean(np.diff(np.sign(s - mn)) != 0))

    mad  = float(np.mean(np.abs(s - mn)))
    sk   = float(skew(s))
    ku   = float(kurtosis(s))
    iqr  = float(np.percentile(s, 75) - np.percentile(s, 25))

    return [mn, sd, smin, smax, rng, rms, p2p, zcr, mad, sk, ku, iqr]


# ── Frequency-domain statistics ────────────────────────────────────────────────

def _freq_domain(s: np.ndarray, fs: float) -> list[float]:
    n = len(s)
    fft_vals = np.abs(np.fft.rfft(s - s.mean())) ** 2   # power spectrum
    freqs    = np.fft.rfftfreq(n, d=1.0 / fs)

    # Guard against all-zero window
    total_power = fft_vals.sum() + 1e-12
    p_norm = fft_vals / total_power

    dom_idx    = int(np.argmax(fft_vals[1:]) + 1)   # skip DC
    dom_freq   = float(freqs[dom_idx])
    dom_energy = float(fft_vals[dom_idx] / total_power)

    # Spectral entropy (normalised)
    spec_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(p_norm)))

    # Spectral centroid
    spec_centroid = float(np.sum(freqs * p_norm))

    # Sub-band power  (0-2 Hz, 2-5 Hz, 5-25 Hz)
    def _band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(fft_vals[mask].sum() / total_power)

    band_low  = _band(0.0, 2.0)
    band_mid  = _band(2.0, 5.0)
    band_high = _band(5.0, 25.0)

    return [dom_freq, dom_energy, spec_entropy, spec_centroid,
            band_low, band_mid, band_high]


# ── SVM / jerk features ────────────────────────────────────────────────────────

def _svm_jerk_features(w: np.ndarray) -> list[float]:
    """
    Signal Vector Magnitude and jerk (first derivative of SVM).
    These are particularly informative for detecting falls.
    """
    acc = w[:, _ACC_IDX].astype(np.float64)
    gyr = w[:, _GYR_IDX].astype(np.float64)

    acc_svm = np.sqrt(np.sum(acc ** 2, axis=1))
    gyr_svm = np.sqrt(np.sum(gyr ** 2, axis=1))

    acc_jerk = np.abs(np.diff(acc_svm))
    gyr_jerk = np.abs(np.diff(gyr_svm))

    # Combined resultant of all 6 channels
    resultant = np.sqrt(np.sum(w.astype(np.float64) ** 2, axis=1))

    return [
        float(acc_svm.mean()), float(acc_svm.std()),  float(acc_svm.max()),
        float(gyr_svm.mean()), float(gyr_svm.std()),  float(gyr_svm.max()),
        float(acc_jerk.mean()), float(acc_jerk.max()),
        float(gyr_jerk.mean()), float(gyr_jerk.max()),
        float(resultant.mean()), float(resultant.max()),
    ]
