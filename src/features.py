"""Physics-grounded spectral features. Per the student manual 4.5."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from src.config import SAMPLE_RATE_HZ, FREQ_BANDS_HZ, N_TOP_PEAKS


def detrend_and_window(x: np.ndarray) -> np.ndarray:
    """Remove DC offset (gravity bias on Z) and apply Hann window."""
    x = x - x.mean()
    return x * np.hanning(len(x))


def fft_magnitude(x: np.ndarray, fs: int = SAMPLE_RATE_HZ):
    """Returns (freqs, mag) for one axis."""
    xw = detrend_and_window(x)
    spectrum = np.fft.rfft(xw)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return freqs, mag


def band_energies(freqs: np.ndarray, mag: np.ndarray, bands=FREQ_BANDS_HZ) -> dict:
    """Integrate spectrum over each band."""
    out = {}
    total = mag.sum() + 1e-12
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        e = mag[mask].sum()
        out[f"band_{lo}_{hi}_e"] = float(e)
        out[f"band_{lo}_{hi}_frac"] = float(e / total)
    return out


def spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Center of mass of the spectrum."""
    s = mag.sum()
    if s < 1e-12:
        return 0.0
    return float((freqs * mag).sum() / s)


def top_peaks(freqs: np.ndarray, mag: np.ndarray, n: int = N_TOP_PEAKS) -> dict:
    """Top-N peak frequencies and amplitudes."""
    peaks_idx, props = find_peaks(mag, height=0)
    if len(peaks_idx) == 0:
        out = {}
        for i in range(n):
            out[f"peak{i+1}_freq"] = 0.0
            out[f"peak{i+1}_amp"] = 0.0
        return out
    heights = props["peak_heights"]
    order = np.argsort(heights)[::-1][:n]
    selected = peaks_idx[order]
    out = {}
    for i in range(n):
        if i < len(selected):
            out[f"peak{i+1}_freq"] = float(freqs[selected[i]])
            out[f"peak{i+1}_amp"] = float(mag[selected[i]])
        else:
            out[f"peak{i+1}_freq"] = 0.0
            out[f"peak{i+1}_amp"] = 0.0
    return out


def half_power_bandwidth(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Width of the dominant peak at -3 dB. Indicator of damping."""
    if mag.max() < 1e-12:
        return 0.0
    peak_idx = int(np.argmax(mag))
    half_power = mag[peak_idx] / np.sqrt(2)
    left = peak_idx
    while left > 0 and mag[left] > half_power:
        left -= 1
    right = peak_idx
    while right < len(mag) - 1 and mag[right] > half_power:
        right += 1
    return float(freqs[right] - freqs[left])


def time_domain_stats(x: np.ndarray) -> dict:
    """RMS, kurtosis, crest factor, peak-to-peak."""
    x_centered = x - x.mean()
    rms = float(np.sqrt(np.mean(x_centered**2)))
    peak = float(np.max(np.abs(x_centered)))
    crest = peak / rms if rms > 1e-12 else 0.0
    if rms > 1e-12:
        kurt = float(np.mean((x_centered / rms) ** 4) - 3.0)
    else:
        kurt = 0.0
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "kurtosis": kurt,
        "ptp": float(np.ptp(x)),
    }


def extract_features_one_axis(x: np.ndarray, axis_name: str) -> dict:
    """All features for one axis. Returns flat dict with axis_name prefix."""
    freqs, mag = fft_magnitude(x)
    feats = {}
    feats.update({f"{axis_name}_{k}": v for k, v in band_energies(freqs, mag).items()})
    feats[f"{axis_name}_centroid"] = spectral_centroid(freqs, mag)
    feats[f"{axis_name}_hpbw"] = half_power_bandwidth(freqs, mag)
    feats.update({f"{axis_name}_{k}": v for k, v in top_peaks(freqs, mag).items()})
    feats.update({f"{axis_name}_{k}": v for k, v in time_domain_stats(x).items()})
    return feats


def extract_features(capture: np.ndarray) -> dict:
    """capture: (N, 3) array of [X, Y, Z]. Returns ~90 features as flat dict."""
    feats = {}
    for i, axis in enumerate(["x", "y", "z"]):
        feats.update(extract_features_one_axis(capture[:, i], axis))
    mag_xyz = np.sqrt((capture**2).sum(axis=1))
    feats.update({f"mag_{k}": v for k, v in time_domain_stats(mag_xyz).items()})
    return feats


def extract_features_batch(paths_df: pd.DataFrame, load_fn) -> pd.DataFrame:
    """Returns features DF with original cols + extracted feature columns."""
    from tqdm import tqdm
    feature_rows = []
    for row in tqdm(paths_df.itertuples(index=False), total=len(paths_df), desc="Features"):
        try:
            cap = load_fn(row.path)
            feats = extract_features(cap)
            feats.update(row._asdict())
            feature_rows.append(feats)
        except Exception as e:
            print(f"FAIL {row.path}: {e}")
    return pd.DataFrame(feature_rows)
