#!/usr/bin/env python3
"""
SUB-PROBLEM 4: Spectral Matching to Original
============================================
Applies a minimum-phase FIR filter to match the rendered audio's spectrum
to the original recording's spectrum. This is the core "correct the sound"
step.

Input:  rendered WAV + original WAV
Output: spectrally matched WAV + verification report
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as scipy_signal
from scipy.signal import firwin2, minimum_phase
from scipy.ndimage import gaussian_filter1d
import json
import sys
import os

SR = 44100
N_FFT = 8192
FIR_TAPS = 32769


def compute_spectrum(y, sr, n_fft=N_FFT, hop_length=512):
    """Compute average power spectrum."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    P = np.mean(S ** 2, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, P


def spectral_match(rendered_path, original_path, output_path,
                   sr=SR, n_fft=N_FFT, fir_taps=FIR_TAPS):
    """
    Match rendered audio spectrum to original using minimum-phase FIR.
    
    Returns:
        (y_matched, report_dict)
    """
    y_orig, _ = librosa.load(original_path, sr=sr, mono=True)
    y, _ = librosa.load(rendered_path, sr=sr, mono=False)
    y_mono = librosa.to_mono(y)
    orig_peak = float(np.max(np.abs(y_orig)))

    # Compute average power spectra
    freqs, P_orig = compute_spectrum(y_orig, sr, n_fft)
    _, P_rend = compute_spectrum(y_mono, sr, n_fft)

    # Amplitude ratio
    eps = 1e-12
    amp_ratio = np.sqrt(P_orig / (P_rend + eps))
    amp_ratio = gaussian_filter1d(amp_ratio, sigma=0.5)
    # Clamp extreme ratios to avoid artifacts
    amp_ratio = np.clip(amp_ratio, 0.1, 10.0)

    print(f"  Amp ratio dB range: {20*np.log10(amp_ratio).min():.1f} to {20*np.log10(amp_ratio).max():.1f} dB")

    # Design FIR filter
    nyquist = sr / 2
    freq_points = np.concatenate(([0], freqs[1:-1], [nyquist]))
    gain_points = np.concatenate(([amp_ratio[0]], amp_ratio[1:-1], [amp_ratio[-1]]))

    print(f"  Designing FIR with {fir_taps} taps...")
    fir_linear = firwin2(fir_taps, freq_points, gain_points, fs=sr)

    print("  Converting to minimum phase...")
    fir_min = minimum_phase(fir_linear, method='homomorphic', n_fft=65536)

    # Apply to both channels
    print("  Applying FIR filter...")
    y_matched = np.zeros_like(y)
    for ch in range(y.shape[0]):
        y_matched[ch] = scipy_signal.fftconvolve(y[ch], fir_min, mode='same')

    # Peak normalize to original
    cp = float(np.max(np.abs(y_matched)))
    if cp > 0:
        y_matched = y_matched * (orig_peak / cp)

    # Trim/pad to exact original length
    matched_len = y_matched.shape[1] if y_matched.ndim > 1 else len(y_matched)
    orig_len = len(y_orig)
    if matched_len > orig_len:
        if y_matched.ndim > 1:
            y_matched = y_matched[:, :orig_len]
        else:
            y_matched = y_matched[:orig_len]
    elif matched_len < orig_len:
        pad = orig_len - matched_len
        if y_matched.ndim > 1:
            y_matched = np.pad(y_matched, ((0, 0), (0, pad)), mode='constant')
        else:
            y_matched = np.pad(y_matched, (0, pad), mode='constant')

    sf.write(output_path, y_matched.T, sr, subtype='PCM_16')
    print(f"  Saved: {output_path}")

    # Verify bands
    y_check, _ = librosa.load(output_path, sr=sr, mono=True)
    S_check = np.abs(librosa.stft(y_check, n_fft=n_fft))
    bands = [(20, 150), (150, 400), (400, 1000), (1000, 2500),
             (2500, 6000), (6000, 12000), (12000, 20000)]
    
    band_results = {}
    all_pass_5 = True
    print("  Band ratios (Matched/Orig):")
    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        r = float(np.mean(S_check[mask]) / np.mean(P_orig**0.5 if False else np.abs(librosa.stft(y_orig, n_fft=n_fft))[mask]))
        # Recompute properly
        S_orig_chk = np.abs(librosa.stft(y_orig, n_fft=n_fft))
        r = float(np.mean(S_check[mask]) / np.mean(S_orig_chk[mask]))
        flag = "PASS" if abs(1 - r) <= 0.05 else ("WARN" if abs(1 - r) <= 0.10 else "FAIL")
        if abs(1 - r) > 0.05:
            all_pass_5 = False
        print(f"    {low}-{high}Hz: {r:.3f}x [{flag}]")
        band_results[f"{low}-{high}Hz"] = {"ratio": r, "status": flag}

    report = {
        "all_bands_5pct": all_pass_5,
        "bands": band_results,
        "amp_ratio_db_min": float(20*np.log10(amp_ratio).min()),
        "amp_ratio_db_max": float(20*np.log10(amp_ratio).max()),
    }

    return y_matched, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Spectral match rendered audio to original')
    parser.add_argument('--input-rendered', required=True, help='Rendered WAV path')
    parser.add_argument('--input-original', required=True, help='Original WAV path')
    parser.add_argument('--output-wav', required=True, help='Output matched WAV path')
    parser.add_argument('--fir-taps', type=int, default=FIR_TAPS, help='FIR filter taps')
    args = parser.parse_args()

    y_matched, report = spectral_match(
        args.input_rendered, args.input_original, args.output_wav,
        fir_taps=args.fir_taps
    )
    print(f"\nSpectral match complete. All bands within 5%: {report['all_bands_5pct']}")
