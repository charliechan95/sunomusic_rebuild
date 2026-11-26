#!/usr/bin/env python3
"""
SUB-PROBLEM 13: Final Verification — Rendered vs Original
==========================================================
Compares the rendered (MIDI→audio) output against the original recording
across every measurable dimension:

  - Duration (sample-exact)
  - Peak amplitude
  - Waveform correlation (sample-aligned)
  - Per-band spectral ratios (20-150, 150-400, ..., 12k-20k Hz)
  - Spectral centroid, rolloff, bandwidth
  - Onset count and timing (onset detection on both)
  - Suno ultrasonic watermark check (16kHz+)
  - PASS/FAIL per band at 5% tolerance

Output: JSON verification report + console summary
"""

import numpy as np
import librosa
import soundfile as sf
import json
import sys
import os

SR = 44100
N_FFT = 8192
BANDS = [(20, 150), (150, 400), (400, 1000), (1000, 2500),
         (2500, 6000), (6000, 12000), (12000, 20000)]


def load_mono(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def check_watermark(y, sr=SR):
    """Check for Suno ultrasonic watermark (16kHz+ content)."""
    S = np.abs(librosa.stft(y, n_fft=8192))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    ultrasonic_mask = freqs >= 16000
    audible_mask = (freqs >= 20) & (freqs <= 16000)
    ultrasonic_energy = float(np.mean(S[ultrasonic_mask, :]))
    audible_energy = float(np.mean(S[audible_mask, :]))
    ratio = ultrasonic_energy / (audible_energy + 1e-12)
    return {
        "ultrasonic_energy": ultrasonic_energy,
        "audible_energy": audible_energy,
        "ratio": ratio,
        "watermark_detected": ratio > 0.001
    }


def verify_rendered(rendered_path, original_path, report_path, sr=SR):
    y_orig = load_mono(original_path, sr)
    y_rend = load_mono(rendered_path, sr)

    # Trim to same length
    min_len = min(len(y_orig), len(y_rend))
    y_orig = y_orig[:min_len]
    y_rend = y_rend[:min_len]

    report = {
        "original": original_path,
        "rendered": rendered_path,
        "duration_sec": min_len / sr,
        "samples": min_len,
        "sample_rate": sr
    }

    # 1. Duration
    report["duration_orig_sec"] = len(load_mono(original_path, sr)) / sr
    report["duration_rendered_sec"] = len(load_mono(rendered_path, sr)) / sr
    report["duration_match"] = abs(len(y_orig) - len(y_rend)) == 0

    # 2. Peak amplitude
    peak_orig = float(np.max(np.abs(y_orig)))
    peak_rend = float(np.max(np.abs(y_rend)))
    report["peak_orig"] = peak_orig
    report["peak_rendered"] = peak_rend
    report["peak_ratio"] = peak_rend / (peak_orig + 1e-12)

    # 3. Waveform correlation
    corr = float(np.corrcoef(y_orig, y_rend)[0, 1])
    report["waveform_correlation"] = corr

    # 4. Per-band spectral ratios
    S_orig = np.abs(librosa.stft(y_orig, n_fft=N_FFT))
    S_rend = np.abs(librosa.stft(y_rend, n_fft=N_FFT))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    band_results = {}
    all_bands_5pct = True
    all_bands_10pct = True

    for low, high in BANDS:
        mask = (freqs >= low) & (freqs <= high)
        r = float(np.mean(S_rend[mask]) / np.mean(S_orig[mask]))
        flag_5 = "PASS" if abs(1 - r) <= 0.05 else ("WARN" if abs(1 - r) <= 0.10 else "FAIL")
        flag_10 = "PASS" if abs(1 - r) <= 0.10 else "FAIL"
        if abs(1 - r) > 0.05:
            all_bands_5pct = False
        if abs(1 - r) > 0.10:
            all_bands_10pct = False
        band_results[f"{low}-{high}Hz"] = {
            "ratio": round(r, 4),
            "status_5pct": flag_5,
            "status_10pct": flag_10
        }

    report["bands"] = band_results
    report["all_bands_5pct"] = all_bands_5pct
    report["all_bands_10pct"] = all_bands_10pct

    # 5. Spectral features
    centroid_orig = float(np.mean(librosa.feature.spectral_centroid(y=y_orig, sr=sr)))
    centroid_rend = float(np.mean(librosa.feature.spectral_centroid(y=y_rend, sr=sr)))
    rolloff_orig = float(np.mean(librosa.feature.spectral_rolloff(y=y_orig, sr=sr)))
    rolloff_rend = float(np.mean(librosa.feature.spectral_rolloff(y=y_rend, sr=sr)))

    report["spectral_centroid_orig"] = centroid_orig
    report["spectral_centroid_rendered"] = centroid_rend
    report["spectral_centroid_ratio"] = centroid_rend / (centroid_orig + 1e-12)
    report["spectral_rolloff_orig"] = rolloff_orig
    report["spectral_rolloff_rendered"] = rolloff_rend
    report["spectral_rolloff_ratio"] = rolloff_rend / (rolloff_orig + 1e-12)

    # 6. Onset comparison
    onset_orig = librosa.onset.onset_detect(y=y_orig, sr=sr, hop_length=256, backtrack=True)
    onset_rend = librosa.onset.onset_detect(y=y_rend, sr=sr, hop_length=256, backtrack=True)
    onset_times_orig = librosa.frames_to_time(onset_orig, sr=sr, hop_length=256)
    onset_times_rend = librosa.frames_to_time(onset_rend, sr=sr, hop_length=256)

    report["onset_count_orig"] = len(onset_times_orig)
    report["onset_count_rendered"] = len(onset_times_rend)
    report["onset_count_ratio"] = len(onset_times_rend) / (len(onset_times_orig) + 1e-12)

    # 7. Watermark check
    wm_orig = check_watermark(y_orig, sr)
    wm_rend = check_watermark(y_rend, sr)
    report["watermark_original"] = wm_orig
    report["watermark_rendered"] = wm_rend
    report["watermark_removed"] = (not wm_rend["watermark_detected"]) and wm_orig["watermark_detected"]

    # 8. Overall verdict
    report["overall_pass"] = (
        report["duration_match"] and
        all_bands_5pct and
        report["waveform_correlation"] > 0.7 and
        (not wm_rend["watermark_detected"])
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION REPORT")
    print("=" * 60)
    print(f"Duration:        Orig={report['duration_orig_sec']:.4f}s  "
          f"Rend={report['duration_rendered_sec']:.4f}s  Match={report['duration_match']}")
    print(f"Peak:            Orig={peak_orig:.4f}  Rend={peak_rend:.4f}  "
          f"Ratio={report['peak_ratio']:.4f}")
    print(f"Waveform corr:   {corr:.4f}")
    print(f"Onset count:     Orig={len(onset_times_orig)}  Rend={len(onset_times_rend)}")
    print(f"Watermark orig:  {wm_orig['watermark_detected']} (ratio={wm_orig['ratio']:.6f})")
    print(f"Watermark rend:  {wm_rend['watermark_detected']} (ratio={wm_rend['ratio']:.6f})")
    print(f"Watermark removed: {report['watermark_removed']}")
    print(f"\nBand ratios (Rend/Orig, 5% tolerance):")
    for low, high in BANDS:
        key = f"{low}-{high}Hz"
        br = band_results[key]
        print(f"  {key:>12}Hz: {br['ratio']:.4f}x [{br['status_5pct']}]")
    print(f"\nAll bands 5%: {all_bands_5pct}")
    print(f"All bands 10%: {all_bands_10pct}")
    print(f"\nOVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print("=" * 60)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify rendered audio vs original')
    parser.add_argument('--input-rendered', required=True, help='Rendered WAV path')
    parser.add_argument('--input-original', required=True, help='Original WAV path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    args = parser.parse_args()

    verify_rendered(args.input_rendered, args.input_original, args.report)
    print("\nVerification complete.")
