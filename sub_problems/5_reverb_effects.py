#!/usr/bin/env python3
"""
SUB-PROBLEM 5: Reverb Effects
=============================
Adds convolution reverb to match the original recording's acoustic space.
Uses an exponentially decaying noise impulse response (Schroeder-style).

Input:  matched WAV
Output: WAV with reverb applied
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as scipy_signal
import sys
import os

SR = 44100


def design_reverb_ir(sr=SR, decay=1.5, predelay_ms=40, reverb_len_sec=3.0):
    """Design a convolution reverb impulse response."""
    predelay_samples = int(predelay_ms / 1000 * sr)
    reverb_len = int(reverb_len_sec * sr)
    t = np.arange(reverb_len) / sr

    rng = np.random.RandomState(42)
    noise = rng.randn(reverb_len)
    envelope = np.exp(-t * 4.0 / decay)  # RT60 approximation
    impulse = noise * envelope
    impulse[:predelay_samples] = 0
    impulse[0] = 1.0  # direct sound

    # Normalize
    impulse = impulse / np.sqrt(np.sum(impulse ** 2)) * 0.5
    return impulse


def add_reverb(y, sr=SR, wet=0.25, decay=1.5, predelay_ms=40):
    """
    Add convolution reverb to audio.
    
    Args:
        y: audio array (channels x samples) or (samples,)
        sr: sample rate
        wet: wet/dry mix ratio (0-1)
        decay: reverb decay time in seconds
        predelay_ms: pre-delay in milliseconds
    
    Returns:
        y_reverb: audio with reverb applied
    """
    if y.ndim == 1:
        y = np.stack([y, y])
    
    ir = design_reverb_ir(sr, decay=decay, predelay_ms=predelay_ms)
    y_reverb = np.zeros_like(y)
    for ch in range(y.shape[0]):
        conv = scipy_signal.fftconvolve(y[ch], ir, mode='same')[:len(y[ch])]
        y_reverb[ch] = (1.0 - wet) * y[ch] + wet * conv
    return y_reverb


def apply_reverb_to_file(input_path, output_path, sr=SR, wet=0.25, decay=1.5, predelay_ms=40):
    """Apply reverb to a WAV file and save."""
    y, _ = librosa.load(input_path, sr=sr, mono=False)
    orig_peak = float(np.max(np.abs(y)))
    
    y_reverb = add_reverb(y, sr, wet=wet, decay=decay, predelay_ms=predelay_ms)
    
    # Gain normalize to original peak
    cp = float(np.max(np.abs(y_reverb)))
    if cp > 0:
        y_reverb = y_reverb * (orig_peak / cp)
    
    sf.write(output_path, y_reverb.T, sr, subtype='PCM_16')
    print(f"  Reverb applied (wet={wet}, decay={decay}s): {output_path}")
    return y_reverb


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add convolution reverb to WAV')
    parser.add_argument('--input-wav', required=True, help='Input WAV path')
    parser.add_argument('--output-wav', required=True, help='Output WAV path')
    parser.add_argument('--wet', type=float, default=0.25, help='Wet/dry mix (0-1)')
    parser.add_argument('--decay', type=float, default=1.5, help='Reverb decay (seconds)')
    parser.add_argument('--predelay-ms', type=float, default=40, help='Pre-delay (ms)')
    args = parser.parse_args()

    apply_reverb_to_file(
        args.input_wav, args.output_wav,
        wet=args.wet, decay=args.decay, predelay_ms=args.predelay_ms
    )
    print("Reverb complete.")
