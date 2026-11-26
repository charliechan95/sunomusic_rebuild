#!/usr/bin/env python3
"""
SUB-PROBLEM 3: MIDI → Audio Render (Piano SoundFont)
====================================================
Renders a MIDI file to audio using fluidsynth with a piano SoundFont.
Program 0 = acoustic grand piano.

Input:  MIDI file + SoundFont (.sf2)
Output: WAV audio file
"""

import subprocess
import os
import sys


def render_midi_to_audio(midi_path, soundfont_path, output_path, sample_rate=44100, gain=1.0):
    """
    Render MIDI to WAV using fluidsynth.
    
    Args:
        midi_path: Path to input MIDI file
        soundfont_path: Path to .sf2 SoundFont (piano)
        output_path: Path for output WAV
        sample_rate: Output sample rate (Hz)
        gain: Fluidsynth gain multiplier
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'fluidsynth', '-F', output_path,
        '-r', str(sample_rate),
        '-g', str(gain),
        '-s',  # no shell
        soundfont_path,
        midi_path
    ]
    print(f"  Running: fluidsynth -r {sample_rate} -g {gain} {soundfont_path} {midi_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Fluidsynth error (rc={result.returncode}):")
        print(f"  {result.stderr[-500:] if result.stderr else 'no stderr'}")
        return False
    if not os.path.exists(output_path):
        print(f"  Output file not created: {output_path}")
        return False
    print(f"  Rendered: {output_path}")
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render MIDI to audio with fluidsynth')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--soundfont', required=True, help='SoundFont .sf2 path')
    parser.add_argument('--output-wav', required=True, help='Output WAV path')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate')
    parser.add_argument('--gain', type=float, default=1.0, help='Fluidsynth gain')
    args = parser.parse_args()

    success = render_midi_to_audio(
        args.input_midi, args.soundfont, args.output_wav,
        args.sample_rate, args.gain
    )
    if success:
        print("\nRender complete.")
    else:
        print("\nRender FAILED.")
        sys.exit(1)
