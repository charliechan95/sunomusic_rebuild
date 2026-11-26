#!/usr/bin/env python3
"""
SUB-PROBLEM 11: Rhythm / Inter-Onset Gap Analysis (Bit by Bit)
================================================================
Analyzes the rhythm/gaps between notes in the original audio vs the
MIDI transcription. Reports every gap discrepancy so they can be
corrected bit by bit.

Approach:
  1. Load original audio, detect onsets -> inter-onset intervals (gaps)
  2. Load MIDI, collect note_on times -> inter-note gaps
  3. Align audio onsets to MIDI notes by nearest time match
  4. For each pair, compute gap_audio vs gap_midi, delta = diff
  5. Report all gaps with deltas > threshold

Output: JSON gap analysis report
"""

import numpy as np
import librosa
from mido import MidiFile
import json
import sys
import os

SR = 44100
TICKS_PER_BEAT = 960


def tick_to_sec(tick, ticks_per_beat=TICKS_PER_BEAT, bpm=120.0):
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat
    return tick * seconds_per_tick


def detect_audio_onsets(audio_path, sr=SR):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=256, backtrack=True,
        pre_max=30, post_max=30, pre_avg=60, post_avg=60,
        delta=0.25, wait=40
    )
    return np.sort(librosa.frames_to_time(onset_frames, sr=sr, hop_length=256))


def get_midi_note_times(midi_path, bpm=120.0, ticks_per_beat=TICKS_PER_BEAT):
    """Get sorted list of note_on times in seconds from MIDI."""
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat

    note_times = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_times.append(abs_tick * seconds_per_tick)
    return sorted(note_times)


def analyze_gaps(audio_path, midi_path, report_path, bpm=120.0,
                 gap_threshold_sec=0.01):
    """
    Compare inter-onset gaps in audio vs MIDI.

    For each consecutive pair of audio onsets:
      - gap_audio = onset[i+1] - onset[i]
    Find nearest MIDI note after onset[i], compute gap_midi similarly.
    Report delta.
    """
    onset_times = detect_audio_onsets(audio_path, SR)
    midi_times = get_midi_note_times(midi_path, bpm)

    print(f"  Audio onsets: {len(onset_times)}")
    print(f"  MIDI note-ons: {len(midi_times)}")

    # Compute audio gaps
    audio_gaps = []
    for i in range(len(onset_times) - 1):
        gap = onset_times[i+1] - onset_times[i]
        audio_gaps.append({
            "start_onset": onset_times[i],
            "end_onset": onset_times[i+1],
            "gap_sec": round(gap, 6)
        })

    # Compute MIDI gaps
    midi_gaps = []
    for i in range(len(midi_times) - 1):
        gap = midi_times[i+1] - midi_times[i]
        midi_gaps.append({
            "start_note": midi_times[i],
            "end_note": midi_times[i+1],
            "gap_sec": round(gap, 6)
        })

    # Align: for each audio gap, find nearest MIDI gap by start time
    comparisons = []
    large_delta_count = 0

    for ag in audio_gaps:
        # Find nearest MIDI gap by start time
        best_midi = None
        best_diff = float('inf')
        for mg in midi_gaps:
            d = abs(mg['start_note'] - ag['start_onset'])
            if d < best_diff:
                best_diff = d
                best_midi = mg

        if best_midi is None:
            continue

        delta = ag['gap_sec'] - best_midi['gap_sec']
        abs_delta = abs(delta)
        if abs_delta > gap_threshold_sec:
            large_delta_count += 1

        comparisons.append({
            "audio_start": round(ag['start_onset'], 6),
            "audio_end": round(ag['end_onset'], 6),
            "audio_gap_sec": ag['gap_sec'],
            "midi_start": round(best_midi['start_note'], 6),
            "midi_end": round(best_midi['end_note'], 6),
            "midi_gap_sec": best_midi['gap_sec'],
            "delta_sec": round(delta, 6),
            "abs_delta_sec": round(abs_delta, 6),
            "significant": abs_delta > gap_threshold_sec
        })

    comparisons.sort(key=lambda x: x['abs_delta_sec'], reverse=True)

    print(f"  Gaps with delta > {gap_threshold_sec}s: {large_delta_count}")

    report = {
        "audio_path": audio_path,
        "midi_path": midi_path,
        "bpm": bpm,
        "audio_onset_count": len(onset_times),
        "midi_note_count": len(midi_times),
        "audio_gap_count": len(audio_gaps),
        "midi_gap_count": len(midi_gaps),
        "gap_threshold_sec": gap_threshold_sec,
        "large_delta_count": large_delta_count,
        "comparisons": comparisons[:200]  # top 200 by delta
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved gap analysis report: {report_path}")

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze rhythm gaps audio vs MIDI')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    parser.add_argument('--bpm', type=float, default=120.0, help='Tempo in BPM')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Gap delta threshold in seconds')
    args = parser.parse_args()

    analyze_gaps(args.input_audio, args.input_midi, args.report,
                 args.bpm, args.threshold)
    print("\nRhythm gap analysis complete.")
