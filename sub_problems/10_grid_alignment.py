#!/usr/bin/env python3
"""
SUB-PROBLEM 10: Timing Grid Alignment (Bit by Bit)
==================================================
Quantizes all note events to a precise timing grid derived from the
original audio's tempo and detected onsets. Ensures every note sits on
the exact grid position.

Surgical approach:
  - Detect tempo from audio onsets (average inter-onset interval).
  - Build a grid of tick positions.
  - For each note_on, snap to nearest grid tick.
  - Apply same delta to matching note_off (preserve duration).
  - Rebuild with precise delta timing.
"""

import numpy as np
import librosa
from mido import MidiFile, MidiTrack, Message, MetaMessage
import json
import sys
import os

SR = 44100
TICKS_PER_BEAT = 960


def tick_to_sec(tick, ticks_per_beat=TICKS_PER_BEAT, bpm=120.0):
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat
    return tick * seconds_per_tick


def sec_to_tick(sec, ticks_per_beat=TICKS_PER_BEAT, bpm=120.0):
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat
    return int(round(sec / seconds_per_tick))


def detect_tempo_from_onsets(audio_path, sr=SR):
    """Estimate tempo (BPM) from audio onset intervals."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=256, backtrack=True,
        pre_max=30, post_max=30, pre_avg=60, post_avg=60,
        delta=0.25, wait=40
    )
    if len(onset_frames) < 2:
        return 120.0
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)
    intervals = np.diff(onset_times)
    # Filter out very short intervals (likely not beats)
    intervals = intervals[intervals > 0.1]
    if len(intervals) == 0:
        return 120.0
    median_interval = float(np.median(intervals))
    # Assume quarter note = median interval (or half, or double)
    bpm = 60.0 / median_interval
    # Normalize to reasonable range (60-200 BPM)
    while bpm > 200:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    return round(bpm, 2)


def align_to_grid(midi_path, audio_path, output_path, report_path,
                  grid_divisions=4):
    """
    Align all note events to a timing grid.

    grid_divisions: number of grid subdivisions per beat (4 = 16th notes)
    
    For each note_on:
      - Snap to nearest grid tick
      - Apply same delta to matching note_off
    """
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    bpm = detect_tempo_from_orig_audio(audio_path) if False else detect_tempo_from_onsets(audio_path, SR)
    print(f"  Detected tempo: {bpm} BPM")

    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat
    grid_tick_spacing = ticks_per_beat / grid_divisions  # ticks per grid division

    # Collect note_ons and note_offs
    note_ons = {}  # (track, channel, note, on_tick) -> info
    note_offs = {}  # (track, channel, note) -> off_tick
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note, abs_tick)
                note_ons[key] = {
                    'track_idx': track_idx, 'channel': msg.channel,
                    'note': msg.note, 'velocity': msg.velocity,
                    'on_tick': abs_tick, 'msg_idx': msg_idx,
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                if key not in note_offs:
                    note_offs[key] = abs_tick

    print(f"  Found {len(note_ons)} note_on events")
    print(f"  Grid spacing: {grid_tick_spacing:.2f} ticks per division")

    # Compute grid snapping for each note_on
    corrections = []
    snapped_count = 0
    on_to_delta = {}

    for key, info in note_ons.items():
        on_tick = info['on_tick']
        # Nearest grid tick
        nearest_grid_tick = int(round(on_tick / grid_tick_spacing) * grid_tick_spacing)
        delta_tick = nearest_grid_tick - on_tick
        snapped = delta_tick != 0
        if snapped:
            snapped_count += 1

        on_to_delta[key] = delta_tick
        corrections.append({
            "track": info['track_idx'], "channel": info['channel'],
            "note": info['note'], "velocity": info['velocity'],
            "old_tick": on_tick, "new_tick": nearest_grid_tick,
            "delta_tick": delta_tick,
            "old_sec": round(tick_to_sec(on_tick, ticks_per_beat, bpm), 6),
            "new_sec": round(tick_to_sec(nearest_grid_tick, ticks_per_beat, bpm), 6),
            "snapped": snapped
        })

    print(f"  Grid-aligned {snapped_count}/{len(note_ons)} notes")

    # Build off_to_delta: match each note_off to its note_on
    note_on_by_note = {}
    for key, delta in on_to_delta.items():
        tid, ch, note, on_tick = key
        note_on_by_note.setdefault((tid, ch, note), []).append((on_tick, delta))

    off_to_delta = {}
    for key, off_tick in note_offs.items():
        tid, ch, note = key
        candidates = note_on_by_note.get((tid, ch, note), [])
        best_delta = 0
        best_on_tick = -1
        for on_tick, delta in candidates:
            if on_tick <= off_tick and on_tick > best_on_tick:
                best_on_tick = on_tick
                best_delta = delta
        off_to_delta[(tid, ch, note, off_tick)] = best_delta

    # Rebuild MIDI tracks
    new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0
        prev_new_tick = 0

        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note, abs_tick)
                if key in on_to_delta:
                    delta = on_to_delta[key]
                    new_tick = abs_tick + delta
                    new_delta = new_tick - prev_new_tick
                    new_track.append(Message(
                        'note_on', channel=msg.channel,
                        note=msg.note, velocity=msg.velocity,
                        time=max(0, new_delta)
                    ))
                    prev_new_tick = new_tick
                else:
                    new_track.append(msg)
                    prev_new_tick = abs_tick
            elif (msg.type == 'note_off' or
                  (msg.type == 'note_on' and msg.velocity == 0)):
                key = (track_idx, msg.channel, msg.note, abs_tick)
                if key in off_to_delta:
                    delta = off_to_delta[key]
                    new_tick = abs_tick + delta
                    new_delta = new_tick - prev_new_tick
                    new_track.append(Message(
                        'note_off', channel=msg.channel,
                        note=msg.note, velocity=0,
                        time=max(0, new_delta)
                    ))
                    prev_new_tick = new_tick
                else:
                    new_track.append(msg)
                    prev_new_tick = abs_tick
            else:
                new_track.append(msg)
                prev_new_tick = abs_tick

        new_track.append(MetaMessage('end_of_track', time=1))
        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"  Saved grid-aligned MIDI: {output_path}")

    report = {
        "input_midi": midi_path,
        "audio_path": audio_path,
        "bpm": bpm,
        "ticks_per_beat": ticks_per_beat,
        "grid_divisions": grid_divisions,
        "grid_tick_spacing": grid_tick_spacing,
        "total_notes": len(note_ons),
        "snapped_notes": snapped_count,
        "corrections": corrections
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved correction report: {report_path}")

    return output_path, report


def detect_tempo_from_orig_audio(audio_path, sr=SR):
    return detect_tempo_from_onsets(audio_path, sr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Align MIDI notes to timing grid')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output aligned MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    parser.add_argument('--grid-divisions', type=int, default=4,
                        help='Grid subdivisions per beat (4=16th notes)')
    args = parser.parse_args()

    align_to_grid(
        args.input_midi, args.input_audio, args.output_midi,
        args.report, args.grid_divisions
    )
    print("\nTiming grid alignment complete.")
