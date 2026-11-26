#!/usr/bin/env python3
"""
SUB-PROBLEM 6 (FIXED): Note Onset Position Correction (Bit by Bit)
===================================================================
Corrects note start positions in a MIDI file by snapping them to the
nearest detected audio onset. When a note_on is shifted, the matching
note_off is shifted by the SAME delta to preserve duration exactly.

Surgical timing:
  - Walk each track, accumulating absolute ticks precisely.
  - prev_new_tick tracks the absolute tick of the last event written
    to the new track; delta = current_event_new_tick - prev_new_tick.
  - When note_on is snapped, note_off gets the same +delta shift.
"""

import numpy as np
import librosa
from mido import MidiFile, MidiTrack, Message, MetaMessage
import json
import sys
import os

SR = 44100
TICKS_PER_BEAT = 960
SECONDS_PER_TICK = 0.5 / TICKS_PER_BEAT


def tick_to_sec(tick):
    return tick * SECONDS_PER_TICK


def sec_to_tick(sec):
    return int(round(sec / SECONDS_PER_TICK))


def detect_audio_onsets(audio_path, sr=SR):
    """Detect precise onset times in original audio."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=256, backtrack=True,
        pre_max=30, post_max=30, pre_avg=60, post_avg=60,
        delta=0.25, wait=40
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)
    return np.sort(onset_times)


def fix_note_positions(midi_path, audio_path, output_path, report_path,
                       tolerance_sec=0.08):
    """
    Bit-by-bit note onset position correction.

    For each note_on event:
      - Convert abs_tick to seconds
      - Find nearest audio onset within tolerance_sec
      - If found, compute delta_tick = new_tick - old_tick
      - Apply same delta_tick to the matching note_off (preserve duration)
      - Rebuild track with precise delta timing
    """
    mid = MidiFile(midi_path)
    mid.ticks_per_beat = TICKS_PER_BEAT

    onset_times = detect_audio_onsets(audio_path, SR)
    print(f"  Detected {len(onset_times)} audio onsets")

    # Collect note_on events and note_off events per track
    # note_ons: (track_idx, channel, note, on_abs_tick) -> info
    # note_offs: (track_idx, channel, note) -> off_abs_tick
    note_ons = {}
    note_offs = {}
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note, abs_tick)
                note_ons[key] = {
                    'track_idx': track_idx,
                    'channel': msg.channel,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'on_abs_tick': abs_tick,
                    'on_sec': tick_to_sec(abs_tick),
                    'msg_idx': msg_idx,
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                # Store the FIRST note_off for this note (should match the note_on)
                if key not in note_offs:
                    note_offs[key] = abs_tick

    print(f"  Found {len(note_ons)} note_on events, {len(note_offs)} note_off events")

    # Compute snapping for each note_on
    corrections = []
    snapped_count = 0
    on_to_delta = {}  # (track, channel, note, on_tick) -> delta_tick

    for key, info in note_ons.items():
        approx_sec = info['on_sec']
        diffs = np.abs(onset_times - approx_sec)
        nearest_idx = int(np.argmin(diffs))
        nearest_sec = float(onset_times[nearest_idx])
        diff = float(diffs[nearest_idx])

        snapped = False
        delta_tick = 0
        new_tick = info['on_abs_tick']
        if diff < tolerance_sec:
            new_tick = sec_to_tick(nearest_sec)
            delta_tick = new_tick - info['on_abs_tick']
            snapped = True
            snapped_count += 1

        on_to_delta[key] = delta_tick
        corrections.append({
            "track": info['track_idx'],
            "channel": info['channel'],
            "note": info['note'],
            "velocity": info['velocity'],
            "old_tick": info['on_abs_tick'],
            "old_sec": round(info['on_sec'], 6),
            "new_tick": new_tick,
            "new_sec": round(tick_to_sec(new_tick), 6),
            "delta_tick": delta_tick,
            "delta_sec": round(tick_to_sec(delta_tick), 6),
            "snapped": snapped,
            "nearest_onset_sec": round(nearest_sec, 6),
            "onset_diff_sec": round(diff, 6)
        })

    print(f"  Snapped {snapped_count}/{len(note_ons)} notes to onsets")

    # Build lookup: (track_idx, channel, note) -> delta_tick (from matching note_on)
    # Match note_off to note_on by (track, channel, note) — use the note_on with
    # the smallest on_tick that is <= off_tick
    note_on_by_note = {}  # (track, channel, note) -> list of (on_tick, delta)
    for key, delta in on_to_delta.items():
        tid, ch, note, on_tick = key
        note_on_by_note.setdefault((tid, ch, note), []).append((on_tick, delta))

    # For each note_off, find the matching note_on delta
    off_to_delta = {}  # (track, channel, note, off_tick) -> delta_tick
    for key, off_tick in note_offs.items():
        tid, ch, note = key
        candidates = note_on_by_note.get((tid, ch, note), [])
        # Find the note_on with largest on_tick <= off_tick
        best_delta = 0
        best_on_tick = -1
        for on_tick, delta in candidates:
            if on_tick <= off_tick and on_tick > best_on_tick:
                best_on_tick = on_tick
                best_delta = delta
        off_to_delta[(tid, ch, note, off_tick)] = best_delta

    # Rebuild MIDI tracks with precise delta timing
    new_mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0
        prev_new_tick = 0  # absolute tick of last event written to new_track

        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time  # original absolute tick of this event

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
    print(f"  Saved position-fixed MIDI: {output_path}")

    report = {
        "input_midi": midi_path,
        "audio_path": audio_path,
        "tolerance_sec": tolerance_sec,
        "total_notes": len(note_ons),
        "snapped_notes": snapped_count,
        "corrections": corrections
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved correction report: {report_path}")

    return output_path, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix note onset positions bit by bit')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output fixed MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    parser.add_argument('--tolerance', type=float, default=0.08, help='Snap tolerance in seconds')
    args = parser.parse_args()

    fix_note_positions(
        args.input_midi, args.input_audio, args.output_midi,
        args.report, args.tolerance
    )
    print("\nNote position fix complete.")
