#!/usr/bin/env python3
"""
SUB-PROBLEM 7: Note Duration / Length Correction (Bit by Bit)
==============================================================
Corrects note durations (note_off positions) in a MIDI file by comparing
each note's rendered envelope against the original audio, one note at a
time (bit by bit).

Approach:
  1. Load MIDI, collect note_on/note_off pairs per (track, channel, note).
  2. Load original audio, compute RMS envelope.
  3. For each note, measure actual audio energy decay in its window.
  4. Estimate optimal note_off time from where audio energy drops below
     a threshold relative to note peak.
  5. Shift note_off tick to match estimated end.
  6. Output per-note duration correction report.

Output: fixed MIDI + JSON correction report.
"""

import numpy as np
import librosa
from mido import MidiFile, MidiTrack, Message, MetaMessage
import json
import sys
import os

SR = 44100
TICKS_PER_BEAT = 960
BPM = 120
SECONDS_PER_TICK = 0.5 / TICKS_PER_BEAT


def tick_to_sec(tick):
    return tick * SECONDS_PER_TICK


def sec_to_tick(sec):
    return int(round(sec / SECONDS_PER_TICK))


def compute_rms_envelope(y, sr, hop_length=256):
    """Compute RMS envelope over time."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def estimate_note_end_sec(y, sr, note_start_sec, note_end_sec_guess,
                          threshold_ratio=0.15, min_duration_sec=0.1,
                          max_extra_sec=2.0):
    """
    Estimate where a note actually ends by looking at RMS decay after
    note onset.

    Args:
        y: mono audio
        sr: sample rate
        note_start_sec: note start time
        note_end_sec_guess: current MIDI note_off estimate
        threshold_ratio: end when RMS drops below this fraction of note peak
        min_duration_sec: minimum allowed note duration
        max_extra_sec: maximum extra time to search beyond guess
    """
    start_sample = int(note_start_sec * sr)
    search_end_sample = min(int((note_end_sec_guess + max_extra_sec) * sr), len(y))
    if start_sample >= search_end_sample:
        return note_end_sec_guess

    segment = y[start_sample:search_end_sample]
    if len(segment) == 0:
        return note_end_sec_guess

    # Peak RMS in the first 100ms of the note (attack region)
    attack_samples = min(int(0.1 * sr), len(segment))
    attack_seg = segment[:attack_samples]
    peak_rms = np.sqrt(np.mean(attack_seg ** 2)) if len(attack_seg) > 0 else 0
    if peak_rms < 1e-6:
        return note_end_sec_guess

    threshold = peak_rms * threshold_ratio

    # Compute RMS in sliding windows across the segment
    hop = 512
    win = 1024
    times_rel = []
    rms_vals = []
    for i in range(0, len(segment) - win + 1, hop):
        seg_rms = np.sqrt(np.mean(segment[i:i+win] ** 2))
        times_rel.append((start_sample + i) / sr)
        rms_vals.append(seg_rms)

    if not rms_vals:
        return note_end_sec_guess

    rms_vals = np.array(rms_vals)
    times_rel = np.array(times_rel)

    # Find first time RMS drops below threshold after the attack
    below = np.where(rms_vals < threshold)[0]
    if len(below) > 0:
        # The end is where it first drops below threshold
        est_end = times_rel[below[0]]
    else:
        # Never drops below threshold; keep current guess
        est_end = note_end_sec_guess

    # Clamp to reasonable bounds
    est_end = max(est_end, note_start_sec + min_duration_sec)
    est_end = min(est_end, note_end_sec_guess + max_extra_sec)

    return float(est_end)


def fix_note_durations(midi_path, audio_path, output_path, report_path,
                       threshold_ratio=0.15):
    """
    Bit-by-bit note duration correction.

    For each note:
      - Get start_sec from note_on tick
      - Get current end_sec from note_off tick (or guess)
      - Estimate actual end from audio RMS decay
      - Shift note_off tick to match
    """
    mid = MidiFile(midi_path)
    mid.ticks_per_beat = TICKS_PER_BEAT

    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Collect note events per (track, channel, note)
    active_notes = {}  # key -> {'on_tick':, 'on_sec':, 'velocity':}
    note_pairs = []  # completed notes
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note)
                active_notes[key] = {
                    'track_idx': track_idx,
                    'channel': msg.channel,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'on_tick': abs_tick,
                    'on_sec': tick_to_sec(abs_tick),
                    'off_msg_idx': None,
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                if key in active_notes:
                    info = active_notes.pop(key)
                    note_pairs.append({
                        **info,
                        'off_tick': abs_tick,
                        'off_sec': tick_to_sec(abs_tick),
                    })

    print(f"  Found {len(note_pairs)} complete notes")

    corrections = []
    fixed_count = 0

    for npair in note_pairs:
        start_sec = npair['on_sec']
        current_end_sec = npair['off_sec']
        current_dur = current_end_sec - start_sec

        # Estimate actual end from audio
        est_end_sec = estimate_note_end_sec(
            y, SR, start_sec, current_end_sec,
            threshold_ratio=threshold_ratio
        )

        est_dur = est_end_sec - start_sec
        delta_sec = est_end_sec - current_end_sec
        delta_tick = sec_to_tick(est_end_sec) - npair['off_tick']

        changed = abs(delta_sec) > 0.005  # > 5ms change
        if changed:
            fixed_count += 1

        npair['new_off_tick'] = sec_to_tick(est_end_sec)
        npair['new_off_sec'] = est_end_sec
        npair['changed'] = changed

        corrections.append({
            "track": npair['track_idx'],
            "channel": npair['channel'],
            "note": npair['note'],
            "velocity": npair['velocity'],
            "on_tick": npair['on_tick'],
            "on_sec": round(start_sec, 6),
            "old_off_tick": npair['off_tick'],
            "old_off_sec": round(current_end_sec, 6),
            "old_dur_sec": round(current_dur, 6),
            "new_off_tick": npair['new_off_tick'],
            "new_off_sec": round(est_end_sec, 6),
            "new_dur_sec": round(est_dur, 6),
            "delta_tick": delta_tick,
            "delta_sec": round(delta_sec, 6),
            "changed": changed
        })

    print(f"  Fixed {fixed_count}/{len(note_pairs)} note durations")

    # Rebuild MIDI tracks with corrected note_off positions
    new_mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    # Build lookup: (track_idx, channel, note) -> new_off_tick
    off_lookup = { (np['track_idx'], np['channel'], np['note']): np['new_off_tick']
                   for np in note_pairs }

    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0
        # Track which note_offs we've already replaced
        replaced_offs = set()

        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time

            if (msg.type == 'note_off' or
                (msg.type == 'note_on' and msg.velocity == 0)):
                key = (track_idx, msg.channel, msg.note)
                if key in off_lookup and key not in replaced_offs:
                    new_off_tick = off_lookup[key]
                    prev_tick = abs_tick - msg.time
                    new_delta = new_off_tick - prev_tick
                    new_track.append(Message(
                        'note_off', channel=msg.channel,
                        note=msg.note, velocity=0,
                        time=max(0, new_delta)
                    ))
                    abs_tick = new_off_tick
                    replaced_offs.add(key)
                else:
                    new_track.append(msg)
                    abs_tick += msg.time
            else:
                new_track.append(msg)
                abs_tick += msg.time

        new_track.append(MetaMessage('end_of_track', time=1))
        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"  Saved duration-fixed MIDI: {output_path}")

    report = {
        "input_midi": midi_path,
        "audio_path": audio_path,
        "threshold_ratio": threshold_ratio,
        "total_notes": len(note_pairs),
        "fixed_notes": fixed_count,
        "corrections": corrections
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved correction report: {report_path}")

    return output_path, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix note durations bit by bit')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output fixed MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='RMS threshold ratio for note end detection')
    args = parser.parse_args()

    fix_note_durations(
        args.input_midi, args.input_audio, args.output_midi,
        args.report, args.threshold
    )
    print("\nNote duration fix complete.")
