#!/usr/bin/env python3
"""
SUB-PROBLEM 9: Note Velocity / Dynamics Correction (Bit by Bit)
================================================================
Corrects note velocities in a MIDI file by measuring the RMS envelope
of the original audio in each note's time window, one note at a time.

Surgical approach:
  - Walk the MIDI track, accumulating absolute ticks precisely.
  - For each note_on, measure audio RMS in [note_start, note_end] window.
  - Map RMS to MIDI velocity (1-127) with calibrated curve.
  - Replace note_on velocity in-place, preserving absolute tick position.
  - Delta timing computed from previous event's NEW absolute tick.
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


def rms_to_velocity(rms):
    """
    Map RMS amplitude to MIDI velocity (1-127).
    Calibrated for piano: RMS 0.0001 -> vel 20, RMS 0.3 -> vel 127.
    Logarithmic mapping to match human loudness perception.
    """
    if rms <= 0:
        return 20
    # Log mapping: vel = 20 + 107 * log10(rms / 0.0001) / log10(0.3 / 0.0001)
    log_rms = np.log10(rms / 0.0001)
    log_max = np.log10(0.3 / 0.0001)  # = log10(3000) ≈ 3.477
    vel = 20 + 107 * (log_rms / log_max)
    return int(np.clip(vel, 1, 127))


def fix_note_velocities(midi_path, audio_path, output_path, report_path):
    """
    Bit-by-bit note velocity correction.

    For each note_on:
      - Get start_sec, end_sec from note_on/note_off pair
      - Measure RMS of original audio in that window
      - Map to velocity
      - Replace velocity, keep absolute tick unchanged
    """
    mid = MidiFile(midi_path)
    mid.ticks_per_beat = TICKS_PER_BEAT

    y_mono, _ = librosa.load(audio_path, sr=SR, mono=True)

    # First pass: collect note_on events and their matching note_off
    # We need end_sec for each note to measure RMS over the full window
    active_notes = {}  # (track, channel, note) -> {on_tick, on_sec, ...}
    note_windows = []  # list of {track, channel, note, on_tick, on_sec, off_sec}
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
                    'on_tick': abs_tick,
                    'on_sec': tick_to_sec(abs_tick),
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                if key in active_notes:
                    info = active_notes.pop(key)
                    note_windows.append({
                        **info,
                        'off_tick': abs_tick,
                        'off_sec': tick_to_sec(abs_tick),
                    })

    print(f"  Found {len(note_windows)} complete notes")

    # Second pass: walk tracks and replace velocities
    corrections = []
    changed_count = 0

    # Build lookup: (track_idx, msg_index_of_note_on) -> new_velocity
    # We need msg indices, so walk again
    velocity_lookup = {}  # (track_idx, msg_idx) -> new_vel
    for nw in note_windows:
        # Measure RMS in the note's audio window
        start_sample = int(nw['on_sec'] * SR)
        end_sample = int(nw['off_sec'] * SR)
        if end_sample <= start_sample or start_sample >= len(y_mono):
            vel = 64
        else:
            segment = y_mono[start_sample:end_sample]
            rms = float(np.sqrt(np.mean(segment ** 2)))
            vel = rms_to_velocity(rms)

        if vel != nw.get('original_velocity'):  # we don't have original here yet
            pass

        nw['new_velocity'] = vel

    # Now walk tracks with precise delta timing
    new_mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    # Build lookup: (track_idx, channel, note, on_tick) -> new_vel
    vel_lookup = {}
    for nw in note_windows:
        vel_lookup[(nw['track_idx'], nw['channel'], nw['note'], nw['on_tick'])] = nw['new_velocity']

    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0
        prev_new_tick = 0  # absolute tick of last event added to new_track

        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time  # absolute position of this event in original

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note, abs_tick)
                if key in vel_lookup:
                    new_vel = vel_lookup[key]
                    # Position unchanged; delta from prev_new_tick
                    new_delta = abs_tick - prev_new_tick
                    new_track.append(Message(
                        'note_on', channel=msg.channel,
                        note=msg.note, velocity=new_vel,
                        time=max(0, new_delta)
                    ))
                    prev_new_tick = abs_tick
                    if new_vel != msg.velocity:
                        changed_count += 1
                        corrections.append({
                            "track": track_idx, "channel": msg.channel,
                            "note": msg.note,
                            "old_velocity": msg.velocity,
                            "new_velocity": new_vel,
                            "tick": abs_tick,
                            "sec": round(tick_to_sec(abs_tick), 6),
                            "changed": True
                        })
                    else:
                        corrections.append({
                            "track": track_idx, "channel": msg.channel,
                            "note": msg.note,
                            "old_velocity": msg.velocity,
                            "new_velocity": new_vel,
                            "tick": abs_tick,
                            "sec": round(tick_to_sec(abs_tick), 6),
                            "changed": False
                        })
                else:
                    new_track.append(msg)
                    prev_new_tick = abs_tick
            else:
                new_track.append(msg)
                prev_new_tick = abs_tick

        new_track.append(MetaMessage('end_of_track', time=1))
        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"  Changed {changed_count}/{len(note_windows)} note velocities")
    print(f"  Saved velocity-fixed MIDI: {output_path}")

    report = {
        "input_midi": midi_path,
        "audio_path": audio_path,
        "total_notes": len(note_windows),
        "changed_notes": changed_count,
        "corrections": corrections
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved correction report: {report_path}")

    return output_path, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix note velocities bit by bit')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output fixed MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    args = parser.parse_args()

    fix_note_velocities(
        args.input_midi, args.input_audio, args.output_midi, args.report
    )
    print("\nNote velocity fix complete.")
