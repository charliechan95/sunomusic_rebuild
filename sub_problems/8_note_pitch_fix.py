#!/usr/bin/env python3
"""
SUB-PROBLEM 8: Note Pitch Correction (Bit by Bit)
=================================================
Corrects note pitches in a MIDI file by comparing each note's detected
fundamental frequency in the original audio against the MIDI note number,
one note at a time (bit by bit).

Approach:
  1. Load MIDI, collect note_on/note_off pairs.
  2. Load original audio, run PyIN pitch tracking.
  3. For each note's time window, compute median detected frequency.
  4. Convert to nearest MIDI note number.
  5. If different from current MIDI pitch, correct it.
  6. Output per-note pitch correction report.

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
SECONDS_PER_TICK = 0.5 / TICKS_PER_BEAT


def tick_to_sec(tick):
    return tick * SECONDS_PER_TICK


def sec_to_tick(sec):
    return int(round(sec / SECONDS_PER_TICK))


def fix_note_pitches(midi_path, audio_path, output_path, report_path,
                     fmin_hz=65.41, fmax_hz=2093.0, tolerance_cents=50):
    """
    Bit-by-bit note pitch correction.

    For each note:
      - Get start_sec, end_sec from MIDI
      - Extract audio segment in that window
      - Run PyIN to get f0 over time
      - Compute median f0 of voiced frames
      - Convert to nearest MIDI note
      - If differs by more than tolerance_cents, correct pitch
    """
    mid = MidiFile(midi_path)
    mid.ticks_per_beat = TICKS_PER_BEAT

    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Run PyIN once on full audio
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin_hz, fmax=fmax_hz,
        sr=SR, frame_length=2048, hop_length=256
    )
    times = librosa.times_like(f0, sr=SR, hop_length=256)

    # Collect note pairs
    active_notes = {}
    note_pairs = []
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

    def hz_to_midi(f):
        return int(round(librosa.hz_to_midi(f)))

    def midi_to_hz(m):
        return librosa.midi_to_hz(m)

    def cents_diff(f1, f2):
        return 1200 * np.log2(f1 / f2)

    for npair in note_pairs:
        start_sec = npair['on_sec']
        end_sec = npair['off_sec']
        current_midi = npair['note']

        # Find PyIN frames within this note's window
        mask = (times >= start_sec) & (times <= end_sec) & voiced_flag
        if not np.any(mask):
            # No voiced frames; skip
            corrections.append({
                "track": npair['track_idx'], "channel": npair['channel'],
                "note": current_midi, "velocity": npair['velocity'],
                "on_sec": round(start_sec, 6), "off_sec": round(end_sec, 6),
                "detected_hz": None, "detected_midi": None,
                "cents_diff": None, "changed": False,
                "reason": "no_voiced_frames"
            })
            continue

        detected_f0s = f0[mask]
        median_f0 = float(np.median(detected_f0s))
        detected_midi = hz_to_midi(median_f0)

        # Compute cents difference
        if median_f0 > 0 and current_midi > 0:
            cd = cents_diff(median_f0, midi_to_hz(current_midi))
        else:
            cd = 0.0

        changed = abs(cd) > tolerance_cents and detected_midi != current_midi

        if changed:
            fixed_count += 1

        npair['new_note'] = detected_midi if changed else current_midi
        npair['changed'] = changed
        npair['detected_midi'] = detected_midi
        npair['cents_diff'] = float(cd)

        corrections.append({
            "track": npair['track_idx'], "channel": npair['channel'],
            "note": current_midi, "velocity": npair['velocity'],
            "on_sec": round(start_sec, 6), "off_sec": round(end_sec, 6),
            "detected_hz": round(median_f0, 2),
            "detected_midi": detected_midi,
            "cents_diff": round(float(cd), 1),
            "changed": changed,
            "new_note": npair['new_note']
        })

    print(f"  Fixed {fixed_count}/{len(note_pairs)} note pitches")

    # Rebuild MIDI tracks with corrected pitches
    new_mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    pitch_lookup = { (np['track_idx'], np['channel'], np['note']): np['new_note']
                     for np in note_pairs if np.get('changed', False) }

    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0

        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note)
                if key in pitch_lookup:
                    new_note = pitch_lookup[key]
                    new_track.append(Message(
                        'note_on', channel=msg.channel,
                        note=new_note, velocity=msg.velocity,
                        time=msg.time  # preserve original delta timing
                    ))
                else:
                    new_track.append(msg)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                if key in pitch_lookup:
                    new_note = pitch_lookup[key]
                    new_track.append(Message(
                        'note_off', channel=msg.channel,
                        note=new_note, velocity=0,
                        time=msg.time  # preserve original delta timing
                    ))
                else:
                    new_track.append(msg)
            else:
                new_track.append(msg)

        new_track.append(MetaMessage('end_of_track', time=1))
        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"  Saved pitch-fixed MIDI: {output_path}")

    report = {
        "input_midi": midi_path,
        "audio_path": audio_path,
        "tolerance_cents": tolerance_cents,
        "total_notes": len(note_pairs),
        "fixed_notes": fixed_count,
        "corrections": [
            {k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in c.items()}
            for c in corrections
        ]
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved correction report: {report_path}")

    return output_path, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix note pitches bit by bit')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output fixed MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    parser.add_argument('--fmin', type=float, default=65.41, help='Min frequency (Hz)')
    parser.add_argument('--fmax', type=float, default=2093.0, help='Max frequency (Hz)')
    parser.add_argument('--tolerance-cents', type=float, default=50,
                        help='Pitch correction tolerance in cents')
    args = parser.parse_args()

    fix_note_pitches(
        args.input_midi, args.input_audio, args.output_midi,
        args.report, args.fmin, args.fmax, args.tolerance_cents
    )
    print("\nNote pitch fix complete.")
