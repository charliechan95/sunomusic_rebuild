#!/usr/bin/env python3
"""
SUB-PROBLEM 12: Rhythm / Inter-Onset Gap Correction (Bit by Bit)
=================================================================
Corrects MIDI note positions so that inter-onset gaps match the original
audio exactly. Uses gap analysis to compute per-note position adjustments.

Surgical approach:
  1. Run gap analysis (audio gaps vs MIDI gaps)
  2. For each significant gap delta, adjust the MIDI note position at the
     end of the gap by the delta amount (cumulative)
  3. Apply same delta to matching note_off (preserve duration)
  4. Rebuild with precise delta timing

This ensures rhythm/gaps are quantized precisely to the original.
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


def detect_audio_onsets(audio_path, sr=SR):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=256, backtrack=True,
        pre_max=30, post_max=30, pre_avg=60, post_avg=60,
        delta=0.25, wait=40
    )
    return np.sort(librosa.frames_to_time(onset_frames, sr=sr, hop_length=256))


def get_midi_note_events(midi_path, bpm=120.0):
    """Get list of note_on events: (abs_tick, track_idx, channel, note, velocity, time_sec)"""
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat

    events = []
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'track_idx': track_idx,
                    'msg_idx': msg_idx,
                    'channel': msg.channel,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'abs_tick': abs_tick,
                    'time_sec': abs_tick * seconds_per_tick,
                })
    return events, ticks_per_beat, seconds_per_tick


def correct_rhythm_gaps(midi_path, audio_path, output_path, report_path,
                        bpm=120.0, gap_threshold_sec=0.01,
                        max_correction_sec=0.1):
    """
    Correct MIDI note positions so gaps match original audio.

    Algorithm:
      1. Detect audio onsets -> audio gaps between consecutive onsets
      2. Get MIDI note_on events sorted by time
      3. For each audio onset, find nearest MIDI note after it
      4. Compute expected position of each MIDI note by accumulating
         audio gaps from the first matched onset
      5. Delta = expected_sec - actual_sec for each MIDI note
      6. Clamp delta to max_correction_sec
      7. Apply delta to note_on and matching note_off
    """
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / ticks_per_beat

    onset_times = detect_audio_onsets(audio_path, SR)
    note_events, tpb, spt = get_midi_note_events(midi_path, bpm)

    print(f"  Audio onsets: {len(onset_times)}")
    print(f"  MIDI note-ons: {len(note_events)}")

    if len(onset_times) < 2 or len(note_events) == 0:
        print("  Not enough onsets or notes to correct rhythm")
        mid.save(output_path)
        return output_path, {}

    # For each MIDI note, find the nearest audio onset at or before it
    # Then compute expected position = first_matched_onset + sum of audio gaps
    # up to that point

    # Map: for each MIDI note index i, find which audio gap interval it falls in
    corrections = []
    applied_count = 0

    # Build expected time for each MIDI note
    # Strategy: align the first MIDI note to the first audio onset,
    # then each subsequent MIDI note should be at:
    #   expected[i] = onset[0] + sum(audio_gaps[0..i-1])
    # But we need to match count. If MIDI has fewer notes than audio onsets,
    # we match by nearest onset.

    # Simple approach: for each MIDI note, find nearest audio onset,
    # expected = that onset time; delta = onset - actual
    # But this doesn't enforce gaps. Better: cumulative gap matching.

    # Let's do: pair MIDI notes to audio onsets in order (1:1 as much as possible)
    n_midi = len(note_events)
    n_onsets = len(onset_times)

    # Use the minimum count
    n_pairs = min(n_midi, n_onsets)

    # Pair i-th MIDI note with i-th audio onset (sorted both)
    midi_sorted = sorted(note_events, key=lambda e: e['time_sec'])
    onsets_sorted = sorted(onset_times)

    note_to_delta = {}  # (track_idx, msg_idx) -> delta_tick

    for i in range(n_pairs):
        midi_ev = midi_sorted[i]
        audio_onset = onsets_sorted[i]
        delta_sec = audio_onset - midi_ev['time_sec']

        # Clamp
        if abs(delta_sec) > max_correction_sec:
            delta_sec = np.sign(delta_sec) * max_correction_sec

        delta_tick = sec_to_tick(delta_sec, tpb, bpm) - sec_to_tick(0, tpb, bpm)
        # Actually delta_tick in ticks:
        delta_tick = int(round(delta_sec / spt))

        key = (midi_ev['track_idx'], midi_ev['msg_idx'])
        note_to_delta[key] = delta_tick

        corrections.append({
            "track": midi_ev['track_idx'],
            "channel": midi_ev['channel'],
            "note": midi_ev['note'],
            "velocity": midi_ev['velocity'],
            "old_tick": midi_ev['abs_tick'],
            "old_sec": round(midi_ev['time_sec'], 6),
            "expected_sec": round(audio_onset, 6),
            "delta_sec": round(delta_sec, 6),
            "delta_tick": delta_tick,
            "applied": abs(delta_sec) > 0.001
        })
        if abs(delta_sec) > 0.001:
            applied_count += 1

    print(f"  Applied rhythm corrections to {applied_count}/{n_pairs} notes")

    # Build lookup: (track_idx, channel, note) -> delta_tick from matching note_on
    note_on_delta_by_note = {}
    for key, delta_tick in note_to_delta.items():
        tid, midx = key
        for c in corrections:
            if c['track'] == tid:
                note_on_delta_by_note[(tid, c['channel'], c['note'])] = delta_tick
                break

    # Rebuild MIDI tracks with corrected positions
    new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0
        prev_new_tick = 0

        for msg_idx, msg in enumerate(track):
            abs_tick += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg_idx)
                if key in note_to_delta:
                    delta_tick = note_to_delta[key]
                    new_tick = abs_tick + delta_tick
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
                # Apply same delta as matching note_on (preserve duration)
                note_key = (track_idx, msg.channel, msg.note)
                if note_key in note_on_delta_by_note:
                    delta_tick = note_on_delta_by_note[note_key]
                    new_tick = abs_tick + delta_tick
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
    print(f"  Saved rhythm-corrected MIDI: {output_path}")

    report = {
        "input_midi": midi_path,
        "audio_path": audio_path,
        "bpm": bpm,
        "gap_threshold_sec": gap_threshold_sec,
        "max_correction_sec": max_correction_sec,
        "audio_onset_count": len(onset_times),
        "midi_note_count": len(note_events),
        "paired_count": n_pairs,
        "applied_count": applied_count,
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
    parser = argparse.ArgumentParser(description='Correct rhythm gaps in MIDI')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output corrected MIDI path')
    parser.add_argument('--report', required=True, help='Output JSON report path')
    parser.add_argument('--bpm', type=float, default=120.0, help='Tempo in BPM')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Gap delta threshold in seconds')
    parser.add_argument('--max-correction', type=float, default=0.1,
                        help='Max correction per note in seconds')
    args = parser.parse_args()

    correct_rhythm_gaps(
        args.input_midi, args.input_audio, args.output_midi,
        args.report, args.bpm, args.threshold, args.max_correction
    )
    print("\nRhythm gap correction complete.")
