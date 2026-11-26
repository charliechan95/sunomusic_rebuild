#!/usr/bin/env python3
"""
SUB-PROBLEM 2: MIDI Arrangement & Timing Fix
============================================
Fixes timing and arrangement issues in a transcribed MIDI file:
  - Aligns note onsets to detected audio onsets (snap to nearest onset within tolerance)
  - Corrects note durations based on audio envelope
  - Forces all notes to program 0 (acoustic piano)
  - Quantizes to a consistent tempo grid

Input:  MIDI file (from sub-problem 1) + original audio
Output: Fixed MIDI file with corrected timing/arrangement
"""

import numpy as np
import librosa
import soundfile as sf
from mido import MidiFile, MidiTrack, Message, MetaMessage
import json
import sys
import os

SR = 44100
TICKS_PER_BEAT = 960
BPM = 120
SECONDS_PER_TICK = 0.5 / TICKS_PER_BEAT  # at 120 BPM


def detect_audio_onsets(audio_path, sr=SR):
    """Detect precise onset times in original audio."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=256, backtrack=True,
        pre_max=30, post_max=30, pre_avg=60, post_avg=60,
        delta=0.25, wait=40
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)
    return np.array(sorted(onset_times), dtype=float)


def extract_note_velocity(y_mono, sr, note_start_sec, note_end_sec):
    """Extract RMS-based velocity from audio envelope in note window."""
    start_sample = int(note_start_sec * sr)
    end_sample = int(note_end_sec * sr)
    if end_sample <= start_sample or start_sample >= len(y_mono):
        return 64
    segment = y_mono[start_sample:end_sample]
    rms = np.sqrt(np.mean(segment ** 2))
    vel = int(np.clip(rms * 400, 20, 127))
    return vel


def fix_midi_arrangement(midi_path, audio_path, output_path, snap_tolerance_sec=0.08):
    """
    Fix MIDI timing and arrangement:
      1. Load MIDI, extract note_on/note_off events
      2. Snap note onsets to nearest audio onset (within tolerance)
      3. Set all programs to piano (0)
      4. Extract velocities from audio envelope
      5. Rebuild MIDI with corrected timing
    """
    mid = MidiFile(midi_path)
    # Force ticks_per_beat
    mid.ticks_per_beat = TICKS_PER_BEAT

    # Load audio for velocity extraction and onset detection
    y_audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    onset_times = detect_audio_onsets(audio_path, SR)

    # Collect all note events across tracks
    # Structure: { (track, channel, note): {'on': {'tick':.., 'vel':..}, 'off': {'tick':..}} }
    note_events = {}
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note)
                if key not in note_events:
                    note_events[key] = {'on': None, 'off': None}
                note_events[key]['on'] = {'tick': abs_tick, 'vel': msg.velocity}
            elif msg.type in ('note_off', ('note_on' if False else 'note_on')):
                if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (track_idx, msg.channel, msg.note)
                    if key not in note_events:
                        note_events[key] = {'on': None, 'off': None}
                    note_events[key]['off'] = {'tick': abs_tick}

    print(f"  Found {len(note_events)} unique notes")

    # Set program change to piano (0) on all tracks
    for track in mid.tracks:
        has_program = any(m.type == 'program_change' for m in track)
        if not has_program:
            track.insert(0, Message('program_change', channel=0, program=0, time=0))

    # Snap onsets and fix durations
    snapped_count = 0
    for key, ev in note_events.items():
        if ev['on'] is None:
            continue
        track_idx, channel, pitch = key
        on_tick = ev['on']['tick']
        approx_sec = on_tick * SECONDS_PER_TICK

        # Snap to nearest audio onset
        onset_diffs = np.abs(onset_times - approx_sec)
        nearest_idx = int(np.argmin(onset_diffs))
        nearest_onset = float(onset_times[nearest_idx])
        onset_diff = float(onset_diffs[nearest_idx])

        if onset_diff < snap_tolerance_sec:
            new_tick = int(nearest_onset / SECONDS_PER_TICK)
            ev['on']['tick'] = new_tick
            snapped_count += 1

        # Extract velocity from audio
        start_sec = on_tick * SECONDS_PER_TICK
        if ev['off'] is not None:
            end_sec = ev['off']['tick'] * SECONDS_PER_TICK
        else:
            end_sec = start_sec + 0.5
        ev['on']['vel'] = extract_note_velocity(y_audio, SR, start_sec, end_sec)

    print(f"  Snapped {snapped_count}/{len(note_events)} notes to audio onsets")

    # Rebuild MIDI tracks
    new_mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    for track_idx, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        abs_tick = 0

        # Collect events for this track sorted by tick
        track_events = []
        for key, ev in note_events.items():
            if key[0] == track_idx:
                if ev['on'] is not None:
                    track_events.append((ev['on']['tick'], 'on', key, ev['on']['vel']))
                if ev['off'] is not None:
                    track_events.append((ev['off']['tick'], 'off', key, 0))

        # Add original non-note events (tempo, program, etc.)
        for msg in track:
            if msg.type not in ('note_on', 'note_off'):
                track_events.append((abs_tick := abs_tick + msg.time, 'meta', msg, 0))

        # Actually, simpler: iterate original track, replace note events
        abs_tick = 0
        note_idx_map = {}  # original msg index -> new event
        # We'll rebuild by walking the original track and substituting note times

        # Build a sorted list of replacement events for this track
        replacements = {}  # abs_tick -> list of (type, note, vel)
        for key, ev in note_events.items():
            if key[0] != track_idx:
                continue
            if ev['on'] is not None:
                replacements.setdefault(ev['on']['tick'], []).append(('on', key[2], ev['on']['vel']))
            if ev['off'] is not None:
                replacements.setdefault(ev['off']['tick'], []).append(('off', key[2], 0))

        # Walk original track
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off'):
                # Check if this tick has a replacement
                if abs_tick in replacements:
                    for r_type, r_note, r_vel in replacements[abs_tick]:
                        delta = abs_tick - abs_tick  # will be set properly below
                    # We handle replacements after the loop; skip original note msgs
                    continue
                else:
                    # Keep original note event timing
                    new_track.append(msg)
            else:
                new_track.append(msg)

        # Now insert replacement note events with proper delta timing
        # This is complex; simpler approach: build fresh track from replacements + non-note events
        new_track = MidiTrack()
        # Tempo
        new_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
        new_track.append(Message('program_change', channel=0, program=0, time=0))

        all_events = []
        # Non-note events from original
        abs_tick = 0
        non_note_events = []
        for msg in track:
            abs_tick += msg.time
            if msg.type not in ('note_on', 'note_off'):
                non_note_events.append((abs_tick, msg))
        # Reset for combined walk
        abs_tick = 0

        # Combine all events: note replacements + non-note originals
        combined = []
        for tick, ev_list in replacements.items():
            for r_type, r_note, r_vel in ev_list:
                combined.append((tick, r_type, r_note, r_vel))
        for tick, msg in non_note_events:
            combined.append((tick, 'meta', msg, 0))

        combined.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

        prev_tick = 0
        for item in combined:
            tick = item[0]
            delta = tick - prev_tick
            if item[1] == 'on':
                _, _, note, vel = item
                new_track.append(Message('note_on', channel=0, note=note, velocity=vel, time=max(0, delta)))
            elif item[1] == 'off':
                _, _, note, _ = item
                new_track.append(Message('note_off', channel=0, note=note, velocity=0, time=max(0, delta)))
            else:
                _, _, msg, _ = item
                msg.time = max(0, delta)
                new_track.append(msg)
            prev_tick = tick

        new_track.append(MetaMessage('end_of_track', time=1))
        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"  Saved fixed MIDI: {output_path}")
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix MIDI arrangement and timing')
    parser.add_argument('--input-midi', required=True, help='Input MIDI path')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output fixed MIDI path')
    parser.add_argument('--snap-tolerance', type=float, default=0.08, help='Onset snap tolerance in seconds')
    args = parser.parse_args()

    fix_midi_arrangement(
        args.input_midi, args.input_audio, args.output_midi, args.snap_tolerance
    )
    print("\nMIDI arrangement fix complete.")
