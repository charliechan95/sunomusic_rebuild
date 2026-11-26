#!/usr/bin/env python3
"""
SUB-PROBLEM 1: Audio → MIDI Transcription
==========================================
Converts the original audio recording into a MIDI file with note onsets,
pitches, velocities, and durations. Uses Basic Pitch for transcription.

Input:  original audio (watermark-removed WAV)
Output: MIDI file with transcribed notes
"""

import numpy as np
import librosa
import soundfile as sf
import json
import sys
import os

SR = 44100


def transcribe_audio_to_midi(audio_path, output_midi_path, output_notes_json=None):
    """
    Transcribe audio to MIDI using librosa-based pitch tracking.
    
    Falls back to librosa.pyin + onset detection if basic_pitch is unavailable.
    Returns a list of note dicts: {start_sec, end_sec, pitch, velocity}
    """
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr

    # --- Step 1: Detect onsets ---
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=256, backtrack=True,
        pre_max=30, post_max=30, pre_avg=60, post_avg=60,
        delta=0.25, wait=40
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)
    print(f"  Detected {len(onset_times)} onsets")

    # --- Step 2: Pitch tracking with PyIN (monophonic estimate per frame) ---
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, frame_length=2048, hop_length=256
    )
    times = librosa.times_like(f0, sr=sr, hop_length=256)

    # --- Step 3: Group voiced frames into notes ---
    notes = []
    i = 0
    while i < len(f0):
        if voiced_flag[i]:
            start_time = times[i]
            # Find end of this voiced segment
            j = i
            while j < len(f0) and voiced_flag[j]:
                j += 1
            end_time = times[j - 1] if j > i else start_time

            # Median pitch in this segment
            segment_pitches = f0[i:j]
            median_pitch = np.median(segment_pitches[segment_pitches > 0])
            midi_note = int(round(librosa.hz_to_note(median_pitch).split('-')[0])) if False else int(round(librosa.hz_to_midi(median_pitch)))
            midi_note = int(np.clip(midi_note, 21, 108))

            # Velocity from RMS in this segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            if end_sample > start_sample:
                rms = np.sqrt(np.mean(y[start_sample:end_sample] ** 2))
                velocity = int(np.clip(rms * 400, 20, 127))
            else:
                velocity = 64

            notes.append({
                "start_sec": round(float(start_time), 4),
                "end_sec": round(float(end_time), 4),
                "pitch": midi_note,
                "velocity": velocity,
                "duration_sec": round(float(end_time - start_time), 4)
            })
            i = j
        else:
            i += 1

    print(f"  Transcribed {len(notes)} notes")

    # --- Step 4: Write MIDI ---
    try:
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        mid = MidiFile(ticks_per_beat=960)
        track = MidiTrack()
        mid.tracks.append(track)

        # Tempo: 120 BPM = 500000 microseconds per beat
        track.append(MetaMessage('set_tempo', tempo=500000, time=0))
        track.append(Message('program_change', channel=0, program=0, time=0))

        # Convert notes to MIDI events sorted by start time
        events = []
        for n in notes:
            start_tick = int(n["start_sec"] / 0.5 * 960)  # 120 BPM: 0.5s per beat
            end_tick = int(n["end_sec"] / 0.5 * 960)
            events.append(('on', start_tick, n["pitch"], n["velocity"]))
            events.append(('off', end_tick, n["pitch"], 0))

        events.sort(key=lambda e: (e[1], 0 if e[0] == 'off' else 1))

        abs_tick = 0
        for ev_type, tick, pitch, vel in events:
            delta = tick - abs_tick
            if ev_type == 'on':
                track.append(Message('note_on', channel=0, note=pitch, velocity=vel, time=max(0, delta)))
            else:
                track.append(Message('note_off', channel=0, note=pitch, velocity=0, time=max(0, delta)))
            abs_tick = tick

        track.append(MetaMessage('end_of_track', time=1))
        mid.save(output_midi_path)
        print(f"  Saved MIDI: {output_midi_path}")
    except ImportError:
        print("  mido not available, saving notes JSON only")

    if output_notes_json:
        with open(output_notes_json, 'w') as f:
            json.dump({"notes": notes, "duration": duration, "sr": sr}, f, indent=2)
        print(f"  Saved notes JSON: {output_notes_json}")

    return notes, duration


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transcribe audio to MIDI')
    parser.add_argument('--input', required=True, help='Input audio WAV path')
    parser.add_argument('--output-midi', required=True, help='Output MIDI path')
    parser.add_argument('--output-json', help='Output notes JSON path')
    args = parser.parse_args()

    notes, duration = transcribe_audio_to_midi(
        args.input, args.output_midi, args.output_json
    )
    print(f"\nTranscription complete: {len(notes)} notes, {duration:.2f}s")
