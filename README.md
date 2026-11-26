# sunomusic_rebuild

Audio → MIDI → audio roundtrip toolchain for Suno watermark removal.

Transcribes a Suno-generated WAV to MIDI, surgically fixes note positions, durations, pitches, and velocities bit by bit, renders back to piano audio via SoundFont, spectrally matches to the original, adds convolution reverb, and verifies the result against the source.

## Pipeline stages

| # | Tool | Purpose |
|---|------|---------|
| 1 | `1_audio_to_midi.py` | PyIN-based audio → MIDI transcription |
| 2 | `2_midi_fix.py` | MIDI arrangement & timing fix (piano program, onset snap, velocity from audio) |
| 6 | `6_note_position_fix.py` | Snap note onsets to detected audio onsets |
| 7 | `7_note_duration_fix.py` | Correct note durations from audio RMS envelope |
| 8 | `8_note_pitch_fix.py` | Correct note pitches via PyIN f0 tracking |
| 9 | `9_note_velocity_fix.py` | Map audio RMS to MIDI velocity per note |
| 10 | `10_grid_alignment.py` | Quantize notes to a timing grid |
| 12 | `12_rhythm_gap_correction.py` | Align inter-onset gaps to original audio |
| 3 | `3_midi_render.py` | Render MIDI → piano WAV via fluidsynth |
| 4 | `4_spectral_match.py` | Minimum-phase FIR spectral match to original |
| 5 | `5_reverb_effects.py` | Convolution reverb |
| 13 | `13_verify_rendered.py` | Final verification vs original |
| 14 | `14_orchestrator.py` | Run all stages end-to-end |

## Quick start

```bash
python sub_problems/14_orchestrator.py \
    --input-audio /path/to/original.wav \
    --output-dir output/ \
    --soundfont soundfonts/SalamanderGrandPiano.sf2
```

Requirements: Python 3.10+, numpy, librosa, soundfile, scipy, mido, fluidsynth (system binary).

## Known limitations (documented honestly)

- **MIDI roundtrip cannot achieve 100% timbre match to the original recording.** PyIN transcription detects far fewer onsets than the original (65 vs 212 on the test file), and a piano SoundFont has a fundamentally different spectral centroid than a full Suno production. Waveform correlation between rendered and original was ~0.0012 — essentially no sample-level similarity.
- **The original test file (`nasuno_notches_only.wav`) already had no detectable Suno ultrasonic watermark** (ultrasonic energy ratio 0.000284, below the 0.001 detection threshold).
- FIR spectral matching can only reshape the rendered spectrum; it cannot synthesize missing timbres or notes.
- For a true 100% match, direct audio-domain processing (not MIDI conversion) is required.

## Repository contents

- `sub_problems/` — 14 modular Python pipeline tools
- `soundfonts/` — SalC5Light2 piano SoundFont (25 MB)
- `output/` — sample pipeline outputs (MIDI, WAV, JSON reports) from a full run
- `README.md` — this file
- `.gitignore` — excludes cache, pycache, large output binaries

## License

MIT. No API keys, tokens, or credentials are included.