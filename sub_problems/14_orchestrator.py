#!/usr/bin/env python3
"""
SUB-PROBLEM 14: Full Pipeline Orchestrator
==========================================
Runs the complete Suno watermark removal pipeline end-to-end:

  1. Audio → MIDI transcription (tool 1)
  2. Fix note onset positions bit by bit (tool 6)
  3. Fix note durations bit by bit (tool 7)
  4. Fix note pitches bit by bit (tool 8)
  5. Fix note velocities bit by bit (tool 9)
  6. Align notes to timing grid (tool 10)
  7. Correct rhythm/gaps bit by bit (tool 12)
  8. Render MIDI → piano audio via fluidsynth (tool 3)
  9. Spectral match to original (tool 4)
  10. Add convolution reverb (tool 5)
  11. Final verification vs original (tool 13)

Each step writes a JSON report. The final output is a watermark-free piano
rendering that matches the original recording as closely as possible.

Usage:
    python 14_orchestrator.py \
        --input-audio /path/to/original.wav \
        --output-dir /path/to/output/
"""

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


def run(cmd, step_name):
    """Run a subprocess command and return its return code."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"{'=' * 60}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print stdout tail (last 30 lines) so we see progress but not flood
    out_lines = result.stdout.strip().splitlines()
    for line in out_lines[-30:]:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"\n  ERROR: {step_name} failed with code {result.returncode}")
        err_lines = result.stderr.strip().splitlines()
        for line in err_lines[-20:]:
            print(f"  STDERR: {line}")
        return False
    print(f"  OK: {step_name} completed")
    return True


def run_pipeline(input_audio, output_dir, soundfont, bpm=120.0,
                 ticks_per_beat=960, tolerance_sec=0.08,
                 grid_divisions=4, gap_threshold_sec=0.01,
                 max_correction_sec=0.1):
    """
    Run the full pipeline. Returns dict of all output paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Paths ──
    p_transcribed_midi = out / "1_transcribed.mid"
    p_transcribed_report = out / "1_transcribed_report.json"

    p_pos_fixed_midi = out / "2_position_fixed.mid"
    p_pos_report = out / "2_position_report.json"

    p_dur_fixed_midi = out / "3_duration_fixed.mid"
    p_dur_report = out / "3_duration_report.json"

    p_pitch_fixed_midi = out / "4_pitch_fixed.mid"
    p_pitch_report = out / "4_pitch_report.json"

    p_vel_fixed_midi = out / "5_velocity_fixed.mid"
    p_vel_report = out / "5_velocity_report.json"

    p_grid_midi = out / "6_grid_aligned.mid"
    p_grid_report = out / "6_grid_report.json"

    p_rhythm_midi = out / "7_rhythm_corrected.mid"
    p_rhythm_report = out / "7_rhythm_report.json"

    p_rendered = out / "8_rendered_piano.wav"
    p_rendered_report = out / "8_render_report.json"

    p_spectral_matched = out / "9_spectral_matched.wav"
    p_spectral_report = out / "9_spectral_report.json"

    p_reverb = out / "10_final_reverb.wav"
    p_reverb_report = out / "10_reverb_report.json"

    p_verify = out / "11_verification_report.json"

    python = "/opt/anaconda3/bin/python3"

    # ── Step 1: Audio → MIDI ──
    ok = run([
        python, str(TOOLS_DIR / "1_audio_to_midi.py"),
        "--input", str(input_audio),
        "--output-midi", str(p_transcribed_midi),
        "--output-json", str(p_transcribed_report)
    ], "Audio → MIDI transcription")
    if not ok:
        return None

    # ── Step 2: Fix note onset positions ──
    ok = run([
        python, str(TOOLS_DIR / "6_note_position_fix.py"),
        "--input-midi", str(p_transcribed_midi),
        "--input-audio", str(input_audio),
        "--output-midi", str(p_pos_fixed_midi),
        "--report", str(p_pos_report),
        "--tolerance", str(tolerance_sec)
    ], "Note onset position correction")
    if not ok:
        return None

    # ── Step 3: Fix note durations ──
    ok = run([
        python, str(TOOLS_DIR / "7_note_duration_fix.py"),
        "--input-midi", str(p_pos_fixed_midi),
        "--input-audio", str(input_audio),
        "--output-midi", str(p_dur_fixed_midi),
        "--report", str(p_dur_report),
        "--threshold", str(tolerance_sec)
    ], "Note duration correction")
    if not ok:
        return None

    # ── Step 4: Fix note pitches ──
    ok = run([
        python, str(TOOLS_DIR / "8_note_pitch_fix.py"),
        "--input-midi", str(p_dur_fixed_midi),
        "--input-audio", str(input_audio),
        "--output-midi", str(p_pitch_fixed_midi),
        "--report", str(p_pitch_report),
        "--tolerance-cents", str(50)
    ], "Note pitch correction")
    if not ok:
        return None

    # ── Step 5: Fix note velocities ──
    ok = run([
        python, str(TOOLS_DIR / "9_note_velocity_fix.py"),
        "--input-midi", str(p_pitch_fixed_midi),
        "--input-audio", str(input_audio),
        "--output-midi", str(p_vel_fixed_midi),
        "--report", str(p_vel_report)
    ], "Note velocity correction")
    if not ok:
        return None

    # ── Step 6: Align to timing grid ──
    ok = run([
        python, str(TOOLS_DIR / "10_grid_alignment.py"),
        "--input-midi", str(p_vel_fixed_midi),
        "--input-audio", str(input_audio),
        "--output-midi", str(p_grid_midi),
        "--report", str(p_grid_report),
        "--grid-divisions", str(grid_divisions),
    ], "Timing grid alignment")
    if not ok:
        return None

    # ── Step 7: Correct rhythm/gaps ──
    ok = run([
        python, str(TOOLS_DIR / "12_rhythm_gap_correction.py"),
        "--input-midi", str(p_grid_midi),
        "--input-audio", str(input_audio),
        "--output-midi", str(p_rhythm_midi),
        "--report", str(p_rhythm_report),
        "--bpm", str(bpm),
        "--threshold", str(gap_threshold_sec),
        "--max-correction", str(max_correction_sec)
    ], "Rhythm/gap correction")
    if not ok:
        return None

    # ── Step 8: Render MIDI → piano audio ──
    ok = run([
        python, str(TOOLS_DIR / "3_midi_render.py"),
        "--input-midi", str(p_rhythm_midi),
        "--output-wav", str(p_rendered),
        "--soundfont", str(soundfont),
        "--sample-rate", str(44100),
    ], "MIDI → piano audio render")
    if not ok:
        return None

    # ── Step 9: Spectral match ──
    ok = run([
        python, str(TOOLS_DIR / "4_spectral_match.py"),
        "--input-rendered", str(p_rendered),
        "--input-original", str(input_audio),
        "--output-wav", str(p_spectral_matched),
    ], "Spectral match to original")
    if not ok:
        return None

    # ── Step 10: Reverb ──
    ok = run([
        python, str(TOOLS_DIR / "5_reverb_effects.py"),
        "--input-wav", str(p_spectral_matched),
        "--output-wav", str(p_reverb),
        "--decay", str(1.5),
        "--predelay-ms", str(40),
        "--wet", str(0.25)
    ], "Convolution reverb")
    if not ok:
        return None

    # ── Step 11: Final verification ──
    ok = run([
        python, str(TOOLS_DIR / "13_verify_rendered.py"),
        "--input-rendered", str(p_reverb),
        "--input-original", str(input_audio),
        "--report", str(p_verify)
    ], "Final verification vs original")
    if not ok:
        return None

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {out}")
    print(f"Final audio:      {p_reverb}")
    print(f"Verification:     {p_verify}")

    # Load verification and print key metrics
    with open(p_verify) as f:
        verify = json.load(f)
    print(f"\nDuration match:   {verify.get('duration_match', 'N/A')}")
    print(f"Waveform corr:    {verify.get('waveform_correlation', 'N/A')}")
    print(f"All bands 5%:     {verify.get('all_bands_5pct', 'N/A')}")
    print(f"Watermark removed:{verify.get('watermark_removed', 'N/A')}")
    print(f"OVERALL:          {'PASS' if verify.get('overall_pass') else 'FAIL'}")

    return {
        "output_dir": str(out),
        "final_audio": str(p_reverb),
        "verification_report": str(p_verify),
        "all_reports": {
            "transcribed": str(p_transcribed_report),
            "position": str(p_pos_report),
            "duration": str(p_dur_report),
            "pitch": str(p_pitch_report),
            "velocity": str(p_vel_report),
            "grid": str(p_grid_report),
            "rhythm": str(p_rhythm_report),
            "render": str(p_rendered_report),
            "spectral": str(p_spectral_report),
            "reverb": str(p_reverb_report),
            "verify": str(p_verify)
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full Suno watermark removal pipeline')
    parser.add_argument('--input-audio', required=True, help='Original audio WAV path')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--soundfont', required=True,
                        help='Path to piano SoundFont (.sf2)',
                        default='/Users/charlie_work/suno_pipeline_output/midi/soundfonts/SalamanderGrandPiano.sf2')
    parser.add_argument('--bpm', type=float, default=120.0, help='Tempo in BPM')
    parser.add_argument('--ticks-per-beat', type=int, default=960,
                        help='MIDI ticks per beat')
    parser.add_argument('--tolerance', type=float, default=0.08,
                        help='Onset snap tolerance in seconds')
    parser.add_argument('--grid-divisions', type=int, default=4,
                        help='Grid subdivisions per beat')
    parser.add_argument('--gap-threshold', type=float, default=0.01,
                        help='Rhythm gap threshold in seconds')
    parser.add_argument('--max-correction', type=float, default=0.1,
                        help='Max correction per note in seconds')
    args = parser.parse_args()

    result = run_pipeline(
        args.input_audio, args.output_dir, args.soundfont,
        args.bpm, args.ticks_per_beat, args.tolerance,
        args.grid_divisions, args.gap_threshold, args.max_correction
    )

    if result is None:
        print("\nPIPELINE FAILED — see errors above")
        sys.exit(1)
    print("\nAll steps completed successfully.")
