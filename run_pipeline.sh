#!/usr/bin/env bash
set -euo pipefail

INPUT_AUDIO="${1:?Usage: ./run_pipeline.sh /path/to/original.wav [output-dir]}"
OUTPUT_DIR="${2:-output}"
SOUNDFONT="${3:-soundfonts/SalamanderGrandPiano.sf2}"

PYTHON="${PYTHON:-/opt/anaconda3/bin/python3}"

"$PYTHON" sub_problems/14_orchestrator.py \
    --input-audio "$INPUT_AUDIO" \
    --output-dir "$OUTPUT_DIR" \
    --soundfont "$SOUNDFONT"