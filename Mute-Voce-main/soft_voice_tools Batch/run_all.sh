#!/bin/sh
# Cross-platform script to install dependencies and run the enhancer
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install librosa soundfile numpy openai-whisper ffmpeg-python --quiet

echo "Running script..."
python3 enhance_soft_voices_full.py "$@"

