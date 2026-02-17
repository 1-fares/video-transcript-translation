#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_file> [--language LANG] [--format FORMAT] [--output PATH] [--no-translate] [--translate-only]"
    exit 1
fi

uv run transcribe.py "$@"
