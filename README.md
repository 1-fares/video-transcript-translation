# video-transcript-translation

Transcribe video/audio files using OpenAI's Whisper API, with automatic language detection and English translation.

## Commands

```bash
# Run the transcriber
./run.sh <video_file> [--language LANG] [--format text|srt|vtt|json|verbose_json] [--output PATH] [--no-translate] [--translate-only]

# Or directly with uv
uv run transcribe.py <video_file> [options]

# Run unit tests only (no API calls)
uv run pytest test_transcribe.py -v

# Run a single test class
uv run pytest test_transcribe.py::TestSegmentsToSrt -v

# Run integration tests (calls OpenAI Whisper API, costs money)
uv run pytest test_transcribe.py -v -m integration
```

**External dependencies:** `ffmpeg` and `ffprobe` must be on PATH.

## Architecture

Everything lives in a single script (`transcribe.py`). The pipeline is:

1. **Audio extraction** (`extract_audio`): ffmpeg strips audio from video → temp MP3
2. **Chunking** (`split_audio`): if audio exceeds 25MB Whisper limit, splits into chunks; chunk duration is dynamically adjusted based on file size ratio
3. **Transcription** (`transcribe_chunk`): calls `client.audio.transcriptions` with `verbose_json` for all output formats (single API call per chunk); converts segments to SRT/VTT locally
4. **Translation** (`translate_chunk`): calls `client.audio.translations` to get English text; only runs when detected language is non-English and format is `text`
5. **Output**: stitches chunks together with correct timestamp offsets, writes to file

Key design decisions:
- Always uses `verbose_json` internally regardless of requested output format, then converts locally — this enables language detection and correct multi-chunk timestamp offsets in a single API call
- Translation only applies to `text` format (not SRT/VTT)
- `--translate-only` skips transcription entirely and calls the translation endpoint directly

## API Key

Set `OPENAI_API_KEY` in a `.env` file in the project root, or export it as an environment variable. The `.env` file is gitignored.

## Tests

- Unit tests (no I/O or API): timestamp formatting and SRT/VTT segment serialization
- Audio extraction tests: use `test_video.ogv` — a French CC-BY-SA video auto-downloaded from Internet Archive on first run (~9 MB, gitignored)
- Integration tests (`@pytest.mark.integration`): full end-to-end with live Whisper API calls; the French source video also exercises the auto-translation path
