#!/usr/bin/env python3
"""
Transcribe video/audio files using OpenAI's Whisper API.
Auto-detects language and includes an English translation if the audio is not in English.

Usage:
    python transcribe.py <file>
    python transcribe.py <file> --language fr --format srt
    python transcribe.py <file> --output subtitles.vtt --format vtt
    python transcribe.py <file> --no-translate
    python transcribe.py <file> --translate-only
    python transcribe.py <file> --chunk-minutes 5

Requires:
    pip install openai
    ffmpeg/ffprobe on PATH
    OPENAI_API_KEY in .env or environment
"""

import argparse
import sys
import os
import subprocess
import tempfile
import time
import json
import math
from collections import namedtuple
from pathlib import Path
from functools import wraps

from openai import OpenAI, APIStatusError

Segment = namedtuple("Segment", ["start", "end", "text"])

# 25 MB limit for Whisper API
MAX_FILE_SIZE = 25 * 1024 * 1024


def _load_dotenv(path: str | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if missing)."""
    if path is None:
        path = str(Path(__file__).parent / ".env")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
    except FileNotFoundError:
        pass


def retry_on_failure(max_retries=3, delay=2, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except APIStatusError as e:
                    if e.status_code < 500:
                        raise  # client errors (4xx) won't succeed on retry
                    retries += 1
                    if retries == max_retries:
                        raise
                    print(f"  Retry {retries}/{max_retries} after server error: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise
                    print(f"  Retry {retries}/{max_retries} after error: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def segments_to_srt(segments: list, offset_seconds: float = 0.0) -> str:
    lines = []
    for seq, seg in enumerate(segments, start=1):
        start = seg.start + offset_seconds
        end = seg.end + offset_seconds
        lines.append(str(seq))
        lines.append(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments: list, offset_seconds: float = 0.0) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = seg.start + offset_seconds
        end = seg.end + offset_seconds
        lines.append(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def extract_audio(input_path: str, audio_path: str) -> None:
    """Extract audio from video as mp3 using ffmpeg."""
    print(f"Extracting audio from {input_path}...")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            "-y",
            audio_path,
        ],
        check=True,
        capture_output=True,
    )


def split_audio(audio_path: str, chunk_dir: str, chunk_minutes: int = 10) -> list[str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    chunk_seconds = chunk_minutes * 60
    file_size = os.path.getsize(audio_path)

    if file_size <= MAX_FILE_SIZE and duration <= chunk_seconds:
        size_mb = file_size / 1024 / 1024
        print(f"Audio: {duration:.0f}s, {size_mb:.1f}MB (single chunk)")
        return [audio_path]

    size_ratio = file_size / MAX_FILE_SIZE
    adjusted_chunk_seconds = min(chunk_seconds, int(chunk_seconds / size_ratio * 0.9))
    total_chunks = math.ceil(duration / adjusted_chunk_seconds)
    print(
        f"Splitting into {total_chunks} chunks (~{adjusted_chunk_seconds}s each, {duration / 60:.1f} min total)..."
    )

    chunks = []
    start = 0.0
    i = 0
    while start < duration:
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")
        print(f"  Writing chunk {i + 1}/{total_chunks}...", end="\r", flush=True)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_path,
                "-ss",
                str(start),
                "-t",
                str(adjusted_chunk_seconds),
                "-acodec",
                "libmp3lame",
                "-q:a",
                "4",
                "-y",
                chunk_path,
            ],
            check=True,
            capture_output=True,
        )
        chunks.append(chunk_path)
        start += adjusted_chunk_seconds
        i += 1

    print()  # newline after \r progress
    return chunks


@retry_on_failure(max_retries=3)
def transcribe_chunk(
    client: OpenAI,
    audio_path: str,
    language: str | None = None,
) -> dict:
    """Transcribe a single audio chunk.
    Returns text, language, segments (as Segment namedtuples), and duration.
    """
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            **({"language": language} if language else {}),
        )

    segments = [Segment(start=s.start, end=s.end, text=s.text) for s in (resp.segments or [])]
    duration = resp.duration if resp.duration else (segments[-1].end if segments else 0)

    return {
        "text": resp.text,
        "language": resp.language,
        "segments": segments,
        "duration": duration,
    }


@retry_on_failure(max_retries=3)
def translate_chunk(client: OpenAI, audio_path: str) -> str:
    """Translate a single audio chunk to English."""
    with open(audio_path, "rb") as f:
        response = client.audio.translations.create(model="whisper-1", file=f)
    return response.text


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with OpenAI Whisper API"
    )
    parser.add_argument("file", help="Input audio or video file")
    parser.add_argument(
        "--language", "-l", help="Language code hint (e.g. en, fr, de, ar)"
    )
    parser.add_argument(
        "--format",
        "-f",
        default="text",
        choices=["text", "srt", "vtt", "json", "verbose_json"],
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o", help="Output file path (auto-generated if omitted)"
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip English translation (text format only)",
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Only translate to English, skip original transcription",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=10,
        metavar="N",
        help="Max chunk duration in minutes when splitting large files (default: 10)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"File not found: {args.file}")

    if args.no_translate and args.format != "text":
        print(
            f"Warning: --no-translate has no effect with --format {args.format} "
            "(translation is only available for text format)"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in .env or as an environment variable")

    client = OpenAI(api_key=api_key)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        ext_map = {
            "text": "txt",
            "srt": "srt",
            "vtt": "vtt",
            "json": "json",
            "verbose_json": "json",
        }
        ext = ext_map.get(args.format, "txt")
        output_path = str(Path(args.file).with_suffix(f".{ext}"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(args.file, audio_path)
        chunks = split_audio(audio_path, tmp_dir, args.chunk_minutes)

        full_transcript = ""
        detected_language = None

        if args.translate_only:
            print("Translate-only mode: skipping transcription...")
            translations = []
            for i, chunk in enumerate(chunks):
                print(
                    f"  Translating chunk {i + 1}/{len(chunks)}...", end="", flush=True
                )
                translations.append(translate_chunk(client, chunk))
                print(" done")
            full_transcript = "\n".join(translations).strip()
            print(f"Translation complete: {len(full_transcript)} chars")
        else:
            all_segments: list[Segment] = []
            all_texts = []
            cumulative_duration = 0.0
            for i, chunk in enumerate(chunks):
                print(
                    f"  Transcribing chunk {i + 1}/{len(chunks)}...", end="", flush=True
                )
                result = transcribe_chunk(client, chunk, args.language)

                for seg in result["segments"]:
                    all_segments.append(
                        Segment(
                            start=seg.start + cumulative_duration,
                            end=seg.end + cumulative_duration,
                            text=seg.text,
                        )
                    )

                all_texts.append(result["text"])
                if detected_language is None:
                    detected_language = result["language"]
                cumulative_duration += result["duration"]
                print(" done")

            print(f"Detected language: {detected_language}")

            if args.format == "srt":
                full_transcript = segments_to_srt(all_segments)
            elif args.format == "vtt":
                full_transcript = segments_to_vtt(all_segments)
            elif args.format == "json":
                full_transcript = json.dumps({
                    "text": "\n".join(all_texts).strip(),
                    "language": detected_language,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in all_segments
                    ],
                })
            elif args.format == "verbose_json":
                full_transcript = json.dumps({
                    "text": "\n".join(all_texts).strip(),
                    "language": detected_language,
                    "duration": cumulative_duration,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in all_segments
                    ],
                })
            else:  # text
                full_transcript = "\n".join(all_texts).strip()

        is_english = (
            detected_language is not None
            and detected_language.lower().startswith("en")
        )

        translation = None
        if (
            not args.translate_only
            and not is_english
            and not args.no_translate
            and args.format == "text"
        ):
            print("Translating to English...")
            translations = []
            for i, chunk in enumerate(chunks):
                print(
                    f"  Translating chunk {i + 1}/{len(chunks)}...", end="", flush=True
                )
                translations.append(translate_chunk(client, chunk))
                print(" done")
            translation = "\n".join(translations).strip()

        with open(output_path, "w", encoding="utf-8") as f:
            if args.format == "text":
                if args.translate_only:
                    f.write("=== English Translation ===\n\n")
                    f.write(full_transcript)
                else:
                    f.write(f"[Language: {detected_language}]\n\n")
                    f.write("=== Original Transcript ===\n\n")
                    f.write(full_transcript)
                    if translation:
                        f.write("\n\n=== English Translation ===\n\n")
                        f.write(translation)
                f.write("\n")
            else:
                f.write(full_transcript)
                f.write("\n")

        print(f"\nOutput written to: {output_path} ({len(full_transcript)} chars)")


if __name__ == "__main__":
    main()
