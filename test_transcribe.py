#!/usr/bin/env python3
"""Tests for transcribe.py"""

import os
import urllib.request
from pathlib import Path
import pytest
from transcribe import (
    Segment,
    format_timestamp_srt,
    format_timestamp_vtt,
    segments_to_srt,
    segments_to_vtt,
)

PROJECT_DIR = Path(__file__).parent

# "Les licences Creative Commons" — French, CC-BY-SA 3.0, ~9 MB
# https://archive.org/details/LesLicencesCreativeCommons
TEST_VIDEO_URL = (
    "https://archive.org/download/LesLicencesCreativeCommons/"
    "Les%20licences%20Creative%20Commons-x1tg4gv.ogv"
)
TEST_VIDEO_PATH = PROJECT_DIR / "test_video.ogv"


@pytest.fixture(scope="session")
def test_video():
    """Return path to test video, downloading it first if not present."""
    if not TEST_VIDEO_PATH.exists():
        print(f"\nDownloading test video from Internet Archive (~9 MB)...")
        urllib.request.urlretrieve(TEST_VIDEO_URL, str(TEST_VIDEO_PATH))
        print(f"Saved to {TEST_VIDEO_PATH}")
    return str(TEST_VIDEO_PATH)


class TestFormatTimestampSrt:
    def test_zero(self):
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_seconds_and_milliseconds(self):
        assert format_timestamp_srt(5.123) == "00:00:05,123"

    def test_minutes(self):
        assert format_timestamp_srt(65.0) == "00:01:05,000"

    def test_hours(self):
        assert format_timestamp_srt(3661.5) == "01:01:01,500"

    def test_large_values(self):
        assert format_timestamp_srt(36000.999) == "10:00:00,999"


class TestFormatTimestampVtt:
    def test_uses_dot_not_comma(self):
        assert format_timestamp_vtt(5.123) == "00:00:05.123"

    def test_zero(self):
        assert format_timestamp_vtt(0.0) == "00:00:00.000"


class TestSegmentsToSrt:
    def test_single_segment(self):
        segments = [Segment(0.0, 5.0, "Hello world")]
        result = segments_to_srt(segments)
        assert "1\n00:00:00,000 --> 00:00:05,000\nHello world" in result

    def test_multiple_segments(self):
        segments = [
            Segment(0.0, 2.0, "First"),
            Segment(2.5, 5.0, "Second"),
        ]
        result = segments_to_srt(segments)
        assert "1\n" in result
        assert "2\n" in result

    def test_offset(self):
        segments = [Segment(0.0, 5.0, "Test")]
        result = segments_to_srt(segments, offset_seconds=60.0)
        assert "00:01:00,000 --> 00:01:05,000" in result

    def test_sequence_renumbering(self):
        segments = [Segment(0.0, 2.0, "A"), Segment(2.0, 4.0, "B")]
        result = segments_to_srt(segments)
        lines = result.strip().split("\n")
        assert lines[0] == "1"
        assert lines[4] == "2"


class TestSegmentsToVtt:
    def test_includes_header(self):
        segments = [Segment(0.0, 5.0, "Test")]
        result = segments_to_vtt(segments)
        assert result.startswith("WEBVTT")

    def test_uses_dot_separator(self):
        segments = [Segment(0.0, 5.0, "Test")]
        result = segments_to_vtt(segments)
        assert "00:00:00.000 --> 00:00:05.000" in result

    def test_offset(self):
        segments = [Segment(0.0, 5.0, "Test")]
        result = segments_to_vtt(segments, offset_seconds=3600.0)
        assert "01:00:00.000 --> 01:00:05.000" in result


class TestAudioExtraction:
    def test_extract_audio(self, test_video, tmp_path):
        from transcribe import extract_audio

        audio_path = str(tmp_path / "audio.mp3")
        extract_audio(test_video, audio_path)

        assert os.path.isfile(audio_path)
        assert os.path.getsize(audio_path) > 0

    def test_split_audio_returns_chunks(self, test_video, tmp_path):
        from transcribe import extract_audio, split_audio

        audio_path = str(tmp_path / "audio.mp3")
        extract_audio(test_video, audio_path)

        chunks = split_audio(audio_path, str(tmp_path))

        assert len(chunks) >= 1
        for chunk in chunks:
            assert os.path.isfile(chunk)


class TestIntegration:
    @pytest.mark.integration
    def test_transcribe_text_format(self, test_video, tmp_path, monkeypatch):
        import sys
        from transcribe import main

        output_path = str(tmp_path / "output.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcribe.py",
                test_video,
                "--format",
                "text",
                "--output",
                output_path,
                "--translate-to",
                "none",
            ],
        )
        main()

        assert os.path.isfile(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "[Language:" in content
        assert "=== Original Transcript ===" in content

    @pytest.mark.integration
    def test_transcribe_srt_format(self, test_video, tmp_path, monkeypatch):
        import sys
        from transcribe import main

        output_path = str(tmp_path / "output.srt")
        monkeypatch.setattr(
            sys,
            "argv",
            ["transcribe.py", test_video, "--format", "srt", "--output", output_path],
        )
        main()

        assert os.path.isfile(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "-->" in content

    @pytest.mark.integration
    def test_translate_to_english(self, test_video, tmp_path, monkeypatch):
        """French video should produce an English translation section."""
        import sys
        from transcribe import main

        output_path = str(tmp_path / "output.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            ["transcribe.py", test_video, "--format", "text", "--output", output_path],
        )
        main()

        assert os.path.isfile(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "[Language: french]" in content or "[Language: fr]" in content
        assert "=== Translation (en) ===" in content

    @pytest.mark.integration
    def test_translate_only(self, test_video, tmp_path, monkeypatch):
        import sys
        from transcribe import main

        output_path = str(tmp_path / "translation.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            ["transcribe.py", test_video, "--translate-only", "--output", output_path],
        )
        main()

        assert os.path.isfile(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "=== Translation (en) ===" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
