import dataclasses
import json
import sys
from pathlib import Path

import pytest

from feedback_collector.models import FrameRef, Report, Section, Segment, Transcript
from feedback_collector.report import handoff_prompt, render_json, render_markdown, render_vtt


@pytest.fixture
def report():
    sections = (
        Section(0.0, 27.0, (Segment(12.4, 27.0, "So this button is confusing."),)),
        Section(27.0, 41.0),
        Section(41.0, 60.0),
        Section(60.0, 402.1, (Segment(60.0, 70.0, "Next issue."),)),
    )
    frames = (
        FrameRef(12.4, "speech", "speech start", "frames/00-00-12.4_speech.png"),
        FrameRef(15.1, "pointer", '"this"', "frames/00-00-15.1_pointer.png"),
        FrameRef(30.2, "silent-change", "screen changed", "frames/00-00-30.2_silent-change.png"),
    )
    transcript = Transcript(tuple(s for section in sections for s in section.segments), "en", "large-v3-turbo", "cuda")
    return Report(
        video=Path("/videos/rec.webm"),
        out_dir=Path("/videos/rec-feedback"),
        duration=402.1,
        width=2560,
        height=1600,
        transcript=transcript,
        sections=sections,
        frames=frames,
        candidate_count=57,
        cap=40,
    )


def test_header_counts_and_reader_notes(report):
    text = render_markdown(report)
    assert text.startswith("# Feedback walkthrough: rec.webm\n")
    assert (
        "`6:42` · `3 sections` · `3 screenshots (cap 40) of 57 candidates` · "
        "transcript: faster-whisper large-v3-turbo (cuda), **not reviewed by a human**"
    ) in text
    assert (
        f'Run `"{sys.executable}" -m feedback_collector frames "/videos/rec.webm" '
        '--at 03:12.5 --out "/videos/rec-feedback/extra"`.'
    ) in text


def test_header_counts_in_the_singular(report):
    single = dataclasses.replace(report, frames=report.frames[:1], candidate_count=1)
    assert "`1 screenshot (cap 40) of 1 candidate`" in render_markdown(single)


def test_sections_interleave_quotes_and_frames(report):
    assert (
        "## 00:00–00:27\n> So this button is confusing.\n\n"
        "![00:12.4 speech start](frames/00-00-12.4_speech.png)\n"
        '![00:15.1 "this"](frames/00-00-15.1_pointer.png)'
    ) in render_markdown(report)


def test_silent_sections_only_appear_with_frames(report):
    text = render_markdown(report)
    assert "## 00:27–00:41 (no narration)\n![00:30.2 screen changed](frames/00-00-30.2_silent-change.png)" in text
    assert "00:41–01:00" not in text


def test_no_speech_note(report):
    silent = dataclasses.replace(report, sections=(Section(0.0, 402.1),), frames=report.frames[2:])
    assert "_No speech was detected in this recording._" in render_markdown(silent)


def test_json_mirrors_report(report):
    data = json.loads(render_json(report))
    assert data["frames"] == {"selected": 3, "candidates": 57, "cap": 40}
    assert data["transcription"]["human_reviewed"] is False
    assert [s["narrated"] for s in data["sections"]] == [True, False, True]
    assert data["sections"][0]["frames"][1]["label"] == '"this"'


def test_vtt():
    assert render_vtt([Segment(12.4, 15.0, "Hello.")]) == "WEBVTT\n\n00:00:12.400 --> 00:00:15.000\nHello.\n\n"


def test_handoff_prompt_quotes_the_path():
    assert handoff_prompt(Path("/x/report.md")).startswith('Read "/x/report.md" and open every screenshot it lists.')
