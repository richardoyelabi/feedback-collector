"""Render a processed recording as Markdown, JSON and WebVTT."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from feedback_collector.models import FrameRef, Report, Section, Segment
from feedback_collector.sections import section_index_at
from feedback_collector.timefmt import clock, clock_precise, duration_label, vtt_time


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def visible_sections(report: Report) -> list[tuple[Section, list[FrameRef]]]:
    grouped: list[list[FrameRef]] = [[] for _ in report.sections]
    for frame in report.frames:
        grouped[section_index_at(report.sections, frame.time)].append(frame)
    return [
        (section, frames)
        for section, frames in zip(report.sections, grouped)
        if section.narrated or frames
    ]


def _header(report: Report, section_count: int) -> str:
    transcript = report.transcript
    stats = " · ".join([
        f"`{duration_label(report.duration)}`",
        f"`{plural(section_count, 'section')}`",
        f"`{plural(len(report.frames), 'screenshot')} (cap {report.cap}) of {plural(report.candidate_count, 'candidate')}`",
        f"transcript: faster-whisper {transcript.model} ({transcript.device}), **not reviewed by a human**",
    ])
    # Claude runs this from the web app's repo, where a venv-installed `feedback-collector` is not on PATH.
    hint = (
        f'"{sys.executable}" -m feedback_collector frames "{report.video}" '
        f'--at 03:12.5 --out "{report.out_dir / "extra"}"'
    )
    return "\n".join([
        f"# Feedback walkthrough: {report.video.name}",
        stats,
        "",
        "**How to read this:** sections are in recording order. "
        "Open every screenshot (paths are relative to this file) before acting.",
        "Narration is speech-to-text and may mishear names; "
        "where words and screenshots disagree, trust the screenshots.",
        f"Need another moment? Run `{hint}`.",
    ])


def _section_block(section: Section, frames: Sequence[FrameRef]) -> str:
    heading = f"## {clock(section.start)}–{clock(section.end)}"
    if not section.narrated:
        heading += " (no narration)"
    block = f"{heading}\n> {section.text}" if section.narrated else heading
    if frames:
        images = "\n".join(f"![{clock_precise(f.time)} {f.label}]({f.path})" for f in frames)
        block += ("\n\n" if section.narrated else "\n") + images
    return block


def render_markdown(report: Report) -> str:
    visible = visible_sections(report)
    blocks = [_header(report, len(visible))]
    if not any(section.narrated for section, _ in visible):
        blocks.append("_No speech was detected in this recording._")
    blocks.extend(_section_block(section, frames) for section, frames in visible)
    return "\n\n".join(blocks) + "\n"


def _seconds(value: float) -> float:
    return round(value, 3)


def render_json(report: Report) -> str:
    transcript = report.transcript
    data = {
        "source": {
            "path": str(report.video),
            "duration": _seconds(report.duration),
            "width": report.width,
            "height": report.height,
        },
        "transcription": {
            "engine": "faster-whisper",
            "model": transcript.model,
            "device": transcript.device,
            "language": transcript.language,
            "human_reviewed": False,
        },
        "frames": {"selected": len(report.frames), "candidates": report.candidate_count, "cap": report.cap},
        "sections": [
            {
                "start": _seconds(section.start),
                "end": _seconds(section.end),
                "narrated": section.narrated,
                "text": section.text,
                "segments": [
                    {"start": _seconds(s.start), "end": _seconds(s.end), "text": s.text}
                    for s in section.segments
                ],
                "frames": [
                    {"time": _seconds(f.time), "reason": f.reason, "label": f.label, "path": f.path}
                    for f in frames
                ],
            }
            for section, frames in visible_sections(report)
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def render_vtt(segments: Sequence[Segment]) -> str:
    cues = "".join(f"{vtt_time(s.start)} --> {vtt_time(s.end)}\n{s.text}\n\n" for s in segments)
    return "WEBVTT\n\n" + cues


def handoff_prompt(report_path: Path) -> str:
    return (
        f'Read "{report_path}" and open every screenshot it lists. '
        "It is narrated feedback I recorded while testing this app. "
        "For each issue I raise, find the code responsible and propose a fix. "
        "Where my words and the screenshots disagree, trust the screenshots."
    )
