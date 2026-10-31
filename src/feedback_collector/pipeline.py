"""Turn one recording into a report folder."""

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from feedback_collector.changes import SAMPLE_FPS, SAMPLE_WIDTH, detect_changes
from feedback_collector.media import MediaError, extract_frames, gray_samples, probe
from feedback_collector.models import FrameRef, Report
from feedback_collector.report import render_json, render_markdown, render_vtt
from feedback_collector.sections import build_sections
from feedback_collector.selection import find_candidates, frame_budget, select_frames
from feedback_collector.timefmt import filename_stamp
from feedback_collector.transcription import Transcriber

OWN_FILES = ("report.md", "report.json", "transcript.vtt", "frames")

logger = logging.getLogger(__name__)


class OutputExistsError(Exception):
    pass


@dataclass(frozen=True)
class Options:
    out_dir: Path
    max_frames: int | None = None
    frame_width: int = 1600
    every: float | None = None
    force: bool = False


def default_out_dir(video: Path) -> Path:
    return video.parent / f"{video.stem}-feedback"


def _own_files(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in OWN_FILES if (out_dir / name).exists()]


def check_out_dir(out_dir: Path, force: bool) -> None:
    if _own_files(out_dir) and not force:
        raise OutputExistsError(f"{out_dir} already has a report; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)


def remove_old_report(out_dir: Path) -> None:
    for path in _own_files(out_dir):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def process(video: Path, options: Options, transcriber: Transcriber) -> Report:
    info = probe(video)
    if not info.has_audio:
        raise MediaError(
            f"{video.name} has no audio track, so there is no narration to transcribe. "
            "Record with the microphone on."
        )
    out_dir = options.out_dir
    check_out_dir(out_dir, options.force)

    logger.info("Transcribing %s…", video.name)
    transcript = transcriber.transcribe(video, offset=info.audio_offset)
    if not transcript.segments:
        logger.warning("No speech detected; the report will only contain screen changes")

    logger.info("Looking for screen changes…")
    changes = detect_changes(gray_samples(video, info, SAMPLE_FPS, SAMPLE_WIDTH))
    sections = build_sections(transcript.segments, info.duration)
    candidates = find_candidates(sections, changes, info.duration, options.every)
    onsets = [change.onset for change in changes]
    cap = frame_budget(info.duration, options.max_frames)
    chosen = select_frames(candidates, onsets, cap)
    candidate_count = len(select_frames(candidates, onsets, None))

    frames: list[FrameRef] = []
    for candidate in chosen:
        time = min(max(candidate.time, 0.0), info.last_frame_time)
        path = f"frames/{filename_stamp(time)}_{candidate.reason}.png"
        if all(frame.path != path for frame in frames):
            frames.append(FrameRef(time, candidate.reason, candidate.label, path))

    # Only now, with the slow and failure-prone steps done, does an old report make way for the new one.
    remove_old_report(out_dir)
    logger.info("Extracting %d screenshots…", len(frames))
    extract_frames(video, info, [(frame.time, out_dir / frame.path) for frame in frames], options.frame_width)

    report = Report(
        video=video,
        out_dir=out_dir,
        duration=info.duration,
        width=info.width,
        height=info.height,
        transcript=transcript,
        sections=tuple(sections),
        frames=tuple(frames),
        candidate_count=candidate_count,
        cap=cap,
    )
    (out_dir / "transcript.vtt").write_text(render_vtt(transcript.segments), encoding="utf-8")
    (out_dir / "report.json").write_text(render_json(report), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def grab_frames(video: Path, times: Sequence[float], out_dir: Path, width: int) -> list[Path]:
    info = probe(video)
    paths = [out_dir / f"{filename_stamp(time)}_extra.png" for time in times]
    extract_frames(video, info, list(zip(times, paths)), width)
    return paths
