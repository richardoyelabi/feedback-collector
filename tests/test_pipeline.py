import json
import re

import pytest

from conftest import NARRATION, FakeTranscriber
from feedback_collector.media import MediaError
from feedback_collector.pipeline import Options, OutputExistsError, grab_frames, process


def test_process_writes_interleaved_report(box_clip, tmp_path):
    out = tmp_path / "out"
    report = process(box_clip, Options(out_dir=out), FakeTranscriber(NARRATION))
    assert [(round(f.time, 1), f.reason) for f in report.frames] == [
        (0.2, "speech"),
        (3.5, "silent-change"),
        (5.0, "speech"),
    ]
    text = (out / "report.md").read_text()
    assert "> This page loads fine." in text
    assert "(no narration)" in text
    assert all((out / path).exists() for path in re.findall(r"\]\((frames/[^)]+)\)", text))
    assert json.loads((out / "report.json").read_text())["frames"]["selected"] == 3
    assert (out / "transcript.vtt").read_text().startswith("WEBVTT")


def test_process_without_speech_keeps_screen_changes(box_clip, tmp_path):
    report = process(box_clip, Options(out_dir=tmp_path / "out"), FakeTranscriber([]))
    assert [f.reason for f in report.frames] == ["silent-change"]
    assert "_No speech was detected in this recording._" in (tmp_path / "out" / "report.md").read_text()


def test_process_rejects_video_without_audio(silent_clip, tmp_path):
    with pytest.raises(MediaError, match="no audio track"):
        process(silent_clip, Options(out_dir=tmp_path / "out"), FakeTranscriber([]))


def test_existing_report_is_refused(box_clip, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "report.md").write_text("mine")
    with pytest.raises(OutputExistsError, match="--force"):
        process(box_clip, Options(out_dir=out), FakeTranscriber(NARRATION))


def test_force_replaces_only_own_files(box_clip, tmp_path):
    out = tmp_path / "out"
    (out / "frames").mkdir(parents=True)
    (out / "report.md").write_text("old")
    (out / "notes.txt").write_text("keep me")
    (out / "frames" / "stale.png").write_bytes(b"x")
    process(box_clip, Options(out_dir=out, force=True), FakeTranscriber(NARRATION))
    assert (out / "notes.txt").read_text() == "keep me"
    assert not (out / "frames" / "stale.png").exists()
    assert (out / "report.md").read_text() != "old"


def test_grab_frames_names_files_by_time(box_clip, tmp_path):
    paths = grab_frames(box_clip, [1.0, 3.5], tmp_path / "extra", width=160)
    assert [p.name for p in paths] == ["00-00-01.0_extra.png", "00-00-03.5_extra.png"]
    assert all(p.exists() for p in paths)
