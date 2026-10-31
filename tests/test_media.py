import shutil

import numpy as np
import pytest

from conftest import png_gray
from feedback_collector import media
from feedback_collector.changes import SAMPLE_FPS, SAMPLE_WIDTH, detect_changes
from feedback_collector.media import MediaError, extract_frames, gray_samples, probe, scaled_size


def test_probe_reads_size_duration_and_audio(box_clip):
    info = probe(box_clip)
    assert (info.width, info.height, info.has_audio) == (320, 200, True)
    assert info.duration == pytest.approx(6.0, abs=0.1)
    assert info.last_frame_time == pytest.approx(5.9, abs=0.05)


def test_probe_without_audio(silent_clip):
    assert not probe(silent_clip).has_audio


def test_probe_falls_back_to_last_frame_without_container_duration(streamed_clip):
    info = probe(streamed_clip)
    assert info.duration == pytest.approx(info.last_frame_time)
    assert info.duration == pytest.approx(5.9, abs=0.05)


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(MediaError, match="does not exist"):
        probe(tmp_path / "nope.webm")


@pytest.mark.parametrize(
    "size, max_width, expected",
    [
        ((2560, 1600), 1600, (1600, 1000)),
        ((1920, 1080), 1600, (1600, 900)),
        ((1365, 767), 1600, (1364, 766)),
        ((320, 200), 1600, (320, 200)),
    ],
)
def test_scaled_size_is_even_and_never_upscales(size, max_width, expected):
    assert scaled_size(*size, max_width) == expected


def test_gray_samples_shape(box_clip):
    samples = gray_samples(box_clip, probe(box_clip), fps=2.0, width=160)
    assert samples.dtype == np.uint8
    assert samples.shape[1:] == (100, 160)
    assert 11 <= samples.shape[0] <= 13


def test_box_clip_has_one_change_near_three_seconds(box_clip):
    events = detect_changes(gray_samples(box_clip, probe(box_clip), fps=SAMPLE_FPS, width=SAMPLE_WIDTH))
    assert len(events) == 1
    assert events[0].onset == pytest.approx(3.0, abs=0.5)


def test_extract_frames_uses_one_ffmpeg_process(box_clip, tmp_path, monkeypatch):
    info = probe(box_clip)
    calls, real_run = [], media._run
    monkeypatch.setattr(media, "_run", lambda args: calls.append(args[0]) or real_run(args))
    targets = [(t, tmp_path / f"{t}.png") for t in (1.0, 3.5, 5.0)]
    extract_frames(box_clip, info, targets, width=160)
    assert calls.count("ffmpeg") == 1
    assert all(path.exists() for _, path in targets)


def test_extract_frames_shows_the_right_moment(box_clip, tmp_path):
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    extract_frames(box_clip, probe(box_clip), [(1.0, before), (3.5, after)], width=160)
    assert png_gray(before, 160, 100)[40:60, 70:90].mean() > 200
    assert png_gray(after, 160, 100)[40:60, 70:90].mean() < 60


def test_extract_frames_clamps_past_end(box_clip, tmp_path):
    late = tmp_path / "late.png"
    extract_frames(box_clip, probe(box_clip), [(99.0, late)], width=160)
    assert late.exists()


def test_awkward_paths_work(box_clip, tmp_path):
    video = tmp_path / "rec 09:08:00 50%.webm"
    shutil.copy(box_clip, video)
    destination = tmp_path / "out %d dir" / "frame.png"
    info = probe(video)
    assert gray_samples(video, info, fps=2.0, width=160).shape[0] > 0
    extract_frames(video, info, [(1.0, destination)], width=160)
    assert destination.exists()
