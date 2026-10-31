import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from feedback_collector.models import Segment, Transcript, Word

NARRATION = [
    Segment(0.2, 1.6, "This page loads fine.",
            (Word(0.2, 0.4, "This"), Word(0.4, 0.7, "page"), Word(0.7, 1.1, "loads"), Word(1.1, 1.6, "fine."))),
    Segment(5.0, 5.8, "Now the box shows."),
]


class FakeTranscriber:
    def __init__(self, segments):
        self.segments = tuple(segments)

    def transcribe(self, video: Path, offset: float = 0.0) -> Transcript:
        return Transcript(self.segments, "en", "fake", "cpu")


BOX_VIDEO = (
    "color=c=white:s=320x200:r=10:d=6,"
    "drawbox=x=140:y=80:w=40:h=40:color=black:t=fill:enable='gte(t,3)'"
)
TONE = "sine=frequency=440:duration=6"

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _ffmpeg(*args: str) -> bytes:
    return subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, capture_output=True).stdout


@pytest.fixture(scope="session")
def clips_dir(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    return tmp_path_factory.mktemp("clips")


@pytest.fixture(scope="session")
def box_clip(clips_dir) -> Path:
    path = clips_dir / "box.webm"
    _ffmpeg("-f", "lavfi", "-i", BOX_VIDEO, "-f", "lavfi", "-i", TONE,
            "-c:v", "libvpx", "-b:v", "300k", "-c:a", "libopus", "-shortest", str(path))
    return path


@pytest.fixture(scope="session")
def silent_clip(clips_dir) -> Path:
    path = clips_dir / "silent.webm"
    _ffmpeg("-f", "lavfi", "-i", BOX_VIDEO, "-c:v", "libvpx", "-b:v", "300k", "-an", str(path))
    return path


@pytest.fixture(scope="session")
def streamed_clip(clips_dir) -> Path:
    path = clips_dir / "streamed.webm"
    data = _ffmpeg("-f", "lavfi", "-i", BOX_VIDEO, "-f", "lavfi", "-i", TONE,
                   "-c:v", "libvpx", "-b:v", "300k", "-c:a", "libopus", "-shortest", "-f", "webm", "pipe:1")
    path.write_bytes(data)
    return path


def png_gray(path: Path, width: int, height: int) -> np.ndarray:
    raw = _ffmpeg("-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
