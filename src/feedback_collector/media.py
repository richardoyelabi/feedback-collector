"""Thin wrappers around ffprobe and ffmpeg."""

import bisect
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SHOWINFO_TIME = re.compile(r"Parsed_showinfo.*\bpts_time:\s*(-?[0-9.]+)")


class MediaError(Exception):
    pass


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    video_start: float
    audio_start: float | None
    last_frame_time: float

    @property
    def has_audio(self) -> bool:
        return self.audio_start is not None

    @property
    def audio_offset(self) -> float:
        return 0.0 if self.audio_start is None else self.audio_start - self.video_start


def _run(args: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(list(args), capture_output=True)
    except FileNotFoundError as error:
        raise MediaError("ffmpeg/ffprobe not found on PATH; install ffmpeg (sudo apt install ffmpeg)") from error
    if result.returncode != 0:
        tail = "\n".join(result.stderr.decode(errors="replace").splitlines()[-20:])
        raise MediaError(f"{args[0]} failed:\n{tail}")
    return result


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(video: Path) -> MediaInfo:
    if not video.exists():
        raise MediaError(f"{video} does not exist")
    path = str(video.resolve())
    metadata = json.loads(_run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,width,height,start_time",
        "-of", "json", path,
    ]).stdout)
    streams = metadata.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise MediaError(f"{video.name} has no video stream")
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video_start = _number(video_stream.get("start_time")) or 0.0
    audio_start = None
    if audio_stream is not None:
        audio_start = _number(audio_stream.get("start_time")) or 0.0

    packets = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", path,
    ]).stdout.decode().split()
    times = [t for t in (_number(p.strip(",")) for p in packets) if t is not None]
    if not times:
        raise MediaError(f"{video.name} has no readable video frames")
    last_frame_time = max(times) - video_start

    duration = _number(metadata.get("format", {}).get("duration"))
    return MediaInfo(
        path=video,
        duration=duration if duration else last_frame_time,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        video_start=video_start,
        audio_start=audio_start,
        last_frame_time=last_frame_time,
    )


def scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    scaled_width = min(max_width, width)
    scaled_width -= scaled_width % 2
    scaled_height = max(2, round(height * scaled_width / width / 2) * 2)
    return scaled_width, scaled_height


def gray_samples(video: Path, info: MediaInfo, fps: float, width: int) -> np.ndarray:
    sample_width, sample_height = scaled_size(info.width, info.height, width)
    raw = _run([
        "ffmpeg", "-v", "error", "-i", str(video.resolve()), "-an",
        "-vf", f"fps={fps:g},scale={sample_width}:{sample_height}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, sample_height, sample_width)


def select_expression(times: Sequence[float]) -> str:
    return "+".join(f"gte(t,{t:.6f})*(isnan(prev_pts)+lt(prev_pts*TB,{t:.6f}))" for t in times)


def extract_frames(video: Path, info: MediaInfo, targets: Sequence[tuple[float, Path]], width: int) -> None:
    if not targets:
        return
    clamped = [(min(max(time, 0.0), info.last_frame_time), destination) for time, destination in targets]
    absolute = sorted({time + info.video_start for time, _ in clamped})
    frame_width, frame_height = scaled_size(info.width, info.height, width)

    with tempfile.TemporaryDirectory() as scratch:
        result = _run([
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
            "-i", str(video.resolve()), "-an",
            "-vf", f"select='{select_expression(absolute)}',scale={frame_width}:{frame_height},showinfo",
            "-fps_mode", "passthrough", "-start_number", "0",
            str(Path(scratch) / "%05d.png"),
        ])
        stderr = result.stderr.decode(errors="replace")
        selected = [float(t) - info.video_start for t in _SHOWINFO_TIME.findall(stderr)]
        files = sorted(Path(scratch).glob("*.png"))
        if not selected or len(files) != len(selected):
            raise MediaError(f"expected {len(selected)} frames from ffmpeg but found {len(files)}")
        for time, destination in clamped:
            index = min(bisect.bisect_left(selected, time - 1e-3), len(files) - 1)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files[index], destination)

    missing = [str(destination) for _, destination in clamped if not destination.exists()]
    if missing:
        raise MediaError(f"frames were not written: {', '.join(missing)}")
