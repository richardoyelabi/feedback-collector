"""Parsing and formatting of times in seconds."""

import math
import re

_SECONDS = re.compile(r"^\d+(\.\d+)?$")
_CLOCK = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_time(text: str) -> float:
    text = text.strip()
    if _SECONDS.match(text):
        return float(text)
    match = _CLOCK.match(text)
    if not match:
        raise ValueError(f"not a time: {text!r} (use seconds, M:SS or H:MM:SS)")
    hours, minutes, seconds = match.groups()
    minutes_value, seconds_value = int(minutes), float(seconds)
    if seconds_value >= 60 or (hours is not None and minutes_value >= 60):
        raise ValueError(f"not a time: {text!r} (minutes and seconds must be below 60)")
    return int(hours or 0) * 3600 + minutes_value * 60 + seconds_value


def _split(whole_seconds: int) -> tuple[int, int, int]:
    hours, rest = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return hours, minutes, seconds


def clock(seconds: float) -> str:
    hours, minutes, secs = _split(math.floor(seconds))
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clock_precise(seconds: float) -> str:
    whole, tenth = divmod(round(seconds * 10), 10)
    hours, minutes, secs = _split(whole)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{tenth}"
    return f"{minutes:02d}:{secs:02d}.{tenth}"


def duration_label(seconds: float) -> str:
    hours, minutes, secs = _split(math.floor(seconds))
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def filename_stamp(seconds: float) -> str:
    whole, tenth = divmod(round(seconds * 10), 10)
    hours, minutes, secs = _split(whole)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{tenth}"


def vtt_time(seconds: float) -> str:
    whole, millis = divmod(round(seconds * 1000), 1000)
    hours, minutes, secs = _split(whole)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
