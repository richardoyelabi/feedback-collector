"""Data passed between the pipeline's steps."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Section:
    start: float
    end: float
    segments: tuple[Segment, ...] = ()

    @property
    def narrated(self) -> bool:
        return bool(self.segments)

    @property
    def speech_start(self) -> float | None:
        return self.segments[0].start if self.segments else None

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments)


@dataclass(frozen=True)
class ChangeEvent:
    onset: float
    settled: float


@dataclass(frozen=True)
class Candidate:
    time: float
    reason: str
    label: str


@dataclass(frozen=True)
class FrameRef:
    time: float
    reason: str
    label: str
    path: str


@dataclass(frozen=True)
class Transcript:
    segments: tuple[Segment, ...]
    language: str
    model: str
    device: str


@dataclass(frozen=True)
class Report:
    video: Path
    out_dir: Path
    duration: float
    width: int
    height: int
    transcript: Transcript
    sections: tuple[Section, ...]
    frames: tuple[FrameRef, ...]
    candidate_count: int
    cap: int
