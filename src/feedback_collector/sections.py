"""Split the recording's timeline into narrated and silent sections."""

from collections.abc import Sequence
from dataclasses import replace

from feedback_collector.models import Section, Segment, Word

MAX_GAP = 1.5
MAX_SECTION_LENGTH = 30.0


def _split_at_pauses(segment: Segment, max_gap: float) -> list[Segment]:
    # With VAD on, Whisper can return one segment that spans a long silence; word timing reveals it.
    pieces: list[list[Word]] = []
    for word in segment.words:
        if pieces and word.start - pieces[-1][-1].end < max_gap:
            pieces[-1].append(word)
        else:
            pieces.append([word])
    if len(pieces) < 2:
        return [segment]
    return [
        Segment(piece[0].start, piece[-1].end, " ".join(word.text for word in piece), tuple(piece))
        for piece in pieces
    ]


def _group(segments: Sequence[Segment], max_gap: float, max_length: float) -> list[list[Segment]]:
    pieces = [piece for segment in segments for piece in _split_at_pauses(segment, max_gap)]
    groups: list[list[Segment]] = []
    for segment in sorted(pieces, key=lambda s: s.start):
        if groups:
            group = groups[-1]
            group_start, group_end = group[0].start, group[-1].end
            if segment.start - group_end < max_gap and group_end - group_start < max_length:
                group.append(segment)
                continue
        groups.append([segment])
    return groups


def build_sections(
    segments: Sequence[Segment],
    duration: float,
    max_gap: float = MAX_GAP,
    max_length: float = MAX_SECTION_LENGTH,
) -> list[Section]:
    sections: list[Section] = []
    cursor = 0.0
    for group in _group(segments, max_gap, max_length):
        start = group[0].start
        if start - cursor >= max_gap:
            sections.append(Section(cursor, start))
        else:
            start = cursor
        sections.append(Section(start, group[-1].end, tuple(group)))
        cursor = group[-1].end

    if not sections:
        return [Section(0.0, duration)]
    if duration - cursor >= max_gap:
        sections.append(Section(cursor, duration))
    else:
        sections[-1] = replace(sections[-1], end=duration)
    return sections


def section_index_at(sections: Sequence[Section], time: float) -> int:
    for index, section in enumerate(sections):
        if time < section.end:
            return index
    return len(sections) - 1
