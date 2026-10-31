"""Choose which moments of a recording become screenshots."""

import bisect
import re
from collections.abc import Sequence

from feedback_collector.models import Candidate, ChangeEvent, Section, Word
from feedback_collector.sections import section_index_at

FRAMES_PER_MINUTE = 6
MIN_BUDGET = 20
MAX_BUDGET = 150
MIN_GAP = 1.0
PRIORITY = {"speech": 1, "pointer": 2, "change": 2, "silent-change": 3, "interval": 4}
POINTING_WORDS = frozenset({"this", "here", "these"})
POINTING_PHRASES = (("right", "there"), ("over", "there"))


def frame_budget(duration: float, max_frames: int | None = None) -> int:
    if max_frames is not None:
        return max_frames
    return max(MIN_BUDGET, min(MAX_BUDGET, round(FRAMES_PER_MINUTE * duration / 60)))


def _normalize(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def _pointer_candidates(words: Sequence[Word]) -> list[Candidate]:
    found = []
    normalized = [_normalize(word.text) for word in words]
    for index, (word, text) in enumerate(zip(words, normalized)):
        if text in POINTING_WORDS:
            found.append(Candidate(word.start, "pointer", f'"{text}"'))
        if tuple(normalized[index:index + 2]) in POINTING_PHRASES:
            found.append(Candidate(word.start, "pointer", f'"{text} {normalized[index + 1]}"'))
    return found


def find_candidates(
    sections: Sequence[Section],
    changes: Sequence[ChangeEvent],
    duration: float,
    every: float | None = None,
) -> list[Candidate]:
    candidates = []
    for section in sections:
        if section.narrated:
            candidates.append(Candidate(section.speech_start, "speech", "speech start"))
            words = [word for segment in section.segments for word in segment.words]
            candidates.extend(_pointer_candidates(words))
    for change in changes:
        narrated = sections[section_index_at(sections, change.settled)].narrated
        reason = "change" if narrated else "silent-change"
        candidates.append(Candidate(change.settled, reason, "screen changed"))
    if every:
        step = 1
        while step * every < duration:
            candidates.append(Candidate(step * every, "interval", f"every {every:g} s"))
            step += 1
    return sorted(candidates, key=lambda c: c.time)


def _change_between(first: float, second: float, onsets: Sequence[float]) -> bool:
    low, high = sorted((first, second))
    index = bisect.bisect_right(onsets, low)
    return index < len(onsets) and onsets[index] <= high


def _suppressed(candidate: Candidate, chosen: Sequence[Candidate], onsets: Sequence[float], min_gap: float) -> bool:
    return any(
        abs(other.time - candidate.time) < min_gap and not _change_between(other.time, candidate.time, onsets)
        for other in chosen
    )


def _spread(items: list[Candidate], count: int) -> list[Candidate]:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    last = len(items) - 1
    return [items[round(i * last / (count - 1))] for i in range(count)]


def select_frames(
    candidates: Sequence[Candidate],
    change_onsets: Sequence[float],
    budget: int | None,
    min_gap: float = MIN_GAP,
) -> list[Candidate]:
    onsets = sorted(change_onsets)
    accepted: list[Candidate] = []
    for level in sorted(set(PRIORITY.values())):
        fresh: list[Candidate] = []
        level_candidates = sorted((c for c in candidates if PRIORITY[c.reason] == level), key=lambda c: c.time)
        for candidate in level_candidates:
            if not _suppressed(candidate, accepted + fresh, onsets, min_gap):
                fresh.append(candidate)
        if budget is not None:
            remaining = budget - len(accepted)
            if remaining <= 0:
                break
            fresh = _spread(fresh, remaining)
        accepted.extend(fresh)
    return sorted(accepted, key=lambda c: c.time)
