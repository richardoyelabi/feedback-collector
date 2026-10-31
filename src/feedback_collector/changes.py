"""Find moments where the screen visibly changes, from low-resolution gray samples."""

import numpy as np

from feedback_collector.models import ChangeEvent

SAMPLE_FPS = 2.0
SAMPLE_WIDTH = 160
PIXEL_DELTA = 16
CHANGED_FRACTION = 0.003
MIN_QUIET = 2


def changed_fractions(samples: np.ndarray, pixel_delta: int = PIXEL_DELTA) -> np.ndarray:
    differences = np.abs(np.diff(samples.astype(np.int16), axis=0))
    return (differences > pixel_delta).mean(axis=(1, 2))


def detect_changes(
    samples: np.ndarray,
    fps: float = SAMPLE_FPS,
    changed_fraction: float = CHANGED_FRACTION,
    min_quiet: int = MIN_QUIET,
) -> list[ChangeEvent]:
    if len(samples) < 2:
        return []
    changing = changed_fractions(samples) >= changed_fraction
    last_sample = len(samples) - 1
    events = []
    index = 0
    while index < len(changing):
        if not changing[index]:
            index += 1
            continue
        first = last = index
        quiet = 0
        index += 1
        while index < len(changing) and quiet < min_quiet:
            if changing[index]:
                last, quiet = index, 0
            else:
                quiet += 1
            index += 1
        events.append(ChangeEvent(onset=(first + 1) / fps, settled=min(last + 2, last_sample) / fps))
    return events
