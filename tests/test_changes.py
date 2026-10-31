import numpy as np

from feedback_collector.changes import detect_changes
from feedback_collector.models import ChangeEvent


def frames(n):
    return np.full((n, 100, 160), 255, dtype=np.uint8)


def test_static_screen_has_no_changes():
    assert detect_changes(frames(10)) == []


def test_small_box_is_a_change():
    samples = frames(12)
    samples[6:, 40:50, 70:80] = 0  # 100 px = 0.625 %
    assert detect_changes(samples) == [ChangeEvent(onset=3.0, settled=3.5)]


def test_cursor_sized_change_is_ignored():
    samples = frames(12)
    samples[6:, 40:43, 70:73] = 0  # 9 px = 0.056 %
    assert detect_changes(samples) == []


def test_faint_change_is_ignored():
    samples = frames(12)
    samples[6:] = 245  # every pixel, but only 10 levels
    assert detect_changes(samples) == []


def test_scroll_burst_is_one_event_at_first_quiet_sample():
    samples = frames(20)
    for i in range(4, 11):
        samples[i:] = 255 - 20 * (i - 3)
    assert detect_changes(samples) == [ChangeEvent(onset=2.0, settled=5.5)]


def test_single_quiet_transition_does_not_split_a_burst():
    samples = frames(12)
    samples[4:, 0:20, 0:20] = 0
    samples[6:, 50:70, 50:70] = 0
    assert detect_changes(samples) == [ChangeEvent(onset=2.0, settled=3.5)]


def test_two_quiet_transitions_split_bursts():
    samples = frames(12)
    samples[4:, 0:20, 0:20] = 0
    samples[7:, 50:70, 50:70] = 0
    assert detect_changes(samples) == [ChangeEvent(onset=2.0, settled=2.5), ChangeEvent(onset=3.5, settled=4.0)]
