from feedback_collector.models import Candidate, ChangeEvent, Section, Segment, Word
from feedback_collector.selection import find_candidates, frame_budget, select_frames


def narrated(start, end, *words):
    text = " ".join(w for _, w in words)
    return Section(start, end, (Segment(start, end, text, tuple(Word(t, t + 0.2, w) for t, w in words)),))


def cand(time, reason="change"):
    return Candidate(time, reason, reason)


def test_frame_budget_scales_with_length():
    budgets = [frame_budget(402.1), frame_budget(30.0), frame_budget(1263.5), frame_budget(3600.0)]
    assert budgets == [40, 20, 126, 150]
    assert frame_budget(402.1, max_frames=7) == 7


def test_speech_and_pointer_candidates():
    section = narrated(
        1.0, 9.0,
        (1.0, "So"), (2.0, "This,"), (3.0, "is"), (4.0, "there"), (5.0, "right"), (5.2, "there."), (6.0, "HERE"),
    )
    assert [(c.time, c.reason, c.label) for c in find_candidates([section], [], duration=9.0)] == [
        (1.0, "speech", "speech start"),
        (2.0, "pointer", '"this"'),
        (5.0, "pointer", '"right there"'),
        (6.0, "pointer", '"here"'),
    ]


def test_changes_are_split_by_narration():
    sections = [Section(0.0, 4.0, (Segment(0.0, 4.0, "hi"),)), Section(4.0, 10.0)]
    found = find_candidates(sections, [ChangeEvent(2.0, 2.5), ChangeEvent(6.0, 6.5)], duration=10.0)
    assert [(c.time, c.reason) for c in found] == [(0.0, "speech"), (2.5, "change"), (6.5, "silent-change")]


def test_every_adds_interval_candidates():
    found = find_candidates([Section(0.0, 35.0)], [], duration=35.0, every=10.0)
    assert [(c.time, c.label) for c in found] == [(10.0, "every 10 s"), (20.0, "every 10 s"), (30.0, "every 10 s")]


def test_priority_one_wins_when_budget_is_tight():
    picked = select_frames([cand(5.0, "pointer"), cand(1.0, "speech"), cand(9.0, "silent-change")], [], budget=1)
    assert [c.reason for c in picked] == ["speech"]


def test_nearby_candidate_is_suppressed():
    picked = select_frames([cand(10.0, "speech"), cand(10.6, "pointer")], [], budget=None)
    assert [c.time for c in picked] == [10.0]


def test_change_between_keeps_both_frames():
    picked = select_frames([cand(12.0, "speech"), cand(12.9)], [12.5], budget=None)
    assert [c.time for c in picked] == [12.0, 12.9]


def test_level_is_spread_evenly_when_budget_runs_out():
    candidates = [cand(0.0, "speech"), *(cand(t) for t in (10.0, 20.0, 30.0, 40.0, 50.0))]
    picked = select_frames(candidates, [], budget=4)
    assert [c.time for c in picked] == [0.0, 10.0, 30.0, 50.0]


def test_unlimited_budget_counts_distinct_moments():
    assert len(select_frames([cand(t) for t in (1.0, 1.5, 3.0, 8.0)], [], budget=None)) == 3
