from feedback_collector.models import Section, Segment
from feedback_collector.sections import build_sections, section_index_at


def seg(start, end, text="x"):
    return Segment(start, end, text)


def spans(sections):
    return [(s.start, s.end, s.narrated) for s in sections]


def test_segments_closer_than_max_gap_share_a_section():
    assert spans(build_sections([seg(0.5, 2.0), seg(2.8, 4.0)], duration=4.5)) == [(0.0, 4.5, True)]


def test_gap_of_max_gap_or_more_becomes_silent_section():
    assert spans(build_sections([seg(0.0, 2.0), seg(5.0, 6.0)], duration=6.0)) == [
        (0.0, 2.0, True),
        (2.0, 5.0, False),
        (5.0, 6.0, True),
    ]


def test_long_narration_splits_at_segment_boundary_after_30_seconds():
    sections = build_sections([seg(t, t + 5.0) for t in range(0, 60, 5)], duration=60.0)
    assert [(s.start, s.end) for s in sections] == [(0.0, 30.0), (30.0, 60.0)]


def test_leading_and_trailing_silence():
    assert spans(build_sections([seg(3.0, 4.0)], duration=10.0)) == [
        (0.0, 3.0, False),
        (3.0, 4.0, True),
        (4.0, 10.0, False),
    ]


def test_no_segments_gives_one_silent_section():
    assert build_sections([], duration=8.0) == [Section(0.0, 8.0)]


def test_sections_are_contiguous():
    segments = [seg(0.4, 1.0), seg(4.0, 9.0), seg(9.5, 40.0), seg(41.0, 42.0)]
    sections = build_sections(segments, duration=50.0)
    assert sections[0].start == 0.0 and sections[-1].end == 50.0
    assert all(a.end == b.start for a, b in zip(sections, sections[1:]))


def test_section_text_and_speech_start():
    section = build_sections([seg(1.0, 2.0, " Hello "), seg(2.5, 3.0, "world. ")], duration=3.0)[0]
    assert (section.text, section.speech_start) == ("Hello world.", 1.0)


def test_section_index_at():
    sections = build_sections([seg(0.0, 2.0), seg(5.0, 6.0)], duration=6.0)
    times = (-1.0, 0.0, 1.9, 2.0, 4.9, 5.0, 6.0, 7.0)
    assert [section_index_at(sections, t) for t in times] == [0, 0, 0, 1, 1, 2, 2, 2]
