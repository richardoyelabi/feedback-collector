import pytest

from feedback_collector.timefmt import (
    clock,
    clock_precise,
    duration_label,
    filename_stamp,
    parse_time,
    vtt_time,
)


@pytest.mark.parametrize(
    "text, expected",
    [("45", 45.0), ("3:12.5", 192.5), ("1:02:03", 3723.0), ("90:00", 5400.0)],
)
def test_parse_time_accepts_seconds_and_clock_forms(text, expected):
    assert parse_time(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "-3", "1:75", "1:2:3:4"])
def test_parse_time_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_clock_floors_and_switches_to_hours():
    assert [clock(12.9), clock(402.1), clock(3723.0)] == ["00:12", "06:42", "1:02:03"]


def test_clock_precise_rounds_with_carry():
    assert [clock_precise(12.36), clock_precise(59.96), clock_precise(3723.26)] == [
        "00:12.4",
        "01:00.0",
        "1:02:03.3",
    ]


def test_duration_label():
    assert [duration_label(402.1), duration_label(59.9), duration_label(3723.0)] == [
        "6:42",
        "0:59",
        "1:02:03",
    ]


def test_filename_stamp():
    assert [filename_stamp(83.4), filename_stamp(3723.26)] == ["00-01-23.4", "01-02-03.3"]


def test_vtt_time():
    assert [vtt_time(12.4), vtt_time(3723.0)] == ["00:00:12.400", "01:02:03.000"]
