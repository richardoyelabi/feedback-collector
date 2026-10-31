# feedback-collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local CLI that turns a narrated screen recording into a folder holding a Markdown report that interleaves transcript and screenshots (plus JSON and WebVTT), ready for Claude Code to read.

**Architecture:** ffprobe/ffmpeg subprocesses handle probing, 2 fps grayscale sampling and one-pass frame extraction. faster-whisper (CTranslate2, with CUDA via pip-installed cuBLAS) handles transcription. Sections, change detection, frame selection and rendering are pure functions, so most logic is unit-tested without media or models.

**Tech Stack:** Python ≥3.11 (dev: 3.13), faster-whisper ≥1.2, numpy, ffmpeg/ffprobe 6.x, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-feedback-collector-design.md`

## Global Constraints

- **Python and venv:** Python `>=3.11`. The dev venv is `.venv`, created from `/home/richard/miniforge3/bin/python3` (3.13.13). Run everything as `.venv/bin/pytest`, `.venv/bin/pip`, `.venv/bin/feedback-collector`.
- **Dependencies, exactly:** runtime `faster-whisper>=1.2`, `numpy>=1.26`; extra `gpu = ["nvidia-cublas-cu12"]`; extra `dev = ["pytest>=8"]`. No Pillow, torch or opencv.
- **ffmpeg/ffprobe:** run as subprocesses with argument lists (never a shell), always given `str(path.resolve())`. image2 `%05d` output patterns only ever point inside a `tempfile` directory.
- **Style:** double quotes, type hints, frozen dataclasses for data, sparse comments. No `print` outside `cli.py`; everything else uses `logging.getLogger(__name__)`.
- **Spec values, verbatim:**
  - default model `large-v3-turbo`;
  - frames are PNG, max width `1600`, never upscaled;
  - budget `6` per minute, floor `20`, ceiling `150`;
  - section merge gap `1.5` s, split at `30` s;
  - suppression window `1.0` s;
  - change detection at `2` fps on a `160` px wide gray image, pixel delta `> 16`, changed fraction `>= 0.003`;
  - output dir `<video dir>/<stem>-feedback/`, frame names `frames/00-01-23.4_speech.png`.
- **No network after the first model download** (`local_files_only=True` first). No audio or video leaves the machine.
- **No git commits unless the user asks.** This overrides the skill's commit steps: each task ends with the full suite green instead.
- **README:** update it in the task that changes user-facing behaviour.

## Review Focus

1. **A recording with no speech** (mic on but silent, or VAD drops everything): the report still lists screen-change frames and says no speech was detected. → Task 8 `test_process_without_speech_keeps_screen_changes`.
2. **Awkward file names:** spaces, colons (as in the user's `Screencast_05_05_2026_09:08:00_AM.webm`) or `%` in the video path or output dir. Probing, sampling and extraction all work. → Task 5 `test_awkward_paths_work`.
3. **Re-running on a folder that already has a report:** refused with a message naming `--force`. `--force` replaces only `report.md`, `report.json`, `transcript.vtt` and `frames/`, and keeps the user's other files. → Task 8 `test_force_replaces_only_own_files`.
4. **Odd or small frame sizes** (1365×767, or a video narrower than `--frame-width`): output sizes are even and never upscaled. → Task 5 `test_scaled_size_is_even_and_never_upscales`.
5. **Times at or past the last frame** (very short clips, a change settling at the very end, `frames --at 99:00`): the last frame is written instead of a missing file. → Task 5 `test_extract_frames_clamps_past_end`.

---

## File Structure

```
pyproject.toml              metadata, deps, console script, pytest config
.gitignore
README.md                   install, GPU notes, usage, output, Claude Code hand-off, recording tips
src/feedback_collector/
  __init__.py               __version__
  __main__.py               python -m feedback_collector
  models.py                 frozen dataclasses shared by all modules
  timefmt.py                time parsing and formatting
  sections.py               narration sections over the timeline
  selection.py              frame candidates, priority/suppression selection, budget
  changes.py                UI-change events from gray samples
  media.py                  ffprobe/ffmpeg wrappers
  report.py                 Markdown / JSON / WebVTT rendering, hand-off prompt
  cuda_libs.py              pip cuBLAS discovery and preload
  transcription.py          faster-whisper adapter
  pipeline.py               orchestration, output-dir handling
  cli.py                    argparse subcommands
tests/
  conftest.py               synthetic ffmpeg clips, FakeTranscriber, NARRATION, png_gray
  test_timefmt.py  test_sections.py  test_selection.py  test_changes.py  test_media.py
  test_report.py  test_cuda_libs.py  test_transcription.py  test_pipeline.py  test_cli.py
```

---

### Task 1: Scaffold and time formatting

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/feedback_collector/__init__.py`, `src/feedback_collector/timefmt.py`
- Test: `tests/test_timefmt.py`

**Interfaces:**
- Produces (`feedback_collector.timefmt`):
  - `parse_time(text: str) -> float`: accepts `"45"`, `"3:12.5"`, `"1:02:03"`. Every field except the leading one must be `< 60`. Raises `ValueError` otherwise.
  - `clock(seconds: float) -> str`: floors to whole seconds; `"MM:SS"` below 1 h, `"H:MM:SS"` from 1 h.
  - `clock_precise(seconds: float) -> str`: rounds to 0.1 s with carry; `"MM:SS.s"` below 1 h, `"H:MM:SS.s"` from 1 h.
  - `duration_label(seconds: float) -> str`: floors; `"M:SS"` below 1 h, `"H:MM:SS"` from 1 h.
  - `filename_stamp(seconds: float) -> str`: rounds to 0.1 s; always `"HH-MM-SS.s"`.
  - `vtt_time(seconds: float) -> str`: rounds to ms; `"HH:MM:SS.mmm"`.

- [ ] **Step 1: Scaffold.**
  - Run `git init`.
  - `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`.
  - `pyproject.toml`: setuptools with a `src` layout, the Global Constraints deps, and `[project.scripts] feedback-collector = "feedback_collector.cli:main"`, plus:
    ```toml
    [tool.pytest.ini_options]
    testpaths = ["tests"]
    markers = ["slow: runs the real speech model on $FEEDBACK_COLLECTOR_SAMPLE"]
    addopts = "-m 'not slow'"
    ```
  - `__init__.py`: `__version__ = "0.1.0"`.
  - `README.md`: a purpose paragraph, a link to the design doc, and a Development section (venv, `pip install -e ".[dev]"`, `pytest`).
- [ ] **Step 2: Create the venv.** Run: `/home/richard/miniforge3/bin/python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`. Expected: faster-whisper, numpy and pytest install without errors.
- [ ] **Step 3: Write the failing tests** in `tests/test_timefmt.py`:
```python
@pytest.mark.parametrize("text, expected", [("45", 45.0), ("3:12.5", 192.5), ("1:02:03", 3723.0), ("90:00", 5400.0)])
def test_parse_time_accepts_seconds_and_clock_forms(text, expected):
    assert parse_time(text) == expected

@pytest.mark.parametrize("text", ["", "abc", "-3", "1:75", "1:2:3:4"])
def test_parse_time_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_time(text)

def test_clock_floors_and_switches_to_hours():
    assert [clock(12.9), clock(402.1), clock(3723.0)] == ["00:12", "06:42", "1:02:03"]

def test_clock_precise_rounds_with_carry():
    assert [clock_precise(12.36), clock_precise(59.96), clock_precise(3723.26)] == ["00:12.4", "01:00.0", "1:02:03.3"]

def test_duration_label():
    assert [duration_label(402.1), duration_label(59.9), duration_label(3723.0)] == ["6:42", "0:59", "1:02:03"]

def test_filename_stamp():
    assert [filename_stamp(83.4), filename_stamp(3723.26)] == ["00-01-23.4", "01-02-03.3"]

def test_vtt_time():
    assert [vtt_time(12.4), vtt_time(3723.0)] == ["00:00:12.400", "01:02:03.000"]
```
- [ ] **Step 4: Run them.** Run: `.venv/bin/pytest tests/test_timefmt.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 5: Implement `timefmt.py`.** Compute integer tenths or milliseconds with `round(seconds * 10)` / `round(seconds * 1000)`, then use `divmod`, so carries are exact.
- [ ] **Step 6: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 2: Shared models and narration sections

**Files:**
- Create: `src/feedback_collector/models.py`, `src/feedback_collector/sections.py`
- Test: `tests/test_sections.py`

**Interfaces:**
- Produces (`feedback_collector.models`), all `@dataclass(frozen=True)`:
  - `Word(start: float, end: float, text: str)`
  - `Segment(start: float, end: float, text: str, words: tuple[Word, ...] = ())`
  - `Section(start: float, end: float, segments: tuple[Segment, ...] = ())`, with properties:
    - `narrated -> bool`
    - `speech_start -> float | None` (the first segment's start)
    - `text -> str` (stripped segment texts joined by one space)
  - `ChangeEvent(onset: float, settled: float)`: onset is the first sample showing the change; settled is where the frame is taken.
  - `Candidate(time: float, reason: str, label: str)`: reason is one of `"speech"`, `"pointer"`, `"change"`, `"silent-change"`, `"interval"`.
  - `FrameRef(time: float, reason: str, label: str, path: str)`: `path` is relative to the output dir.
  - `Transcript(segments: tuple[Segment, ...], language: str, model: str, device: str)`
  - `Report(video: Path, out_dir: Path, duration: float, width: int, height: int, transcript: Transcript, sections: tuple[Section, ...], frames: tuple[FrameRef, ...], candidate_count: int, cap: int)`
- Produces (`feedback_collector.sections`):
  - `MAX_GAP = 1.5`, `MAX_SECTION_LENGTH = 30.0`
  - `build_sections(segments: Sequence[Segment], duration: float, max_gap: float = MAX_GAP, max_length: float = MAX_SECTION_LENGTH) -> list[Section]`: contiguous sections covering `[0, duration]`, in order.
  - `section_index_at(sections: Sequence[Section], time: float) -> int`: the section with `start <= time < end`. Times below 0 map to the first section; times at or past the last end map to the last.

Rules `build_sections` follows:
1. Sort segments by start. A segment joins the current group when `segment.start - group_end < max_gap` **and** `group_end - group_start < max_length`. Otherwise it starts a new group.
2. Walk the groups with `cursor = 0.0`:
   - If `group.start - cursor >= max_gap`, emit a silent `Section(cursor, group.start)` and start the narrated section at `group.start`.
   - Otherwise start the narrated section at `cursor`; smaller gaps are absorbed.
   - The narrated section ends at its last segment's end, and `cursor` moves there.
3. After the last group: if `duration - cursor >= max_gap`, emit a silent `Section(cursor, duration)`. Otherwise extend the last narrated section's `end` to `duration`.
4. With no segments the result is `[Section(0.0, duration)]`.

- [ ] **Step 1: Write the failing tests** in `tests/test_sections.py` (with the helper `seg(start, end, text="x") -> Segment(start, end, text)`):
```python
def spans(sections): return [(s.start, s.end, s.narrated) for s in sections]

def test_segments_closer_than_max_gap_share_a_section():
    assert spans(build_sections([seg(0.5, 2.0), seg(2.8, 4.0)], duration=4.5)) == [(0.0, 4.5, True)]

def test_gap_of_max_gap_or_more_becomes_silent_section():
    assert spans(build_sections([seg(0.0, 2.0), seg(5.0, 6.0)], duration=6.0)) == [(0.0, 2.0, True), (2.0, 5.0, False), (5.0, 6.0, True)]

def test_long_narration_splits_at_segment_boundary_after_30_seconds():
    sections = build_sections([seg(t, t + 5.0) for t in range(0, 60, 5)], duration=60.0)
    assert [(s.start, s.end) for s in sections] == [(0.0, 30.0), (30.0, 60.0)]

def test_leading_and_trailing_silence():
    assert spans(build_sections([seg(3.0, 4.0)], duration=10.0)) == [(0.0, 3.0, False), (3.0, 4.0, True), (4.0, 10.0, False)]

def test_no_segments_gives_one_silent_section():
    assert build_sections([], duration=8.0) == [Section(0.0, 8.0)]

def test_sections_are_contiguous():
    sections = build_sections([seg(0.4, 1.0), seg(4.0, 9.0), seg(9.5, 40.0), seg(41.0, 42.0)], duration=50.0)
    assert sections[0].start == 0.0 and sections[-1].end == 50.0
    assert all(a.end == b.start for a, b in zip(sections, sections[1:]))

def test_section_text_and_speech_start():
    section = build_sections([seg(1.0, 2.0, " Hello "), seg(2.5, 3.0, "world. ")], duration=3.0)[0]
    assert (section.text, section.speech_start) == ("Hello world.", 1.0)

def test_section_index_at():
    sections = build_sections([seg(0.0, 2.0), seg(5.0, 6.0)], duration=6.0)
    assert [section_index_at(sections, t) for t in (-1.0, 0.0, 1.9, 2.0, 4.9, 5.0, 6.0, 7.0)] == [0, 0, 0, 1, 1, 2, 2, 2]
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_sections.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `models.py` and `sections.py` as specified.
- [ ] **Step 4: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 3: Frame candidates, selection and budget

**Files:**
- Create: `src/feedback_collector/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `Section`, `Segment`, `Word`, `ChangeEvent` and `Candidate` from `models`; `section_index_at` from `sections`.
- Produces (`feedback_collector.selection`):
  - Constants: `FRAMES_PER_MINUTE = 6`, `MIN_BUDGET = 20`, `MAX_BUDGET = 150`, `MIN_GAP = 1.0`, `PRIORITY = {"speech": 1, "pointer": 2, "change": 2, "silent-change": 3, "interval": 4}`, `POINTING_WORDS = frozenset({"this", "here", "these"})`, `POINTING_PHRASES = (("right", "there"), ("over", "there"))`.
  - `frame_budget(duration: float, max_frames: int | None = None) -> int`: returns `max_frames` when given, else `max(MIN_BUDGET, min(MAX_BUDGET, round(FRAMES_PER_MINUTE * duration / 60)))`.
  - `find_candidates(sections: Sequence[Section], changes: Sequence[ChangeEvent], duration: float, every: float | None = None) -> list[Candidate]`: sorted by time.
  - `select_frames(candidates: Sequence[Candidate], change_onsets: Sequence[float], budget: int | None, min_gap: float = MIN_GAP) -> list[Candidate]`: sorted by time. `budget=None` means unlimited and is used for the candidate count.

Candidate rules:
- Each narrated section gives `Candidate(section.speech_start, "speech", "speech start")`.
- **Pointing words:** go through the words of a narrated section's segments in order and normalize each one (lowercase, letters only).
  - A normalized word in `POINTING_WORDS` gives `Candidate(word.start, "pointer", f'"{normalized}"')`.
  - Each consecutive pair in `POINTING_PHRASES` gives `Candidate(first.start, "pointer", '"right there"')` (or `'"over there"'`).
- Each change gives `Candidate(settled, "change", "screen changed")` when the section containing `settled` is narrated, and `Candidate(settled, "silent-change", "screen changed")` otherwise.
- With `every`, add `Candidate(k * every, "interval", f"every {every:g} s")` for k = 1, 2, … while the time is `< duration`.

Selection (the algorithm the signature doesn't fix):
```python
accepted: list[Candidate] = []
for level in sorted(set(PRIORITY.values())):
    fresh: list[Candidate] = []
    for candidate in sorted((c for c in candidates if PRIORITY[c.reason] == level), key=lambda c: c.time):
        if not _suppressed(candidate, accepted + fresh, change_onsets, min_gap):
            fresh.append(candidate)
    if budget is not None:
        remaining = budget - len(accepted)
        if remaining <= 0:
            break
        fresh = _spread(fresh, remaining)
    accepted.extend(fresh)
return sorted(accepted, key=lambda c: c.time)
```
- `_suppressed`: true when some chosen `a` has `abs(a.time - c.time) < min_gap` and no onset `o` satisfies `min(a.time, c.time) < o <= max(a.time, c.time)`.
- `_spread(items, k)`:
  - returns the list unchanged when `len(items) <= k`;
  - returns the middle item when `k == 1`;
  - otherwise returns the items at indices `round(i * (n - 1) / (k - 1))` for `i in range(k)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_selection.py`:
```python
def narrated(start, end, *words):
    text = " ".join(w for _, w in words)
    return Section(start, end, (Segment(start, end, text, tuple(Word(t, t + 0.2, w) for t, w in words)),))

def cand(time, reason="change"): return Candidate(time, reason, reason)

def test_frame_budget_scales_with_length():
    assert [frame_budget(402.1), frame_budget(30.0), frame_budget(1263.5), frame_budget(3600.0)] == [40, 20, 126, 150]
    assert frame_budget(402.1, max_frames=7) == 7

def test_speech_and_pointer_candidates():
    section = narrated(1.0, 9.0, (1.0, "So"), (2.0, "This,"), (3.0, "is"), (4.0, "there"), (5.0, "right"), (5.2, "there."), (6.0, "HERE"))
    assert [(c.time, c.reason, c.label) for c in find_candidates([section], [], duration=9.0)] == [
        (1.0, "speech", "speech start"), (2.0, "pointer", '"this"'), (5.0, "pointer", '"right there"'), (6.0, "pointer", '"here"')]

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
    assert [c.time for c in select_frames([cand(10.0, "speech"), cand(10.6, "pointer")], [], budget=None)] == [10.0]

def test_change_between_keeps_both_frames():
    assert [c.time for c in select_frames([cand(12.0, "speech"), cand(12.9)], [12.5], budget=None)] == [12.0, 12.9]

def test_level_is_spread_evenly_when_budget_runs_out():
    picked = select_frames([cand(0.0, "speech"), *(cand(t) for t in (10.0, 20.0, 30.0, 40.0, 50.0))], [], budget=4)
    assert [c.time for c in picked] == [0.0, 10.0, 30.0, 50.0]

def test_unlimited_budget_counts_distinct_moments():
    assert len(select_frames([cand(t) for t in (1.0, 1.5, 3.0, 8.0)], [], budget=None)) == 3
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_selection.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `selection.py`** as specified.
- [ ] **Step 4: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 4: UI-change detection

**Files:**
- Create: `src/feedback_collector/changes.py`
- Test: `tests/test_changes.py`

**Interfaces:**
- Consumes: `ChangeEvent` from `models`.
- Produces (`feedback_collector.changes`):
  - Constants: `SAMPLE_FPS = 2.0`, `SAMPLE_WIDTH = 160`, `PIXEL_DELTA = 16`, `CHANGED_FRACTION = 0.003`, `MIN_QUIET = 2`.
  - `changed_fractions(samples: np.ndarray, pixel_delta: int = PIXEL_DELTA) -> np.ndarray`: shape `(n - 1,)`. Each value is the fraction of pixels whose absolute difference between consecutive samples is `> pixel_delta`, computed in `int16`.
  - `detect_changes(samples: np.ndarray, fps: float = SAMPLE_FPS, changed_fraction: float = CHANGED_FRACTION, min_quiet: int = MIN_QUIET) -> list[ChangeEvent]`.

Burst rule:
- Transition `i` (sample `i` → `i+1`) counts as changing when its fraction is `>= changed_fraction`.
- A burst starts at a changing transition `first`. It ends after `min_quiet` consecutive quiet transitions; `last` is its final changing transition.
- Each burst gives one event: `onset = (first + 1) / fps` and `settled = min(last + 2, n - 1) / fps`.

- [ ] **Step 1: Write the failing tests** in `tests/test_changes.py`. Samples are `(n, 100, 160)` uint8 with a white (255) background:
```python
def frames(n): return np.full((n, 100, 160), 255, dtype=np.uint8)

def test_static_screen_has_no_changes():
    assert detect_changes(frames(10)) == []

def test_small_box_is_a_change():
    samples = frames(12); samples[6:, 40:50, 70:80] = 0          # 100 px = 0.625 %
    assert detect_changes(samples) == [ChangeEvent(onset=3.0, settled=3.5)]

def test_cursor_sized_change_is_ignored():
    samples = frames(12); samples[6:, 40:43, 70:73] = 0          # 9 px = 0.056 %
    assert detect_changes(samples) == []

def test_faint_change_is_ignored():
    samples = frames(12); samples[6:] = 245                      # every pixel, only 10 levels
    assert detect_changes(samples) == []

def test_scroll_burst_is_one_event_at_first_quiet_sample():
    samples = frames(20)
    for i in range(4, 11):
        samples[i:] = 255 - 20 * (i - 3)                         # changes into samples 4..10
    assert detect_changes(samples) == [ChangeEvent(onset=2.0, settled=5.5)]

def test_single_quiet_transition_does_not_split_a_burst():
    samples = frames(12); samples[4:, 0:20, 0:20] = 0; samples[6:, 50:70, 50:70] = 0
    assert detect_changes(samples) == [ChangeEvent(onset=2.0, settled=3.5)]

def test_two_quiet_transitions_split_bursts():
    samples = frames(12); samples[4:, 0:20, 0:20] = 0; samples[7:, 50:70, 50:70] = 0
    assert detect_changes(samples) == [ChangeEvent(onset=2.0, settled=2.5), ChangeEvent(onset=3.5, settled=4.0)]
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_changes.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `changes.py`** with the burst rule above.
- [ ] **Step 4: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 5: Media (ffprobe and ffmpeg)

**Files:**
- Create: `src/feedback_collector/media.py`, `tests/conftest.py`
- Test: `tests/test_media.py`

**Interfaces:**
- Consumes: `SAMPLE_FPS`, `SAMPLE_WIDTH` and `detect_changes` from `changes`, in one integration test.
- Produces (`feedback_collector.media`):
  - `class MediaError(Exception)`.
  - `@dataclass(frozen=True) MediaInfo(path: Path, duration: float, width: int, height: int, video_start: float, audio_start: float | None, last_frame_time: float)`, with properties:
    - `has_audio -> bool`;
    - `audio_offset -> float`: `audio_start - video_start`, or `0.0` without audio.
    `duration` and `last_frame_time` are relative to the first video frame; `video_start` and `audio_start` are the streams' absolute `start_time`.
  - `probe(video: Path) -> MediaInfo`.
  - `scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]`: width is `min(max_width, width)` rounded down to even; height is `round(height * w / width)` rounded to the nearest even, at least 2.
  - `gray_samples(video: Path, info: MediaInfo, fps: float, width: int) -> np.ndarray`: shape `(n, h, w)`, uint8, with `(w, h) = scaled_size(info.width, info.height, width)`.
  - `select_expression(times: Sequence[float]) -> str`: given absolute times, returns `"+".join(f"gte(t,{t:.6f})*(isnan(prev_pts)+lt(prev_pts*TB,{t:.6f}))" for t in times)`. This selects the first frame at or after each time, and was verified on ffmpeg 6.1.
  - `extract_frames(video: Path, info: MediaInfo, targets: Sequence[tuple[float, Path]], width: int) -> None`: writes one PNG for each `(time, destination)` pair.
  - `_run(args: Sequence[str]) -> subprocess.CompletedProcess[bytes]`: the only caller of `subprocess.run` (captures output, no shell). It raises `MediaError("ffmpeg/ffprobe not found on PATH; install ffmpeg (sudo apt install ffmpeg)")` when the binary is missing, and a `MediaError` carrying the last 20 stderr lines on a non-zero exit.

Command shapes, where PATH is always `str(video.resolve())`:
- **probe** runs two commands:
  - `ffprobe -v error -show_entries format=duration:stream=codec_type,width,height,start_time -of json PATH`
  - `ffprobe -v error -select_streams v:0 -show_entries packet=pts_time -of csv=p=0 PATH`

  Then:
  - `last_frame_time = max(pts) - video_start`.
  - `duration` is `float(format.duration)` when numeric, else `last_frame_time`.
  - A `start_time` that is missing or `"N/A"` counts as `0.0`.
  - A missing file raises `MediaError(f"{video} does not exist")`; no video stream raises `MediaError(f"{video.name} has no video stream")`.
- **gray:** `ffmpeg -v error -i PATH -an -vf fps={fps:g},scale={w}:{h}:flags=area,format=gray -f rawvideo -pix_fmt gray -`.
- **extract**, in one process:
  1. Clamp each time to `[0, info.last_frame_time]`, then take the unique sorted absolute times `t + info.video_start`.
  2. In a `tempfile.TemporaryDirectory()`, run `ffmpeg -hide_banner -nostats -loglevel info -i PATH -an -vf "select='{expr}',scale={w}:{h},showinfo" -fps_mode passthrough -start_number 0 {tmp}/%05d.png`.
  3. Parse `pts_time:([0-9.]+)` from the stderr lines containing `Parsed_showinfo`. Output file `k` has canonical time `times[k]` (absolute minus `video_start`).
  4. Each target gets file `bisect_left(times, target_time - 1e-3)` (the last file when past the end), copied to its destination with parents created.
  5. Raise `MediaError` naming any destination that still doesn't exist.

`conftest.py`: session-scoped clip fixtures in `tmp_path_factory`. The media tests are skipped when `shutil.which("ffmpeg") is None`.
- `box_clip`: `ffmpeg -y -v error -f lavfi -i "color=c=white:s=320x200:r=10:d=6,drawbox=x=140:y=80:w=40:h=40:color=black:t=fill:enable='gte(t,3)'" -f lavfi -i "sine=frequency=440:duration=6" -c:v libvpx -b:v 300k -c:a libopus -shortest box.webm`
- `silent_clip`: the same video with `-an`.
- `streamed_clip`: the same as `box_clip` but written with `-f webm pipe:1` and saved from stdout, so the container has no duration.
- `png_gray(path: Path, width: int, height: int) -> np.ndarray`: decodes a PNG with `ffmpeg -v error -i PATH -f rawvideo -pix_fmt gray -`.

- [ ] **Step 1: Write the failing tests** in `tests/test_media.py`:
```python
def test_probe_reads_size_duration_and_audio(box_clip):
    info = probe(box_clip)
    assert (info.width, info.height, info.has_audio) == (320, 200, True)
    assert info.duration == pytest.approx(6.0, abs=0.1)
    assert info.last_frame_time == pytest.approx(5.9, abs=0.05)

def test_probe_without_audio(silent_clip):
    assert not probe(silent_clip).has_audio

def test_probe_falls_back_to_last_frame_without_container_duration(streamed_clip):
    info = probe(streamed_clip)
    assert info.duration == pytest.approx(info.last_frame_time)
    assert info.duration == pytest.approx(5.9, abs=0.05)

def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(MediaError, match="does not exist"):
        probe(tmp_path / "nope.webm")

@pytest.mark.parametrize("size, max_width, expected", [
    ((2560, 1600), 1600, (1600, 1000)), ((1920, 1080), 1600, (1600, 900)),
    ((1365, 767), 1600, (1364, 766)), ((320, 200), 1600, (320, 200))])
def test_scaled_size_is_even_and_never_upscales(size, max_width, expected):
    assert scaled_size(*size, max_width) == expected

def test_gray_samples_shape(box_clip):
    samples = gray_samples(box_clip, probe(box_clip), fps=2.0, width=160)
    assert samples.dtype == np.uint8 and samples.shape[1:] == (100, 160) and 11 <= samples.shape[0] <= 13

def test_box_clip_has_one_change_near_three_seconds(box_clip):
    events = detect_changes(gray_samples(box_clip, probe(box_clip), fps=SAMPLE_FPS, width=SAMPLE_WIDTH))
    assert len(events) == 1 and events[0].onset == pytest.approx(3.0, abs=0.5)

def test_extract_frames_uses_one_ffmpeg_process(box_clip, tmp_path, monkeypatch):
    info = probe(box_clip)
    calls, real_run = [], media._run
    monkeypatch.setattr(media, "_run", lambda args: calls.append(args[0]) or real_run(args))
    targets = [(t, tmp_path / f"{t}.png") for t in (1.0, 3.5, 5.0)]
    extract_frames(box_clip, info, targets, width=160)
    assert calls.count("ffmpeg") == 1 and all(path.exists() for _, path in targets)

def test_extract_frames_shows_the_right_moment(box_clip, tmp_path):
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    extract_frames(box_clip, probe(box_clip), [(1.0, before), (3.5, after)], width=160)
    assert png_gray(before, 160, 100)[40:60, 70:90].mean() > 200
    assert png_gray(after, 160, 100)[40:60, 70:90].mean() < 60

def test_extract_frames_clamps_past_end(box_clip, tmp_path):
    late = tmp_path / "late.png"
    extract_frames(box_clip, probe(box_clip), [(99.0, late)], width=160)
    assert late.exists()

def test_awkward_paths_work(box_clip, tmp_path):
    video = tmp_path / "rec 09:08:00 50%.webm"
    shutil.copy(box_clip, video)
    destination = tmp_path / "out %d dir" / "frame.png"
    info = probe(video)
    assert gray_samples(video, info, fps=2.0, width=160).shape[0] > 0
    extract_frames(video, info, [(1.0, destination)], width=160)
    assert destination.exists()
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_media.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `media.py`** with the command shapes above.
- [ ] **Step 4: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 6: Report rendering

**Files:**
- Create: `src/feedback_collector/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Report`, `Section`, `FrameRef`, `Segment` and `Transcript` from `models`; `clock`, `clock_precise`, `duration_label` and `vtt_time` from `timefmt`; `section_index_at` from `sections`.
- Produces (`feedback_collector.report`):
  - `visible_sections(report: Report) -> list[tuple[Section, list[FrameRef]]]`: frames grouped by `section_index_at(sections, frame.time)`. Narrated sections always appear; silent ones only when they have frames.
  - `render_markdown(report: Report) -> str`
  - `render_json(report: Report) -> str` (indent 2, times rounded to 3 decimals, visible sections only)
  - `render_vtt(segments: Sequence[Segment]) -> str`
  - `handoff_prompt(report_path: Path) -> str`

Exact Markdown copy, with blocks separated by one blank line. Pluralize `section`/`screenshot` only when the count isn't 1.
```
# Feedback walkthrough: {video.name}
`{duration_label}` · `{n} sections` · `{selected} screenshots (cap {cap}) of {candidate_count} candidates` · transcript: faster-whisper {model} ({device}), **not reviewed by a human**

**How to read this:** sections are in recording order. Open every screenshot (paths are relative to this file) before acting.
Narration is speech-to-text and may mishear names; where words and screenshots disagree, trust the screenshots.
Need another moment? Run `feedback-collector frames "{video}" --at 03:12.5 --out "{out_dir}/extra"`.
```
After the header:
- When no visible section is narrated, add `_No speech was detected in this recording._`.
- Then for each visible section:
  - the heading `## {clock(start)}–{clock(end)}`, with ` (no narration)` appended for silent sections;
  - for narrated sections, `> {section.text}` followed by a blank line;
  - one `![{clock_precise(time)} {label}]({path})` line per frame.

JSON keys:
- `source`: `{path, duration, width, height}`
- `transcription`: `{engine: "faster-whisper", model, device, language, human_reviewed: false}`
- `frames`: `{selected, candidates, cap}`
- `sections`: a list of `{start, end, narrated, text, segments: [{start, end, text}], frames: [{time, reason, label, path}]}`

VTT: `"WEBVTT\n\n"` followed by `f"{vtt_time(start)} --> {vtt_time(end)}\n{text}\n\n"` for each segment.

Hand-off copy: `Read "{report_path}" and open every screenshot it lists. It is narrated feedback I recorded while testing this app. For each issue I raise, find the code responsible and propose a fix. Where my words and the screenshots disagree, trust the screenshots.`

- [ ] **Step 1: Write the failing tests** in `tests/test_report.py`. The `report` fixture uses:
  - `video=/videos/rec.webm`, `out_dir=/videos/rec-feedback`, duration 402.1, size 2560×1600;
  - a transcript with model `large-v3-turbo`, device `cuda`, language `en`;
  - `candidate_count=57`, `cap=40`;
  - four sections: `Section(0.0, 27.0, (Segment(12.4, 27.0, "So this button is confusing."),))`, `Section(27.0, 41.0)`, `Section(41.0, 60.0)` and `Section(60.0, 402.1, (Segment(60.0, 70.0, "Next issue."),))`;
  - three frames: `FrameRef(12.4, "speech", "speech start", "frames/00-00-12.4_speech.png")`, `FrameRef(15.1, "pointer", '"this"', "frames/00-00-15.1_pointer.png")` and `FrameRef(30.2, "silent-change", "screen changed", "frames/00-00-30.2_silent-change.png")`.
```python
def test_header_counts_and_reader_notes(report):
    text = render_markdown(report)
    assert text.startswith("# Feedback walkthrough: rec.webm\n")
    assert "`6:42` · `3 sections` · `3 screenshots (cap 40) of 57 candidates` · transcript: faster-whisper large-v3-turbo (cuda), **not reviewed by a human**" in text
    assert 'Run `feedback-collector frames "/videos/rec.webm" --at 03:12.5 --out "/videos/rec-feedback/extra"`.' in text

def test_sections_interleave_quotes_and_frames(report):
    assert ('## 00:00–00:27\n> So this button is confusing.\n\n![00:12.4 speech start](frames/00-00-12.4_speech.png)\n'
            '![00:15.1 "this"](frames/00-00-15.1_pointer.png)') in render_markdown(report)

def test_silent_sections_only_appear_with_frames(report):
    text = render_markdown(report)
    assert "## 00:27–00:41 (no narration)\n![00:30.2 screen changed](frames/00-00-30.2_silent-change.png)" in text
    assert "00:41–01:00" not in text

def test_no_speech_note(report):
    silent = dataclasses.replace(report, sections=(Section(0.0, 402.1),), frames=report.frames[2:])
    assert "_No speech was detected in this recording._" in render_markdown(silent)

def test_json_mirrors_report(report):
    data = json.loads(render_json(report))
    assert data["frames"] == {"selected": 3, "candidates": 57, "cap": 40}
    assert data["transcription"]["human_reviewed"] is False
    assert [s["narrated"] for s in data["sections"]] == [True, False, True]
    assert data["sections"][0]["frames"][1]["label"] == '"this"'

def test_vtt():
    assert render_vtt([Segment(12.4, 15.0, "Hello.")]) == "WEBVTT\n\n00:00:12.400 --> 00:00:15.000\nHello.\n\n"

def test_handoff_prompt_quotes_the_path():
    assert handoff_prompt(Path("/x/report.md")).startswith('Read "/x/report.md" and open every screenshot it lists.')
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_report.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `report.py`.**
- [ ] **Step 4: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 7: CUDA libraries and transcription

**Files:**
- Create: `src/feedback_collector/cuda_libs.py`, `src/feedback_collector/transcription.py`
- Test: `tests/test_cuda_libs.py`, `tests/test_transcription.py`
- Modify: `README.md` (add a GPU section)

**Interfaces:**
- Consumes: `Word`, `Segment` and `Transcript` from `models`.
- Produces (`feedback_collector.cuda_libs`):
  - `CUBLAS_RELATIVE = Path("nvidia/cublas/lib/libcublas.so.12")`
  - `find_cublas(search_paths: Iterable[str | Path] | None = None) -> Path | None`: the first `entry / CUBLAS_RELATIVE` that exists, searching `sys.path` by default. CUDA 13 copies such as `nvidia/cu13/lib/libcublas.so.13` never match.
  - `preload_cublas(search_paths: Iterable[str | Path] | None = None) -> bool`: calls `ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)`; returns `False` when nothing is found or on `OSError`.
  - `cuda_ready() -> bool`: `preload_cublas()` and then `ctranslate2.get_cuda_device_count() > 0`, imported lazily. Returns `False` on `ImportError` or `RuntimeError`.
- Produces (`feedback_collector.transcription`):
  - `DEFAULT_MODEL = "large-v3-turbo"`
  - `class Transcriber(Protocol)`, with `transcribe(self, video: Path, offset: float = 0.0) -> Transcript`.
  - `load_whisper_model(name: str, device: str, compute_type: str, local_files_only: bool) -> Any`: imports `faster_whisper.WhisperModel` lazily and returns `WhisperModel(name, device=device, compute_type=compute_type, local_files_only=local_files_only)`.
  - `class WhisperTranscriber`:
    - `__init__(self, model: str = DEFAULT_MODEL, device: str = "auto", language: str | None = None, hotwords: str | None = None, model_factory: Callable[..., Any] = load_whisper_model, cuda_check: Callable[[], bool] = cuda_ready)`
    - `transcribe(self, video: Path, offset: float = 0.0) -> Transcript`
  - `LocalEntryNotFoundError`, re-exported from huggingface_hub. Confirm the path first with `.venv/bin/python -c "from huggingface_hub.errors import LocalEntryNotFoundError"` and use wherever it lives in the installed version.

How `WhisperTranscriber` behaves:
- **Devices:** `"auto"` tries `["cuda", "cpu"]` when `cuda_check()` is true, else `["cpu"]`. `"cuda"` tries `["cuda"]` only, `"cpu"` tries `["cpu"]` only. The compute type is `float16` on cuda and `int8` on cpu.
- **Loading:** call `model_factory(name, device, compute_type, local_files_only=True)` first. On `LocalEntryNotFoundError`, log `Downloading speech model {name} (one time)…` and call it again with `local_files_only=False`.
- **Transcribing:** inside the same `try`, call `model.transcribe(str(video), language=language, hotwords=hotwords, vad_filter=True, word_timestamps=True)` and convert **every** segment.
- **Fallback:** on `(RuntimeError, ValueError, OSError)`:
  - if devices remain, log the warning `CUDA transcription failed ({error}); retrying on CPU` and continue;
  - on the last device, re-raise.
- **Conversion:** each segment becomes `Segment(s.start + offset, s.end + offset, s.text.strip(), tuple(Word(w.start + offset, w.end + offset, w.word.strip()) for w in (s.words or ())))`, and the result is `Transcript(segments, info.language, name, device)`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_cuda_libs.py`:
```python
def make_lib(root, relative="nvidia/cublas/lib/libcublas.so.12"):
    lib = root / relative; lib.parent.mkdir(parents=True); lib.touch(); return lib

def test_find_cublas_in_pip_layout(tmp_path):
    assert find_cublas([tmp_path]) == make_lib(tmp_path)

def test_find_cublas_ignores_cuda_13(tmp_path):
    make_lib(tmp_path, "nvidia/cu13/lib/libcublas.so.13")
    assert find_cublas([tmp_path]) is None

def test_preload_cublas_loads_globally(tmp_path, monkeypatch):
    lib, calls = make_lib(tmp_path), []
    monkeypatch.setattr(cuda_libs.ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))
    assert preload_cublas([tmp_path]) is True and calls == [(str(lib), ctypes.RTLD_GLOBAL)]

def test_preload_cublas_reports_unloadable_file(tmp_path):
    make_lib(tmp_path)                       # empty file is not a shared object
    assert preload_cublas([tmp_path]) is False
```
In `tests/test_transcription.py`, `FakeModel(segments=(), fail=None)` records the `transcribe` kwargs and returns `(generator, SimpleNamespace(language="en"))`. The generator raises `fail` on first iteration when set. `fake_segment` is a `SimpleNamespace(start=1.0, end=2.0, text=" Hello there. ", words=[SimpleNamespace(start=1.0, end=1.4, word=" Hello"), SimpleNamespace(start=1.5, end=2.0, word=" there.")])`.
```python
def test_segments_are_converted_and_shifted():
    transcriber = WhisperTranscriber(device="cpu", model_factory=lambda *a, **k: FakeModel([fake_segment]), cuda_check=lambda: False)
    transcript = transcriber.transcribe(Path("x.webm"), offset=0.25)
    assert transcript.segments[0] == Segment(1.25, 2.25, "Hello there.", (Word(1.25, 1.65, "Hello"), Word(1.75, 2.25, "there.")))
    assert (transcript.language, transcript.model, transcript.device) == ("en", "large-v3-turbo", "cpu")

def test_options_reach_faster_whisper():
    model = FakeModel()
    WhisperTranscriber(device="cpu", language="en", hotwords="Stripe, SKU", model_factory=lambda *a, **k: model, cuda_check=lambda: False).transcribe(Path("x.webm"))
    assert model.kwargs == {"language": "en", "hotwords": "Stripe, SKU", "vad_filter": True, "word_timestamps": True}

def test_model_loads_offline_first_then_downloads():
    calls = []
    def factory(name, device, compute_type, local_files_only):
        calls.append(local_files_only)
        if local_files_only:
            raise transcription.LocalEntryNotFoundError("not cached")
        return FakeModel()
    WhisperTranscriber(device="cpu", model_factory=factory, cuda_check=lambda: False).transcribe(Path("x.webm"))
    assert calls == [True, False]

def test_cuda_failure_while_reading_segments_retries_on_cpu():
    models, calls = {"cuda": FakeModel(fail=RuntimeError("cuBLAS failed")), "cpu": FakeModel([fake_segment])}, []
    def factory(name, device, compute_type, local_files_only):
        calls.append((device, compute_type)); return models[device]
    transcript = WhisperTranscriber(model_factory=factory, cuda_check=lambda: True).transcribe(Path("x.webm"))
    assert transcript.device == "cpu" and calls == [("cuda", "float16"), ("cpu", "int8")]

def test_explicit_cuda_does_not_fall_back():
    transcriber = WhisperTranscriber(device="cuda", model_factory=lambda *a, **k: FakeModel(fail=RuntimeError("boom")), cuda_check=lambda: True)
    with pytest.raises(RuntimeError, match="boom"):
        transcriber.transcribe(Path("x.webm"))

@pytest.mark.slow
def test_real_model_on_sample():
    sample = os.environ.get("FEEDBACK_COLLECTOR_SAMPLE") or pytest.skip("set FEEDBACK_COLLECTOR_SAMPLE to a narrated recording")
    transcript = WhisperTranscriber().transcribe(Path(sample))
    assert transcript.segments
    assert [s.start for s in transcript.segments] == sorted(s.start for s in transcript.segments)
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_cuda_libs.py tests/test_transcription.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `cuda_libs.py` and `transcription.py`.
- [ ] **Step 4: Install the GPU extra and check CUDA.** Run: `.venv/bin/pip install -e ".[dev,gpu]" && .venv/bin/python -c "from feedback_collector.cuda_libs import cuda_ready; print(cuda_ready())"`. Expected: `True`.
- [ ] **Step 5: README GPU section.** Cover:
  - `pip install -e ".[gpu]"`; no CUDA toolkit or sudo is needed;
  - the automatic CPU fallback, and `--device cpu`;
  - the one-time model download to `~/.cache/huggingface`.
- [ ] **Step 6: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 8: Pipeline and CLI

**Files:**
- Create: `src/feedback_collector/pipeline.py`, `src/feedback_collector/cli.py`, `src/feedback_collector/__main__.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`
- Modify: `tests/conftest.py` (add `FakeTranscriber` and `NARRATION`), `README.md` (add Usage, Output and the Claude Code hand-off)

**Interfaces:**
- Consumes:
  - `probe`, `gray_samples`, `extract_frames`, `MediaError` from `media`;
  - `detect_changes`, `SAMPLE_FPS`, `SAMPLE_WIDTH` from `changes`;
  - `build_sections` from `sections`;
  - `find_candidates`, `select_frames`, `frame_budget` from `selection`;
  - `render_markdown`, `render_json`, `render_vtt`, `handoff_prompt` from `report`;
  - `Transcriber`, `WhisperTranscriber`, `DEFAULT_MODEL` from `transcription`;
  - `filename_stamp`, `parse_time` from `timefmt`.
- Produces (`feedback_collector.pipeline`):
  - `OWN_FILES = ("report.md", "report.json", "transcript.vtt", "frames")`
  - `class OutputExistsError(Exception)`
  - `@dataclass(frozen=True) Options(out_dir: Path, max_frames: int | None = None, frame_width: int = 1600, every: float | None = None, force: bool = False)`
  - `default_out_dir(video: Path) -> Path`: returns `video.parent / f"{video.stem}-feedback"`.
  - `prepare_out_dir(out_dir: Path, force: bool) -> None`:
    - when any `OWN_FILES` entry exists, without `force` it raises `OutputExistsError(f"{out_dir} already has a report; pass --force to replace it")`, and with `force` it removes only those entries;
    - then it runs `mkdir(parents=True, exist_ok=True)`.
  - `process(video: Path, options: Options, transcriber: Transcriber) -> Report`
  - `grab_frames(video: Path, times: Sequence[float], out_dir: Path, width: int) -> list[Path]`: writes `out_dir / f"{filename_stamp(t)}_extra.png"` for each time.
- Produces (`feedback_collector.cli`):
  - `build_parser() -> argparse.ArgumentParser`
  - `main(argv: Sequence[str] | None = None, transcriber_factory: Callable[[argparse.Namespace], Transcriber] | None = None) -> int`

What `process` does, in order:
1. `info = probe(video)`. Without audio, raise `MediaError(f"{video.name} has no audio track, so there is no narration to transcribe. Record with the microphone on.")`.
2. `prepare_out_dir(options.out_dir, options.force)`.
3. `transcript = transcriber.transcribe(video, offset=info.audio_offset)`. When it has no segments, log a warning `No speech detected; the report will only contain screen changes`.
4. `changes = detect_changes(gray_samples(video, info, SAMPLE_FPS, SAMPLE_WIDTH))`, then `sections = build_sections(...)` and `candidates = find_candidates(sections, changes, info.duration, options.every)`.
5. With `onsets = [c.onset for c in changes]`:
   - `cap = frame_budget(info.duration, options.max_frames)`;
   - `chosen = select_frames(candidates, onsets, cap)`;
   - `candidate_count = len(select_frames(candidates, onsets, None))`.
6. For each chosen candidate, clamp `t = min(max(c.time, 0.0), info.last_frame_time)` and build `FrameRef(t, reason, label, f"frames/{filename_stamp(t)}_{reason}.png")`, skipping a path that's already taken.
7. `extract_frames` into `out_dir`.
8. Write `transcript.vtt`, `report.json` and `report.md` as UTF-8, and return the `Report`.

CLI:
```
feedback-collector process VIDEO [--out DIR] [--force] [--model NAME] [--device {auto,cuda,cpu}] [--language CODE]
                                 [--vocab TERMS] [--max-frames N] [--frame-width PX] [--every SECONDS]
feedback-collector frames VIDEO --at TIMES [--out DIR] [--frame-width PX]
```
- **Defaults:** `--model large-v3-turbo`, `--device auto`, `--frame-width 1600`. `frames --out` defaults to `default_out_dir(video) / "extra"`.
- **Argument types:** `--at` is a comma-separated list of `parse_time` values. `--max-frames`, `--frame-width` and `--every` must be positive. Bad values are argparse errors (exit 2).
- **`main` setup:**
  - calls `logging.basicConfig(level=logging.INFO, format="%(message)s")`;
  - resolves `VIDEO` to an absolute path;
  - builds `WhisperTranscriber(args.model, args.device, args.language, hotwords=args.vocab)` unless a factory is passed;
  - catches `MediaError` and `OutputExistsError`, prints `error: {error}` to stderr, and returns 1.
- **Output:** on success, `process` prints:
```
Report: {out_dir}/report.md
{n} screenshots in {s} sections (cap {cap}, {candidates} candidates)

Paste into Claude Code:
{handoff_prompt(out_dir / "report.md")}
```
  `frames` prints each written path on its own line. `__main__.py` is `raise SystemExit(main())`.

Add to `conftest.py`:
- `FakeTranscriber(segments)`: its `transcribe` returns `Transcript(tuple(segments), "en", "fake", "cpu")`.
- `NARRATION = [Segment(0.2, 1.6, "This page loads fine.", (Word(0.2, 0.4, "This"), Word(0.4, 0.7, "page"), Word(0.7, 1.1, "loads"), Word(1.1, 1.6, "fine."))), Segment(5.0, 5.8, "Now the box shows.")]`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_pipeline.py`:
```python
def test_process_writes_interleaved_report(box_clip, tmp_path):
    out = tmp_path / "out"
    report = process(box_clip, Options(out_dir=out), FakeTranscriber(NARRATION))
    assert [(round(f.time, 1), f.reason) for f in report.frames] == [(0.2, "speech"), (3.5, "silent-change"), (5.0, "speech")]
    text = (out / "report.md").read_text()
    assert "> This page loads fine." in text and "(no narration)" in text
    assert all((out / path).exists() for path in re.findall(r"\]\((frames/[^)]+)\)", text))
    assert json.loads((out / "report.json").read_text())["frames"]["selected"] == 3
    assert (out / "transcript.vtt").read_text().startswith("WEBVTT")

def test_process_without_speech_keeps_screen_changes(box_clip, tmp_path):
    report = process(box_clip, Options(out_dir=tmp_path / "out"), FakeTranscriber([]))
    assert [f.reason for f in report.frames] == ["silent-change"]
    assert "_No speech was detected in this recording._" in (tmp_path / "out" / "report.md").read_text()

def test_process_rejects_video_without_audio(silent_clip, tmp_path):
    with pytest.raises(MediaError, match="no audio track"):
        process(silent_clip, Options(out_dir=tmp_path / "out"), FakeTranscriber([]))

def test_existing_report_is_refused(box_clip, tmp_path):
    out = tmp_path / "out"; out.mkdir(); (out / "report.md").write_text("mine")
    with pytest.raises(OutputExistsError, match="--force"):
        process(box_clip, Options(out_dir=out), FakeTranscriber(NARRATION))

def test_force_replaces_only_own_files(box_clip, tmp_path):
    out = tmp_path / "out"; (out / "frames").mkdir(parents=True)
    (out / "report.md").write_text("old"); (out / "notes.txt").write_text("keep me"); (out / "frames" / "stale.png").write_bytes(b"x")
    process(box_clip, Options(out_dir=out, force=True), FakeTranscriber(NARRATION))
    assert (out / "notes.txt").read_text() == "keep me"
    assert not (out / "frames" / "stale.png").exists() and (out / "report.md").read_text() != "old"

def test_grab_frames_names_files_by_time(box_clip, tmp_path):
    paths = grab_frames(box_clip, [1.0, 3.5], tmp_path / "extra", width=160)
    assert [p.name for p in paths] == ["00-00-01.0_extra.png", "00-00-03.5_extra.png"] and all(p.exists() for p in paths)
```
In `tests/test_cli.py`, with `fake = lambda args: FakeTranscriber(NARRATION)`:
```python
def test_defaults():
    args = build_parser().parse_args(["process", "v.webm"])
    assert (args.model, args.device, args.frame_width, args.max_frames, args.every) == ("large-v3-turbo", "auto", 1600, None, None)

def test_process_command_prints_report_and_prompt(box_clip, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["process", str(box_clip), "--out", str(out)], transcriber_factory=fake) == 0
    stdout = capsys.readouterr().out
    assert f"Report: {out / 'report.md'}" in stdout
    assert f'Read "{out / "report.md"}" and open every screenshot it lists.' in stdout

def test_default_out_dir_is_next_to_video(box_clip, tmp_path):
    video = tmp_path / "rec.webm"; shutil.copy(box_clip, video)
    assert main(["process", str(video)], transcriber_factory=fake) == 0
    assert (tmp_path / "rec-feedback" / "report.md").exists()

def test_frames_command(box_clip, tmp_path, capsys):
    assert main(["frames", str(box_clip), "--at", "0:01,3.5", "--out", str(tmp_path / "extra")]) == 0
    assert capsys.readouterr().out.split() == [str(tmp_path / "extra" / "00-00-01.0_extra.png"), str(tmp_path / "extra" / "00-00-03.5_extra.png")]

def test_errors_exit_with_message(silent_clip, tmp_path, capsys):
    assert main(["process", str(silent_clip), "--out", str(tmp_path / "o")], transcriber_factory=fake) == 1
    assert "error:" in capsys.readouterr().err

def test_bad_time_is_a_usage_error(box_clip):
    with pytest.raises(SystemExit) as exit_info:
        main(["frames", str(box_clip), "--at", "soon"])
    assert exit_info.value.code == 2
```
- [ ] **Step 2: Run them.** Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_cli.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `pipeline.py`, `cli.py` and `__main__.py`.
- [ ] **Step 4: README.** Add:
  - Usage: both subcommands with every option;
  - the output folder layout and a sample `report.md`;
  - the Claude Code hand-off (paste the printed prompt; Claude can run `frames` itself).
- [ ] **Step 5: Run the suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.

---

### Task 9: Real recordings, tuning and docs

**Files:**
- Modify: `README.md`. Change `src/feedback_collector/changes.py` / `selection.py` constants (with their tests) only if tuning calls for it.

`$SCRATCH` = `/tmp/claude-1000/-home-richard-code-personal-feedback-collector/4dec59b1-0e49-4af5-b3d8-5445d362d2ba/scratchpad`. Outputs go there, not into `~/Videos`.

- [ ] **Step 1: Slow test on GPU.** Run: `FEEDBACK_COLLECTOR_SAMPLE=/home/richard/Videos/Kooha/Kooha-2026-05-05-10-56-16.webm .venv/bin/pytest -m slow -v`. Expected: PASS. The first run downloads the model.
- [ ] **Step 2: Kooha recording** (container duration `N/A`). Run: `time .venv/bin/feedback-collector process /home/richard/Videos/Kooha/Kooha-2026-05-05-10-56-16.webm --out $SCRATCH/kooha`. Expected: exit 0, and the `report.md` header says `(cuda)`.
- [ ] **Step 3: 21-minute recording** (variable frame rate). Run: `time .venv/bin/feedback-collector process "/home/richard/Videos/Screencast_05_05_2026_09:08:00_AM.webm" --out $SCRATCH/long`. Expected:
  - exit 0 in about 2-3 minutes;
  - the stdout line reads `cap 126`, with selected ≤ 126.
- [ ] **Step 4: Inspect.** Read both reports and view about 10 frames, especially change frames after scrolling, plus a few speech frames against their quotes.
  - **Mid-scroll frames:** raise `MIN_QUIET` to 3.
  - **Obvious UI changes missing:** lower `CHANGED_FRACTION`.
  - **Trivial changes flooding the report:** raise it.
  - **Pointing words misfiring:** adjust `POINTING_WORDS` / `POINTING_PHRASES`.

  Update only the tests tied to a changed constant, then run `.venv/bin/pytest -v`.
- [ ] **Step 5: README.**
  - **Recording tips:** Kooha mic on and desktop audio off; point while saying "this"/"here"; pause between issues; use `--vocab` for product names.
  - **Troubleshooting:** the CPU-fallback warning, the first-run download, the "no audio track" error.
  - **Performance:** the runtimes measured in Steps 2-3.
- [ ] **Step 6: Final suite.** Run: `.venv/bin/pytest -v`. Expected: all pass.
