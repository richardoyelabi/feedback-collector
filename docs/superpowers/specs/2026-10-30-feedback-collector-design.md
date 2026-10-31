# feedback-collector design

## Why this exists

Feedback gets recorded as screen videos of someone testing a web app while narrating ("this button is confusing", "I expected X but got Y"). Claude Code can read text and images but not video or audio. Each recording therefore has to become a timestamped transcript plus well-chosen screenshots, combined into one file that interleaves the two.

No mature local tool did this for existing recordings when this was written (October 2026):
- The claude-video `/watch` plugin keeps frames and transcript as separate in-session lists.
- The MarkuprPlus CLI caps a report at 20 frames chosen at speech pauses.
- Gripe only records inside Chrome.
- Jam is cloud-only, with a 5-minute free cap.

## Intent

Turn one narrated screen recording (for example a Kooha WebM/MP4 of a web app) into a folder Claude Code can read in one pass: a Markdown report where each stretch of narration sits next to screenshots of what was on screen while it was said.

## Goals

- `feedback-collector process ~/Videos/rec.webm` produces `~/Videos/rec-feedback/` containing `report.md`, `report.json`, `transcript.vtt` and `frames/`.
- `feedback-collector frames ~/Videos/rec.webm --at 3:12.5,4:01 [--out DIR]` grabs extra screenshots on demand. Claude can run it itself, and the report header tells it how.
- Accurate local transcription on an NVIDIA GPU, falling back to CPU automatically.
- Screenshots at the right moments: when each narration chunk starts, when the narrator says a pointing word ("this", "here", "these", "this one", "right there"), and once the UI settles after a change. Optionally also every N seconds.
- **Budget:** about 6 screenshots per minute, with a floor of 20 and a ceiling of 150. `--max-frames` overrides. At least one per narration section comes first, then frames showing the result of an action or what was pointed at, then the rest.
- The report is written for an AI reader: chronological sections, quoted narration with times, relative image paths, a "how to read this" header, and counts of selected vs. candidate frames so a capped subset never looks like full coverage.
- Prints a ready-to-paste Claude Code prompt pointing at the report.

## Non-goals (v1)

- Recording (the screen recorder keeps doing that).
- Turning feedback into tasks (Claude Code does that from the report).
- Console or network capture.
- Multiple speakers.
- A GUI or editor.
- Contact sheets.
- Watch-folder mode.

## Anti-goals

- Audio or video leaving the machine.
- Silently dropped frames.
- Overwriting a hand-edited report: an existing report in the output dir is refused without `--force`.
- An install that needs a system CUDA toolkit or sudo.

## Constraints

- Python ≥3.11.
- ffmpeg/ffprobe on PATH.
- Test-driven with pytest.
- Readable code with sparse comments and double quotes.
- A README plus this design doc.
- No network use after the one-time model download.
- PyAV is capped below 19: faster-whisper 1.2.1 still passes the `av.open(metadata_errors=…)` argument that PyAV 19 removed. Lift the cap once a faster-whisper release includes the upstream fix.

## Decisions

1. **Packaging.** A Python package with a `src/` layout, installed with venv + pip, exposing one console script with two subcommands, `process` and `frames`.
2. **Transcription.**
   - faster-whisper with default model `large-v3-turbo`; `--model large-v3` gives maximum accuracy.
   - `--vocab "Term, Term"` goes into `hotwords`, which is re-sent with every 30 s window. `initial_prompt` stops influencing the output after about the first minute.
   - Speech is found with faster-whisper's bundled Silero VAD, and each speech stretch is decoded on its own via `clip_timestamps`, with word timestamps on.
     - This keeps word times true to the recording. faster-whisper's own `vad_filter` glues speech together, and that moved the first word after a silence back across it.
     - A recording without speech skips the model entirely.
   - The model loads with `local_files_only=True` first, so routine runs never touch the network.
   - Transcript times are shifted by the audio/video `start_time` difference.
3. **GPU.**
   - Only `nvidia-cublas-cu12` is needed: CTranslate2's Linux wheels are built against CUDA 12.8 with cuDNN off.
   - `nvidia/cublas/lib/libcublas.so.12` is preloaded by path before faster-whisper is used, so nobody touches `LD_LIBRARY_PATH`.
   - `--device auto` retries the whole transcription on CPU int8 if CUDA fails. A missing cuBLAS only errors on the first encode, so the retry covers consuming the segments, not just loading the model.
   - `--device cuda` runs the same readiness check, which also preloads cuBLAS, and stops with a clear error if no GPU is usable.
4. **UI-change detection.**
   - ffmpeg decodes the video to a 2 fps, 160 px wide grayscale pipe.
   - A change means at least 0.3% of pixels differ by more than 16 levels. This catches toasts, inline errors and menus, while cursor movement stays below it.
   - Consecutive changing samples (scrolling, video, animation) form one burst, captured at the first quiet sample after it, so the frame shows the settled UI.
5. **Sections.** Segments are first split wherever their word timings show a pause of 1.5 s or more, since Whisper can return one segment spanning two remarks. Consecutive segments then merge while the gap between them is under 1.5 s. A section splits at the next segment boundary once it passes about 30 s. Silent stretches that contain selected screen changes become "(no narration)" sections.
6. **Frame selection.**
   - Candidates are speech starts, pointing words, settled screen changes, silent-stretch changes and the optional `--every N`.
   - Selection works by priority with suppression: a candidate within 1 s of one already chosen is dropped, unless a screen change lies between them.
   - Priority 1 is the first frame of every section; 2 is changes during narration plus pointing words; 3 is silent-stretch changes; 4 is every-N.
   - Within a level, frames are spread evenly across the timeline.
7. **Images.** PNG, at most 1600 px wide (a 2560×1600 screen becomes 1600×1000, about 2.1k Claude tokens), never upscaled. `--frame-width 2560` helps with tiny text.
8. **Extraction.**
   - Every frame comes out of a single ffmpeg pass. Kooha's VP8 output has almost no keyframes, so per-frame seeking would take minutes on long recordings.
   - The chain is `fps=10,select='eq(n,…)+…',scale=…`. The fps filter puts frames on a 0.1 s grid counted from the first frame and repeats the last frame through gaps. So an anchor gets the screen as it was at that moment (rounded up to the next 0.1 s), even on variable-frame-rate recordings that emit no frames while the screen is still. Selecting by grid index avoids every dependence on time bases, container start offsets and ffmpeg's rounded time printouts.
   - Anchors are clamped to the last grid slot, and every expected PNG is checked.
9. **Probing.** Duration falls back to the last packet time when the container reports `N/A`. A file with no audio stream stops `process` with a clear message.
10. **Output.**
    - The default is `<video dir>/<stem>-feedback/`, and `--out` overrides it. Frames are named like `frames/00-01-23.4_speech.png`.
    - `--force` replaces only the tool's own files (`report.md`, `report.json`, `transcript.vtt`, `frames/`) and keeps anything else in the folder.
    - An existing report is refused before any work starts. With `--force`, it's removed only once transcription and frame selection have succeeded, so a failed run leaves it intact.

## `report.md` shape

```markdown
# Feedback walkthrough: rec-2026-10-30.webm
`6:42` · `41 sections` · `40 screenshots (cap 40) of 57 candidates` · transcript: faster-whisper large-v3-turbo (cuda), **not reviewed by a human**

**How to read this:** sections are in recording order. Open every screenshot (paths are relative to this file) before acting.
Narration is speech-to-text and may mishear names; where words and screenshots disagree, trust the screenshots.
Need another moment? Run `feedback-collector frames "/home/richard/Videos/rec-2026-10-30.webm" --at 03:12.5 --out "/home/richard/Videos/rec-2026-10-30-feedback/extra"`.

## 00:12–00:27
> So this button is confusing, I expected it to submit the form but nothing happened.

![00:12.4 speech start](frames/00-00-12.4_speech.png)
![00:15.1 "this"](frames/00-00-15.1_pointer.png)
![00:19.0 screen changed](frames/00-00-19.0_change.png)

## 00:27–00:41 (no narration)
![00:30.2 screen changed](frames/00-00-30.2_silent-change.png)
```

`report.json` holds the same data:
- `source`: path, duration, width, height.
- `transcription`: engine, model, device, language, `human_reviewed: false`.
- `frames`: selected, candidates, cap.
- `sections[]`: start, end, narrated, text, segments, and `frames[]` with time, reason, label and path.

## Pipeline

probe → transcribe → detect changes → build sections → select frames within the budget → extract frames in one pass → write `transcript.vtt`, `report.json`, `report.md` → print the hand-off prompt.

## Verification

- `pytest` is green. `FEEDBACK_COLLECTOR_SAMPLE=… pytest -m slow` runs real faster-whisper on the GPU.
- `process` on a Kooha recording with `N/A` duration and on a 21-minute variable-frame-rate recording:
  - the header says `cuda`;
  - the frame count is about 6/min;
  - frames show the settled UI rather than mid-scroll;
  - the run takes about a minute or two.
- A fresh 1-2 minute clip (mic on, desktop audio off), narrating three issues while pointing at them:
  - the report renders in VS Code's preview;
  - the times match the video;
  - the frames show the pointed-at element.
- In a Claude Code session opened on the web app's repo, the printed prompt leads Claude to open the frames, map each issue to code, and call `frames` for an extra moment.
