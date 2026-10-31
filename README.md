# feedback-collector

Turn a narrated screen recording into something Claude Code can read. You record yourself testing a web app while talking through what's wrong. `feedback-collector` transcribes your narration locally and grabs screenshots at the moments that matter. It then writes one Markdown report where each stretch of narration sits next to what was on screen while you said it.

Everything runs on your machine; no audio or video is uploaded anywhere.

The design and the reasoning behind it are in [docs/superpowers/specs/2026-10-30-feedback-collector-design.md](docs/superpowers/specs/2026-10-30-feedback-collector-design.md).

## Install

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on your `PATH` (`sudo apt install ffmpeg`).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[gpu]"     # drop [gpu] on machines without an NVIDIA GPU
```

## Usage

### Make a report

```bash
.venv/bin/feedback-collector process ~/Videos/rec.webm
```

This writes `~/Videos/rec-feedback/` and prints a prompt to paste into Claude Code.

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `<video dir>/<video name>-feedback` | Where the report goes |
| `--force` | off | Replace an existing report. Only `report.md`, `report.json`, `transcript.vtt` and `frames/` are removed; other files in the folder stay |
| `--vocab "Stripe, SKU"` | none | Product or component names the speech model should spell correctly |
| `--language en` | auto-detect | Spoken language |
| `--model NAME` | `large-v3-turbo` | Whisper model; `large-v3` is slower and slightly more accurate |
| `--device auto\|cuda\|cpu` | `auto` | See [GPU transcription](#gpu-transcription) |
| `--max-frames N` | 6 per minute (min 20, max 150) | Screenshot cap |
| `--frame-width PX` | `1600` | Max screenshot width; frames are never upscaled. Raise it (e.g. `2560`) if small text is unreadable |
| `--every SECONDS` | off | Also take a screenshot every N seconds |

### Grab extra screenshots

```bash
.venv/bin/feedback-collector frames ~/Videos/rec.webm --at 3:12.5,4:01
```

This writes `~/Videos/rec-feedback/extra/00-03-12.5_extra.png` and similar, and prints each path. Times can be seconds (`192.5`), `M:SS` or `H:MM:SS`. The report tells Claude about this command, so Claude can run it itself when it needs a moment the report skipped.

## What you get

```
rec-feedback/
├── report.md         # hand this to Claude Code
├── report.json       # the same content, machine-readable
├── transcript.vtt    # the raw transcript (also works as subtitles in video players)
└── frames/
    ├── 00-00-12.4_speech.png
    ├── 00-00-15.1_pointer.png
    └── 00-00-30.2_silent-change.png
```

`report.md` walks through the recording in order. Each section quotes what you said, with the screenshots of what was on screen at the time:

```markdown
## 00:12–00:27
> So this button is confusing, I expected it to submit the form but nothing happened.

![00:12.4 speech start](frames/00-00-12.4_speech.png)
![00:15.1 "this"](frames/00-00-15.1_pointer.png)
![00:19.0 screen changed](frames/00-00-19.0_change.png)
```

Screenshots are taken at:
- the start of each stretch of narration;
- the moment you say a pointing word ("this", "here", "these", "right there");
- each time the screen changes, once it has settled (after scrolling or an animation finishes);
- every N seconds, if you ask for it with `--every`.

Near-duplicates within a second of each other are dropped, unless the screen changed between them. When there are more moments than the cap allows, the first frame of each narrated section wins, then changes and pointing words, then silent stretches. The header always says how many screenshots were kept out of how many candidates.

## Hand it to Claude Code

Open Claude Code in your web app's repository and paste the prompt `process` printed. It looks like this:

```
Read "/home/you/Videos/rec-feedback/report.md" and open every screenshot it lists. It is narrated feedback I recorded while testing this app. For each issue I raise, find the code responsible and propose a fix. Where my words and the screenshots disagree, trust the screenshots.
```

## Recording tips

- **Record your microphone.** In Kooha, switch the microphone on. Also switch desktop audio off, or app sounds and notifications get transcribed as if you said them. GNOME's built-in recorder (Ctrl+Alt+Shift+R) records no audio at all, so `process` will refuse its files.
- **Point while you talk.** Saying "this", "here" or "right there" while the cursor is on the thing gets you a screenshot at that exact moment.
- **Pause between issues.** A pause of 1.5 seconds or more starts a new section in the report, so each problem gets its own heading.
- **Spell out your vocabulary.** Pass product and component names with `--vocab` so the transcript gets them right.
- **Mind what's on screen.** Screenshots capture everything visible: notifications, autofill suggestions with your email or phone number, other tabs. They stay on your machine, but Claude sees them when you share the report.

## GPU transcription

Transcription uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper). On a machine with an NVIDIA GPU, install the `gpu` extra:

```bash
.venv/bin/pip install -e ".[gpu]"
```

This adds NVIDIA's cuBLAS for CUDA 12 as a pip package. `feedback-collector` loads it itself at startup, so you don't need a system CUDA toolkit, sudo, or `LD_LIBRARY_PATH`. Only the NVIDIA driver must be installed.

- With the default `--device auto`, a GPU problem prints a warning and the whole transcription reruns on the CPU (int8). This is slower, but you still get a report.
- `--device cpu` skips the GPU entirely. `--device cuda` insists on the GPU and stops with an error if it can't be used, instead of falling back.
- The first run downloads the speech model (about 1.6 GB for `large-v3-turbo`) into `~/.cache/huggingface`. Later runs load it from there and never touch the network.

## Performance

Measured on an RTX 4070 Laptop GPU with `--model small`:

| Recording | Result | Time |
|---|---|---|
| 5 minutes, narrated, 2560×1600 | 11 sections, 30 screenshots | 7 s |
| 21 minutes, no narration, 2560×1600 | 126 screenshots | 22 s |

Finding screen changes and extracting screenshots each take one pass over the video, about 10 s for 21 minutes of 2560×1600 footage. Recordings without speech skip the speech model entirely.

## Troubleshooting

**The model download keeps failing.** On a slow or unstable connection the one-time download can break partway. huggingface_hub discards partial files, so each rerun starts from zero. You have two options:

- Use a smaller model: `--model small` (about 480 MB) or `--model base` (about 145 MB), instead of the default `large-v3-turbo` (about 1.6 GB).
- Download the model folder yourself with a tool that resumes, then pass the folder to `--model`:

  ```bash
  dest=~/.cache/feedback-collector/models/faster-whisper-small
  mkdir -p "$dest"
  for f in config.json tokenizer.json vocabulary.txt model.bin; do
    until curl -L -C - -o "$dest/$f" "https://huggingface.co/Systran/faster-whisper-small/resolve/main/$f"; do sleep 5; done
  done
  .venv/bin/feedback-collector process rec.webm --model "$dest"
  ```

**"has no audio track".** The recording has no sound at all. Record again with the microphone on.

**"CUDA transcription failed … retrying on CPU".** The GPU couldn't be used (driver or library problem), so the run continued on the CPU. Check that the `gpu` extra is installed and `nvidia-smi` works.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests build tiny synthetic clips with ffmpeg, so `ffmpeg` and `ffprobe` must be on your `PATH`.
