"""Command-line entry point: `feedback-collector process` and `feedback-collector frames`."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from feedback_collector.media import MediaError
from feedback_collector.pipeline import Options, OutputExistsError, default_out_dir, grab_frames, process
from feedback_collector.report import handoff_prompt, visible_sections
from feedback_collector.timefmt import parse_time
from feedback_collector.transcription import DEFAULT_MODEL, ModelDownloadError, Transcriber, WhisperTranscriber


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def _times(text: str) -> list[float]:
    try:
        return [parse_time(part) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedback-collector",
        description="Turn a narrated screen recording into a transcript + screenshot report for Claude Code.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("process", help="transcribe a recording and write report.md with screenshots")
    run.add_argument("video", type=Path)
    run.add_argument("--out", type=Path, help="output folder (default: <video dir>/<video name>-feedback)")
    run.add_argument("--force", action="store_true", help="replace an existing report in the output folder")
    run.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model (default: {DEFAULT_MODEL})")
    run.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--language", help="spoken language code, e.g. en (default: detect)")
    run.add_argument("--vocab", help='names the speech model should spell right, e.g. "Stripe, SKU, Checkout"')
    run.add_argument("--max-frames", type=_positive_int, help="screenshot cap (default: 6 per minute, 20-150)")
    run.add_argument("--frame-width", type=_positive_int, default=1600, help="max screenshot width in px")
    run.add_argument("--every", type=_positive_float, help="also take a screenshot every N seconds")

    grab = commands.add_parser("frames", help="grab extra screenshots at given times")
    grab.add_argument("video", type=Path)
    grab.add_argument("--at", type=_times, required=True, help="comma-separated times, e.g. 3:12.5,4:01")
    grab.add_argument("--out", type=Path, help="output folder (default: <video name>-feedback/extra)")
    grab.add_argument("--frame-width", type=_positive_int, default=1600, help="max screenshot width in px")
    return parser


def _transcriber(args: argparse.Namespace) -> Transcriber:
    return WhisperTranscriber(args.model, args.device, args.language, hotwords=args.vocab)


def main(
    argv: Sequence[str] | None = None,
    transcriber_factory: Callable[[argparse.Namespace], Transcriber] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    video = args.video.expanduser().resolve()
    try:
        if args.command == "frames":
            out_dir = (args.out or default_out_dir(video) / "extra").expanduser().resolve()
            for path in grab_frames(video, args.at, out_dir, args.frame_width):
                print(path)
            return 0

        options = Options(
            out_dir=(args.out or default_out_dir(video)).expanduser().resolve(),
            max_frames=args.max_frames,
            frame_width=args.frame_width,
            every=args.every,
            force=args.force,
        )
        report = process(video, options, (transcriber_factory or _transcriber)(args))
    except (MediaError, OutputExistsError, ModelDownloadError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    report_path = report.out_dir / "report.md"
    print(f"Report: {report_path}")
    print(
        f"{len(report.frames)} screenshots in {len(visible_sections(report))} sections "
        f"(cap {report.cap}, {report.candidate_count} candidates)"
    )
    print()
    print("Paste into Claude Code:")
    print(handoff_prompt(report_path))
    return 0
