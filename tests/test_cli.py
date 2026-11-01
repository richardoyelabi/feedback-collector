import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import NARRATION, FakeTranscriber
from feedback_collector.cli import build_parser, main
from feedback_collector.transcription import CudaUnavailableError, ModelDownloadError


def fake(args):
    return FakeTranscriber(NARRATION)


def test_defaults():
    args = build_parser().parse_args(["process", "v.webm"])
    assert (args.model, args.device, args.frame_width, args.max_frames, args.every) == (
        "large-v3-turbo", "auto", 1600, None, None,
    )


def test_process_command_prints_report_and_prompt(box_clip, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["process", str(box_clip), "--out", str(out)], transcriber_factory=fake) == 0
    stdout = capsys.readouterr().out
    assert f"Report: {out / 'report.md'}" in stdout
    assert f'Read "{out / "report.md"}" and open every screenshot it lists.' in stdout


def test_summary_line_counts_in_the_singular(box_clip, tmp_path, capsys):
    code = main(["process", str(box_clip), "--out", str(tmp_path / "out")], transcriber_factory=lambda args: FakeTranscriber(()))
    assert code == 0
    assert "1 screenshot in 1 section (cap 20, 1 candidate)" in capsys.readouterr().out


def test_default_out_dir_is_next_to_video(box_clip, tmp_path):
    video = tmp_path / "rec.webm"
    shutil.copy(box_clip, video)
    assert main(["process", str(video)], transcriber_factory=fake) == 0
    assert (tmp_path / "rec-feedback" / "report.md").exists()


def test_frames_command(box_clip, tmp_path, capsys):
    assert main(["frames", str(box_clip), "--at", "0:01,3.5", "--out", str(tmp_path / "extra")]) == 0
    assert capsys.readouterr().out.split() == [
        str(tmp_path / "extra" / "00-00-01.0_extra.png"),
        str(tmp_path / "extra" / "00-00-03.5_extra.png"),
    ]


def test_report_hint_runs_from_anywhere_without_the_tool_on_path(box_clip, tmp_path):
    out = tmp_path / "out"
    assert main(["process", str(box_clip), "--out", str(out)], transcriber_factory=fake) == 0
    hint = re.search(r"Need another moment\? Run `(.+)`\.", (out / "report.md").read_text()).group(1)
    only_ffmpeg = {"PATH": str(Path(shutil.which("ffmpeg")).parent)}
    subprocess.run(hint, shell=True, cwd=tmp_path, env=only_ffmpeg, check=True, capture_output=True)
    assert (out / "extra" / "00-03-12.5_extra.png").exists()


def test_errors_exit_with_message(silent_clip, tmp_path, capsys):
    assert main(["process", str(silent_clip), "--out", str(tmp_path / "o")], transcriber_factory=fake) == 1
    assert "error:" in capsys.readouterr().err


class OfflineTranscriber:
    def transcribe(self, video, offset=0.0):
        raise ModelDownloadError("could not download speech model large-v3-turbo: network down")


def test_model_download_failure_exits_with_message(box_clip, tmp_path, capsys):
    code = main(["process", str(box_clip), "--out", str(tmp_path / "o")], transcriber_factory=lambda args: OfflineTranscriber())
    assert code == 1
    assert "error: could not download speech model large-v3-turbo" in capsys.readouterr().err


class NoGpuTranscriber:
    def transcribe(self, video, offset=0.0):
        raise CudaUnavailableError("--device cuda was requested but no usable CUDA GPU was found")


def test_unusable_gpu_exits_with_message(box_clip, tmp_path, capsys):
    code = main(["process", str(box_clip), "--out", str(tmp_path / "o")], transcriber_factory=lambda args: NoGpuTranscriber())
    assert code == 1
    assert "error: --device cuda was requested" in capsys.readouterr().err


def test_bad_time_is_a_usage_error(box_clip):
    with pytest.raises(SystemExit) as exit_info:
        main(["frames", str(box_clip), "--at", "soon"])
    assert exit_info.value.code == 2
