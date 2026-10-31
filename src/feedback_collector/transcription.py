"""Local speech-to-text with faster-whisper."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from huggingface_hub.errors import LocalEntryNotFoundError

from feedback_collector.cuda_libs import cuda_ready
from feedback_collector.models import Segment, Transcript, Word

DEFAULT_MODEL = "large-v3-turbo"
COMPUTE_TYPES = {"cuda": "float16", "cpu": "int8"}
SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    pass


class Transcriber(Protocol):
    def transcribe(self, video: Path, offset: float = 0.0) -> Transcript: ...


def locate_whisper_model(name: str, local_files_only: bool) -> str:
    if Path(name).expanduser().is_dir():
        return str(Path(name).expanduser())
    from faster_whisper.utils import download_model

    return download_model(name, local_files_only=local_files_only)


def load_whisper_model(path: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(path, device=device, compute_type=compute_type)


def load_audio(video: Path) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(video), sampling_rate=SAMPLE_RATE)


def find_speech(audio: np.ndarray) -> list[float]:
    """Speech stretches as a flat [start, end, start, end, ...] list in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunks = get_speech_timestamps(audio, VadOptions())
    return [round(chunk[edge] / SAMPLE_RATE, 3) for chunk in chunks for edge in ("start", "end")]


def _convert(segment: Any, offset: float) -> Segment:
    words = tuple(
        Word(word.start + offset, word.end + offset, word.word.strip()) for word in (segment.words or ())
    )
    return Segment(segment.start + offset, segment.end + offset, segment.text.strip(), words)


class WhisperTranscriber:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        language: str | None = None,
        hotwords: str | None = None,
        model_factory: Callable[[str, str, str], Any] = load_whisper_model,
        cuda_check: Callable[[], bool] = cuda_ready,
        model_locator: Callable[[str, bool], str] = locate_whisper_model,
        audio_loader: Callable[[Path], np.ndarray] = load_audio,
        speech_finder: Callable[[np.ndarray], list[float]] = find_speech,
    ) -> None:
        self.model = model
        self.device = device
        self.language = language
        self.hotwords = hotwords
        self.model_factory = model_factory
        self.cuda_check = cuda_check
        self.model_locator = model_locator
        self.audio_loader = audio_loader
        self.speech_finder = speech_finder

    def _devices(self) -> list[str]:
        if self.device == "auto":
            return ["cuda", "cpu"] if self.cuda_check() else ["cpu"]
        return [self.device]

    def _locate(self) -> str:
        try:
            return self.model_locator(self.model, True)
        except LocalEntryNotFoundError:
            logger.info("Downloading speech model %s (one time)…", self.model)
        try:
            return self.model_locator(self.model, False)
        except Exception as error:
            raise ModelDownloadError(
                f"could not download speech model {self.model}: {error}. "
                "Check your network connection and run again. On a slow or unreliable connection, "
                "use a smaller model (--model small or --model base), or download the model folder "
                "with a resumable tool and pass it with --model /path/to/model-folder."
            ) from error

    def transcribe(self, video: Path, offset: float = 0.0) -> Transcript:
        audio = self.audio_loader(video)
        clips = self.speech_finder(audio)
        if not clips:
            return Transcript((), self.language or "", self.model, "none")

        # Located once, outside the device fallback, so a network error is never mistaken for a CUDA one.
        model_path = self._locate()
        devices = self._devices()
        for device in devices:
            try:
                model = self.model_factory(model_path, device, COMPUTE_TYPES[device])
                # Decoding each speech stretch on its own keeps word times true to the recording;
                # faster-whisper's vad_filter glues speech together and can drag words across silences.
                segments, info = model.transcribe(
                    audio,
                    language=self.language,
                    hotwords=self.hotwords,
                    word_timestamps=True,
                    vad_filter=False,
                    clip_timestamps=clips,
                )
                # A missing cuBLAS only surfaces once segments are decoded, so convert inside the try.
                converted = tuple(_convert(segment, offset) for segment in segments)
            except (RuntimeError, ValueError, OSError) as error:
                if device == devices[-1]:
                    raise
                logger.warning("CUDA transcription failed (%s); retrying on CPU", error)
                continue
            return Transcript(converted, info.language, self.model, device)
        raise RuntimeError("no device available for transcription")
