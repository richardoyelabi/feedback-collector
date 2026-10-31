import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from feedback_collector import transcription
from feedback_collector.models import Segment, Word
from feedback_collector.transcription import ModelDownloadError, WhisperTranscriber


AUDIO = np.zeros(16000 * 10, dtype=np.float32)


class FakeModel:
    def __init__(self, segments=(), fail=None):
        self.segments = list(segments)
        self.fail = fail
        self.audio = None
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs

        def generate():
            if self.fail:
                raise self.fail
            yield from self.segments

        return generate(), SimpleNamespace(language="en")


fake_segment = SimpleNamespace(
    start=1.0,
    end=2.0,
    text=" Hello there. ",
    words=[SimpleNamespace(start=1.0, end=1.4, word=" Hello"), SimpleNamespace(start=1.5, end=2.0, word=" there.")],
)


def returning(model):
    return lambda *args, **kwargs: model


def cached_locator(name, local_files_only):
    return f"/models/{name}"


def make(**overrides):
    options = {
        "device": "cpu",
        "cuda_check": lambda: False,
        "model_locator": cached_locator,
        "audio_loader": lambda video: AUDIO,
        "speech_finder": lambda audio: [1.0, 3.0],
    }
    options.update(overrides)
    return WhisperTranscriber(**options)


def test_segments_are_converted_and_shifted():
    transcript = make(model_factory=returning(FakeModel([fake_segment]))).transcribe(Path("x.webm"), offset=0.25)
    assert transcript.segments[0] == Segment(1.25, 2.25, "Hello there.", (Word(1.25, 1.65, "Hello"), Word(1.75, 2.25, "there.")))
    assert (transcript.language, transcript.model, transcript.device) == ("en", "large-v3-turbo", "cpu")


def test_options_reach_faster_whisper():
    model = FakeModel()
    make(language="en", hotwords="Stripe, SKU", model_factory=returning(model)).transcribe(Path("x.webm"))
    assert model.kwargs == {
        "language": "en",
        "hotwords": "Stripe, SKU",
        "word_timestamps": True,
        "vad_filter": False,
        "clip_timestamps": [1.0, 3.0],
    }


def test_speech_is_decoded_clip_by_clip():
    # Whisper's own VAD glues speech together, which drags words across long silences.
    model = FakeModel()
    make(speech_finder=lambda audio: [10.0, 11.0, 41.0, 42.0], model_factory=returning(model)).transcribe(Path("x.webm"))
    assert model.audio is AUDIO
    assert model.kwargs["clip_timestamps"] == [10.0, 11.0, 41.0, 42.0]
    assert model.kwargs["vad_filter"] is False


def test_no_speech_skips_the_model():
    loads = []
    transcriber = make(speech_finder=lambda audio: [], model_factory=lambda *args: loads.append(args) or FakeModel())
    transcript = transcriber.transcribe(Path("x.webm"))
    assert transcript.segments == ()
    assert loads == []


def test_find_speech_in_silence_is_empty():
    assert transcription.find_speech(np.zeros(16000 * 5, dtype=np.float32)) == []


def test_located_model_path_is_what_gets_loaded():
    loaded = []
    factory = lambda path, device, compute_type: loaded.append(path) or FakeModel()
    make(model_factory=factory).transcribe(Path("x.webm"))
    assert loaded == ["/models/large-v3-turbo"]


def test_model_loads_offline_first_then_downloads():
    calls = []

    def locator(name, local_files_only):
        calls.append(local_files_only)
        if local_files_only:
            raise transcription.LocalEntryNotFoundError("not cached")
        return "/models/downloaded"

    make(model_locator=locator, model_factory=returning(FakeModel())).transcribe(Path("x.webm"))
    assert calls == [True, False]


def test_download_failure_is_not_treated_as_a_cuda_failure(caplog):
    loads = []

    def locator(name, local_files_only):
        if local_files_only:
            raise transcription.LocalEntryNotFoundError("not cached")
        raise RuntimeError("network down")

    transcriber = make(
        device="auto",
        cuda_check=lambda: True,
        model_locator=locator,
        model_factory=lambda path, device, compute_type: loads.append(device) or FakeModel(),
    )
    with pytest.raises(ModelDownloadError, match="large-v3-turbo.*network down"):
        transcriber.transcribe(Path("x.webm"))
    assert loads == []
    assert "CUDA" not in caplog.text


def test_download_error_suggests_a_way_forward():
    def locator(name, local_files_only):
        if local_files_only:
            raise transcription.LocalEntryNotFoundError("not cached")
        raise RuntimeError("read timed out")

    with pytest.raises(ModelDownloadError) as error_info:
        make(model_locator=locator, model_factory=returning(FakeModel())).transcribe(Path("x.webm"))
    message = str(error_info.value)
    assert "resume" not in message  # huggingface_hub discards partial downloads, so nothing resumes
    assert "--model small" in message
    assert "--model /path/to/model-folder" in message


def test_model_is_located_once_when_falling_back_to_cpu():
    located = []
    models = {"cuda": FakeModel(fail=RuntimeError("cuBLAS failed")), "cpu": FakeModel([fake_segment])}

    def locator(name, local_files_only):
        located.append(local_files_only)
        return "/models/cached"

    transcript = make(
        device="auto",
        cuda_check=lambda: True,
        model_locator=locator,
        model_factory=lambda path, device, compute_type: models[device],
    ).transcribe(Path("x.webm"))
    assert transcript.device == "cpu"
    assert located == [True]


def test_cuda_failure_while_reading_segments_retries_on_cpu():
    models = {"cuda": FakeModel(fail=RuntimeError("cuBLAS failed")), "cpu": FakeModel([fake_segment])}
    calls = []

    def factory(path, device, compute_type):
        calls.append((device, compute_type))
        return models[device]

    transcript = make(device="auto", cuda_check=lambda: True, model_factory=factory).transcribe(Path("x.webm"))
    assert transcript.device == "cpu"
    assert calls == [("cuda", "float16"), ("cpu", "int8")]


def test_explicit_cuda_does_not_fall_back():
    transcriber = make(device="cuda", cuda_check=lambda: True, model_factory=returning(FakeModel(fail=RuntimeError("boom"))))
    with pytest.raises(RuntimeError, match="boom"):
        transcriber.transcribe(Path("x.webm"))


def test_faster_whisper_can_decode_a_recording(box_clip):
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(box_clip))
    assert len(audio) / 16000 == pytest.approx(6.0, abs=0.1)


@pytest.mark.slow
def test_real_model_on_sample():
    sample = os.environ.get("FEEDBACK_COLLECTOR_SAMPLE") or pytest.skip("set FEEDBACK_COLLECTOR_SAMPLE to a narrated recording")
    model = os.environ.get("FEEDBACK_COLLECTOR_MODEL", transcription.DEFAULT_MODEL)
    transcript = WhisperTranscriber(model=model).transcribe(Path(sample))
    print(f"device={transcript.device} segments={len(transcript.segments)}")
    assert transcript.segments
    assert [s.start for s in transcript.segments] == sorted(s.start for s in transcript.segments)
