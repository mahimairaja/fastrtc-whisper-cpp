from functools import lru_cache
from pathlib import Path
from typing import Protocol

import click
import numpy as np
from numpy.typing import NDArray

from ..constants import AVAILABLE_MODELS

curr_dir = Path(__file__).parent


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        ...


class WhisperCppSTT(STTModel):
    def __init__(self, model: str = "base.en", models_dir=None):
        try:
            from pywhispercpp.model import Model
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "fastrtc-whisper-cpp is required for speech-to-text using whisper.cpp."
                "Install it with `pip install fastrtc-whisper-cpp`."
            )
        if model not in AVAILABLE_MODELS:
            formatted_models = "\n".join(f"  - {m}" for m in AVAILABLE_MODELS)
            raise ValueError(
                f"Model '{model}' not found. Available models:\n{formatted_models}"
            )

        self.model = Model(
            model=model,
            models_dir=models_dir,
        )

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sr, audio_np = audio  # type: ignore

        if audio_np.dtype == np.int16:
            # Convert int16 to float32 and normalize to [-1, 1]
            audio_np = audio_np.astype(np.float32) / 32768.0

        if sr != 16000:
            try:
                import librosa
            except ImportError:
                raise ImportError(
                    "Pass a 16000Hz audio or "
                    "install librosa with `pip install librosa`."
                )
            audio_np: NDArray[np.float32] = librosa.resample(
                audio_np, orig_sr=sr, target_sr=16000
            )

        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)

        segments = self.model.transcribe(audio_np)
        return " ".join(segment.text for segment in segments)

    def list_models(self):
        click.echo(f"{click.style('INFO', fg='blue')}:\t  Available models:")
        [
            click.echo(f"{click.style('MODEL', fg='green')}\t  {model}")
            for model in AVAILABLE_MODELS
        ]


@lru_cache
def get_stt_model(
    model: str = "base.en",
    models_dir: str = None,
) -> STTModel:
    if model not in AVAILABLE_MODELS:
        formatted_models = "\n".join(f"  - {m}" for m in AVAILABLE_MODELS)
        raise ValueError(
            f"Model '{model}' not found. Available models:\n{formatted_models}"
        )

    m = WhisperCppSTT(model, models_dir)

    # sample audio array
    sample_rate = 16000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)

    print(click.style("INFO", fg="green") + ":\t  Warming up STT model.")

    m.stt((16000, audio))
    print(click.style("INFO", fg="green") + ":\t  STT model warmed up.")
    return m
