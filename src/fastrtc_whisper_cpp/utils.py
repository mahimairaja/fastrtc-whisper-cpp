from typing import Tuple

import numpy as np


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[int, np.ndarray]:
    """
    Load an audio file and return it in the format expected by the STT model.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (sample_rate, audio_data)
    """
    try:
        import librosa

        audio, sr = librosa.load(file_path, sr=target_sr)
        return (sr, audio)
    except ImportError:
        raise ImportError(
            "librosa is required for loading audio files. "
            "Install it with `pip install librosa`."
        )
