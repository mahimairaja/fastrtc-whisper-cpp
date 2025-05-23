# Whisper CPP for FastRTC

A PyPI package that wraps Whisper.cpp for speech-to-text (STT) transcription, compatible with the FastRTC STTModel protocol. This package provides efficient, CPU-based speech recognition using the optimized Whisper.cpp implementation.

## Installation

```bash
pip install fastrtc-whisper-cpp
```

For audio file loading capabilities, install with the audio extras:

```bash
pip install "fastrtc-whisper-cpp[audio]"
```

For development:

```bash
pip install "fastrtc-whisper-cpp[dev]"
```

## Usage

### Basic Usage

```python
from fastrtc_whisper_cpp import get_stt_model
import numpy as np

# Create the model (downloads from HF if not cached)
model = get_stt_model()

# Example: Create a sample audio array (actual audio would come from a file or mic)
sample_rate = 16000
audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence

# Transcribe
text = model.stt((sample_rate, audio_data))
print(f"Transcription: {text}")
```

### Loading Audio Files

If you've installed with the audio extras:

```python
from fastrtc_whisper_cpp import get_stt_model, load_audio

# Load model
model = get_stt_model()

# Load audio file (automatically resamples to 16kHz)
audio = load_audio("path/to/audio.wav")

# Transcribe
text = model.stt(audio)
print(f"Transcription: {text}")
```

### Using with FastRTC

```python
from fastrtc_whisper_cpp import get_stt_model

# Create the model
whisper_model = get_stt_model()

# Use within FastRTC applications
# (Follow FastRTC documentation for integration details)
```

## Available Models

The package supports various Whisper.cpp models with different sizes and quantization levels:

- English-only models (faster, smaller):
  - `tiny.en`, `tiny.en-q5_1`, `tiny.en-q8_0`
  - `base.en`, `base.en-q5_1`, `base.en-q8_0`
  - `small.en`, `small.en-q5_1`, `small.en-q8_0`
  - `medium.en`, `medium.en-q5_0`, `medium.en-q8_0`

- Multilingual models:
  - `tiny`, `tiny-q5_1`, `tiny-q8_0`
  - `base`, `base-q5_1`, `base-q8_0`
  - `small`, `small-q5_1`, `small-q8_0`
  - `medium`, `medium-q5_0`, `medium-q8_0`
  - `large-v1`
  - `large-v2`, `large-v2-q5_0`, `large-v2-q8_0`
  - `large-v3`, `large-v3-q5_0`
  - `large-v3-turbo`, `large-v3-turbo-q5_0`, `large-v3-turbo-q8_0`

Example:

```python
from fastrtc_whisper_cpp import get_stt_model

# Choose a specific model
model = get_stt_model("medium.en-q8_0")
```

## Advanced Configuration

You can configure the model with specific parameters:

```python
from fastrtc_whisper_cpp import WhisperCppSTT

# Configure with specific model and models directory
model = WhisperCppSTT(
    model="medium.en",
    models_dir="/path/to/models"  # Optional custom models directory
)
```

## Requirements

- Python 3.10+
- numpy
- pywhispercpp
- librosa (optional, for audio file loading)
- click (for CLI features)

## Development

Clone the repository and install in development mode:

```bash
git clone https://github.com/mahimairaja/fastrtc-whisper-cpp.git
cd fastrtc-whisper-cpp
pip install -e ".[dev,audio]"
```

## License

MIT
