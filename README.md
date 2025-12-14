# Whisper Audio Transcription API

A REST API service for audio transcription using OpenAI's Whisper model. This service provides automatic speech recognition with language detection capabilities.

## Features

- Audio transcription using OpenAI's Whisper medium model
- Automatic language detection
- Support for multiple audio formats (MP3, WAV, OGG, FLAC)
- Voice Activity Detection (VAD) for silence removal
- CUDA GPU acceleration support
- Containerized deployment with Docker

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- CUDA-compatible GPU (optional, for faster inference)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/Lukas-LLS/Whisper.git
cd Whisper/whisper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```

3. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:5001`

### Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/Lukas-LLS/Whisper.git
cd Whisper/whisper
```

2. Build the Docker image:
```bash
docker build -t whisper-whisper .
```

3. Build and run with Docker Compose:

```bash
docker compose up -d
```

The API will be available at `http://localhost:5001`

## Usage

### API Endpoint

**POST** `/api/v1/transcribe`

Transcribe audio files to text with automatic language detection.

#### Request

- **Content-Type**: `audio/wav`, `audio/mpeg`, `audio/ogg`, or `audio/flac`
- **Body**: Raw audio file bytes

#### Response

```json
{
  "transcription": "The transcribed text from the audio file"
}
```

### Example with cURL

```bash
curl -X POST http://localhost:5001/api/v1/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @your-audio-file.wav
```

### Example with Python

```python
import requests

with open('audio.wav', 'rb') as audio_file:
    response = requests.post(
        'http://localhost:5001/api/v1/transcribe',
        headers={'Content-Type': 'audio/wav'},
        data=audio_file
    )
    
print(response.json()['transcription'])
```

## Configuration

### Whisper Model

The default model is `openai/whisper-large-v3`. To use a different model, edit `whisper/whisper_model.py`:

```python
WHISPER_MODEL_ID = "openai/whisper-large-v3"
```

Available models:
- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large`
- `openai/whisper-large-v2`
- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`

### Server Configuration

The server runs on port 5001 by default. To change this, modify the last line in `whisper/main.py`:

```python
app.run(host="0.0.0.0", port=5001)  # Change port as needed
```

## Technical Details

### Audio Processing

1. **Resampling**: Audio is resampled to 16kHz (Whisper's expected input rate)
2. **Dynamic Range Adjustment**: Audio is normalized to 16-bit dynamic range for optimal model performance
3. **Voice Activity Detection**: Silence is removed using VAD with configurable trigger levels
4. **Language Detection**: First 3 seconds of audio are used to detect the spoken language

### Performance

- PyTorch 2.0 compilation is automatically enabled when available for improved performance
- CUDA GPU acceleration is used automatically when available
- Inference time varies based on audio length and hardware
