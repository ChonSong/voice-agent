"""
Transcription backend using Distil-Whisper via Faster-Whisper (CTranslate2).

Distil-Whisper is a distilled version of Whisper that runs 5-6x faster
than whisper-large-v3 with minimal accuracy loss. Faster-Whisper wraps
CTranslate2 for optimized inference — combining both gives a compounding
speed boost especially on GPU or longer audio.

Model: Systran/faster-distil-whisper-small.en (~165M params, cached locally)
Fallback: faster-whisper tiny (if distil model not available)
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# CTranslate2-converted Distil-Whisper model (via Systran)
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/"
    "snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4"
)


def load_model(device: str = "cpu", compute_type: str = "int8"):
    """Load the Distil-Whisper model via Faster-Whisper."""
    from faster_whisper import WhisperModel

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Distil-Whisper model not found at {MODEL_PATH}. "
            "Download it first: huggingface-cli download distil-whisper/distil-small.en"
        )

    logger.info("Loading Distil-Whisper model via Faster-Whisper (CTranslate2)...")
    model = WhisperModel(
        MODEL_PATH,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    logger.info("Distil-Whisper model loaded successfully")
    return model


# Global model instance (lazy-loaded)
_model: Optional[object] = None


def transcribe_audio(wav_path: str) -> Optional[str]:
    """Transcribe a WAV file using Distil-Whisper via Faster-Whisper (CTranslate2)."""
    global _model

    try:
        if _model is None:
            _model = load_model()

        segments, info = _model.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=False,  # Disabled — WebM decode audio is too quiet for VAD
        )

        transcript = " ".join(segment.text for segment in segments).strip()
        logger.info(
            "Transcribed via Distil-Whisper+Faster-Whisper: %s (%.2fs, %.1f%% conf)",
            transcript[:80],
            info.duration,
            info.language_probability * 100,
        )
        return transcript

    except Exception as e:
        logger.error("Transcription error: %s", e)
        return None


def transcribe_audio_raw(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
    """Transcribe raw audio bytes (16-bit PCM, mono)."""
    import tempfile
    import wave

    global _model

    try:
        if _model is None:
            _model = load_model()

        # Write to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

        try:
            segments, info = _model.transcribe(
                wav_path,
                language="en",
                beam_size=5,
                vad_filter=False,  # Disabled — WebM decode audio is too quiet for VAD
            )
            transcript = " ".join(segment.text for segment in segments).strip()
            logger.info(
                "Transcribed raw audio: %s (%.2fs)",
                transcript[:80],
                info.duration,
            )
            return transcript
        finally:
            os.unlink(wav_path)

    except Exception as e:
        logger.error("Raw transcription error: %s", e)
        return None
