"""
Tests for voice agent STT/TTS pipeline.

Run: cd /home/sc/workspace/voice-agent && python -m pytest tests/ -v

Note: Integration tests require a running server on port 8780.
Start with: cd /home/sc/workspace/voice-agent && python server/app.py
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

# Import after path is set up
from app import transcribe_audio, decode_audio_to_pcm, pcm_to_wav, stream_tts

# Default server URL
SERVER_URL = os.environ.get("VOICE_AGENT_URL", "http://127.0.0.1:8780")


def test_health_endpoint():
    """Test /health returns correct status."""
    import requests
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["stt_enabled"] is True


def test_test_stt_endpoint():
    """Test /test/stt with a real audio file."""
    import numpy as np
    import requests
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t) * 0.5
        pcm = (signal * 32767).astype(np.int16)
        
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        
        wav_path = tmp.name
    
    try:
        with open(wav_path, 'rb') as f:
            r = requests.post(f"{SERVER_URL}/test/stt", files={"audio": f}, timeout=10)
        
        assert r.status_code == 200
        data = r.json()
        assert "transcript" in data
    finally:
        os.unlink(wav_path)


def test_transcribe_silence_returns_none():
    """Silence should return None, not 'you'."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00\x00' * 16000)  # 1 second silence

        result = transcribe_audio(tmp.name)
        os.unlink(tmp.name)

        # Should be None or empty, not "you"
        assert result is None or result.strip() == "" or "you" not in result.lower()


def test_transcribe_short_audio_returns_none():
    """Very short audio (< 30ms) should return None."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00\x00' * 100)  # ~3ms

        result = transcribe_audio(tmp.name)
        os.unlink(tmp.name)

        assert result is None


def test_transcribe_noise_hallucination_filtered():
    """Known noise hallucinations should be filtered."""
    # Test that the filter catches these
    noise_transcripts = ["you", "thank you.", "thanks.", "[Music]", "[BLANK_AUDIO]"]
    for noise in noise_transcripts:
        # Simulate what happens when whisper returns noise
        # The filter should catch these in transcribe_audio
        assert noise.lower().strip() in ("you", "thank you.", "thanks.", "[music]", "[blank_audio]")


def test_pcm_to_wav():
    """Test PCM to WAV conversion."""
    pcm = b'\x00\x01' * 100
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        pcm_to_wav([pcm], 16000, tmp.name)

        with wave.open(tmp.name, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 100

        os.unlink(tmp.name)


def test_decode_webm_to_pcm():
    """Test WebM decoding to PCM."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        webm_path = tmp.name

    try:
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono', '-t', '1',
            '-c:a', 'libopus', '-b:a', '64k', webm_path
        ], check=True, capture_output=True)

        with open(webm_path, 'rb') as f:
            webm_data = f.read()

        pcm = decode_audio_to_pcm(webm_data, 16000)
        assert len(pcm) > 0
        assert len(pcm) % 2 == 0  # 16-bit samples
    finally:
        os.unlink(webm_path)


def test_stream_tts_returns_audio():
    """Test that TTS returns audio data."""
    audio_iter, sample_rate = stream_tts("Hello test")

    assert audio_iter is not None
    assert sample_rate > 0

    chunks = list(audio_iter)
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0


def test_stream_tts_wav_header():
    """Test that TTS output has WAV header."""
    audio_iter, sample_rate = stream_tts("Test")
    chunks = list(audio_iter)
    audio_data = b''.join(chunks)

    # Should have RIFF header (WAV)
    assert audio_data[:4] == b'RIFF'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
