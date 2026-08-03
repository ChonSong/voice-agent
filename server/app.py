"""
Voice Agent Server — Hermes native TTS/STT pipeline.

Integrates with the Hermes agent's existing providers:
- TTS: tools.tts_streaming (streaming PCM)
- STT: tools.transcription_tools.transcribe_audio (file-based)

Browser → WebSocket → save PCM → transcribe → TTS stream → browser
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path

import aiohttp
from aiohttp import web

# ── Hermes agent path ───────────────────────────────────────────────────────

HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES_AGENT))

# ── Configuration ───────────────────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 8780
PUBLIC_DIR = Path(__file__).parent.parent / "public"
SAMPLE_RATE = 16000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Hermes imports ──────────────────────────────────────────────────────────

def get_tts_config():
    from tools.tts_tool import _load_tts_config
    return _load_tts_config()


def stream_tts(text: str):
    """Generate TTS audio using the agent's configured provider.
    
    Returns (audio_bytes, sample_rate) or (None, None) on failure.
    Uses streaming provider if available, otherwise falls back to
    text_to_speech_tool (works with all providers including Edge).
    """
    try:
        from tools.tts_streaming import resolve_streaming_provider
        tts_config = get_tts_config()
        streamer = resolve_streaming_provider(tts_config)
        if streamer is not None:
            logger.info("TTS streaming at %d Hz", streamer.sample_rate)
            return streamer.stream(text), streamer.sample_rate
    except Exception as exc:
        logger.debug("Streaming TTS not available: %s", exc)
    
    # Fallback: use text_to_speech_tool (works with all providers)
    try:
        import json
        import tempfile
        import subprocess
        from tools.tts_tool import text_to_speech_tool
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            out_path = tmp.name
        
        try:
            result_str = text_to_speech_tool(text, output_path=out_path)
            result = json.loads(result_str)
            if result.get("success"):
                # Convert MP3 to WAV for browser compatibility
                wav_path = out_path.replace(".mp3", ".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", out_path, wav_path],
                    capture_output=True, timeout=10, check=True
                )
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                os.unlink(wav_path)
                logger.info("TTS via text_to_speech_tool: %d bytes", len(audio_data))
                return iter([audio_data]), 24000
            logger.warning("TTS failed: %s", result.get("error"))
            return None, None
        finally:
            os.unlink(out_path)
    except Exception as exc:
        logger.error("TTS error: %s", exc)
        return None, None


def transcribe_audio(wav_path: str):
    """Transcribe a WAV file using Distil-Whisper via Faster-Whisper (CTranslate2)."""
    try:
        # Import the local Distil-Whisper module (lazy import so Hermes deps work)
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from stt import transcribe_audio as distil_transcribe
        transcript = distil_transcribe(wav_path)
        return transcript
    except ImportError:
        # Fallback: Hermes faster-whisper tiny if Distil-Whisper model not downloaded
        from tools.transcription_tools import transcribe_audio as hermes_transcribe
        result = hermes_transcribe(wav_path)
        if isinstance(result, dict):
            if result.get("success"):
                return result.get("transcript", "").strip()
            else:
                logger.error("Hermes STT fallback error: %s", result.get("error"))
                return None
        return str(result).strip() if result else None
    except Exception as e:
        logger.error("Transcription error: %s", e)
        return None


# ── Audio helpers ───────────────────────────────────────────────────────────

def pcm_to_wav(pcm_chunks: list[bytes], sample_rate: int, path: str):
    """Write raw PCM chunks (16-bit signed, mono) to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in pcm_chunks:
            wf.writeframes(chunk)


def decode_audio_to_pcm(blob_bytes: bytes, target_sample_rate: int) -> bytes:
    """Decode WebM/Opus blob to raw PCM using ffmpeg."""
    try:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as inp:
            inp.write(blob_bytes)
            inp_path = inp.name

        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", inp_path,
                "-f", "s16le",
                "-ar", str(target_sample_rate),
                "-ac", "1",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        os.unlink(inp_path)

        if proc.returncode != 0:
            logger.warning("ffmpeg decode failed: %s", proc.stderr.decode()[:200])
            return b""
        return proc.stdout
    except Exception as exc:
        logger.error("Audio decode failed: %s", exc)
        return b""


# ── Routes ──────────────────────────────────────────────────────────────────

async def index_handler(request):
    resp = web.FileResponse(str(PUBLIC_DIR / "index.html"))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


async def health_handler(request):
    try:
        from tools.transcription_tools import is_stt_enabled, _load_stt_config, _get_provider
        stt_config = _load_stt_config()
        stt_enabled = is_stt_enabled(stt_config)
        stt_provider = _get_provider(stt_config) if stt_enabled else None
    except Exception:
        stt_enabled = False
        stt_provider = None

    tts_config = get_tts_config()
    return web.json_response({
        "status": "ok",
        "stt_enabled": stt_enabled,
        "stt_provider": stt_provider,
        "tts_provider": tts_config.get("provider") if tts_config else None,
    })


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    audio_chunks: list[bytes] = []
    is_webm = False

    logger.info("WebSocket client connected")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                # Detect WebM by EBML header (only on first chunk)
                if not audio_chunks and data[:4] == b'\x1a\x45\xdf\xa3':
                    is_webm = True
                audio_chunks.append(data)

            elif msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "audio_start":
                    audio_chunks.clear()
                    is_webm = False
                    await ws.send_json({"type": "status", "text": "Recording..."})

                elif msg_type == "audio_end":
                    await ws.send_json({"type": "status", "text": "Transcribing..."})

                    if not audio_chunks:
                        await ws.send_json({"type": "transcript", "text": ""})
                        await ws.send_json({"type": "done"})
                        continue

                    # Save audio to file and transcribe
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        wav_path = tmp.name

                    try:
                        if is_webm:
                            # Combine all WebM chunks and decode to PCM
                            combined = b''.join(audio_chunks)
                            pcm = await asyncio.to_thread(decode_audio_to_pcm, combined, SAMPLE_RATE)
                            if pcm:
                                pcm_to_wav([pcm], SAMPLE_RATE, wav_path)
                        else:
                            pcm_to_wav(audio_chunks, SAMPLE_RATE, wav_path)
                        
                        # Send periodic keepalive to prevent browser timeout
                        async def keepalive():
                            while not ws.closed:
                                await asyncio.sleep(3)
                                if not ws.closed:
                                    try:
                                        await ws.send_json({"type": "status", "text": "Still transcribing..."})
                                    except:
                                        break
                        
                        # Run transcription with keepalive
                        keepalive_task = asyncio.create_task(keepalive())
                        try:
                            transcript = await asyncio.to_thread(transcribe_audio, wav_path)
                        finally:
                            keepalive_task.cancel()
                        
                        if transcript:
                            await ws.send_json({"type": "transcript", "text": transcript})
                            await respond_with_tts(ws, transcript)
                        else:
                            await ws.send_json({"type": "transcript", "text": "(could not transcribe)"})
                            await ws.send_json({"type": "done"})
                    finally:
                        os.unlink(wav_path)
                        audio_chunks.clear()

                elif msg_type == "text":
                    text = data.get("text", "")
                    await ws.send_json({"type": "transcript", "text": text})
                    await respond_with_tts(ws, text)

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

    except Exception as exc:
        logger.error("WebSocket error: %s", exc)

    logger.info("WebSocket client disconnected")
    return ws


async def respond_with_tts(ws, text: str):
    """Generate TTS audio and stream it back via WebSocket."""
    await ws.send_json({"type": "status", "text": "Speaking..."})

    try:
        audio_iter, sample_rate = await asyncio.to_thread(stream_tts, text)

        if audio_iter is None:
            await ws.send_json({"type": "error", "text": "TTS provider not available"})
            await ws.send_json({"type": "done"})
            return

        # Send sample rate as first text message before binary chunks
        await ws.send_json({"type": "audio_start", "sampleRate": sample_rate})

        # Stream audio chunks
        for chunk in audio_iter:
            if not ws.closed:
                await ws.send_bytes(chunk)

        await ws.send_json({"type": "done"})
    except Exception as exc:
        logger.error("TTS streaming error: %s", exc)
        await ws.send_json({"type": "error", "text": str(exc)})
        await ws.send_json({"type": "done"})


# ── Test endpoint: upload audio file for STT ───────────────────────────────
async def test_stt_handler(request):
    """Test STT by uploading an audio file directly."""
    reader = await request.multipart()
    field = await reader.next()
    if field and field.name == "audio":
        data = await field.read()
        suffix = field.filename.rsplit(".", 1)[-1] if field.filename else "webm"
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
            tmp.write(data)
            audio_path = tmp.name
        
        try:
            if suffix in ("webm", "opus"):
                pcm = decode_audio_to_pcm(data, SAMPLE_RATE)
                wav_path = audio_path.replace(f".{suffix}", ".wav")
                if pcm:
                    pcm_to_wav([pcm], SAMPLE_RATE, wav_path)
                else:
                    return web.json_response({"error": "Failed to decode audio"}, status=400)
            else:
                wav_path = audio_path
            
            transcript = transcribe_audio(wav_path)
            return web.json_response({"transcript": transcript})
        finally:
            os.unlink(audio_path)
            if wav_path != audio_path:
                os.unlink(wav_path)
    
    return web.json_response({"error": "No audio file provided"}, status=400)


# ── App ─────────────────────────────────────────────────────────────────────

app = web.Application()
app.router.add_get("/", index_handler)
app.router.add_get("/health", health_handler)
app.router.add_get("/ws", websocket_handler)
app.router.add_post("/test/stt", test_stt_handler)
app.router.add_static("/vendor", PUBLIC_DIR / "vendor")
app.router.add_static("/", PUBLIC_DIR)


if __name__ == "__main__":
    print(f"Voice Agent Server → http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
