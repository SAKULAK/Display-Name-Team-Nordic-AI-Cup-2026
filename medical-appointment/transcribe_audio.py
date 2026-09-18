"""Fast WhisperX transcription and word alignment with pre-warming."""

import gc
import logging
import subprocess
import numpy as np
import torch
import whisperx

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None
_ALIGN_MODEL = None
_ALIGN_METADATA = None


def load_audio_from_bytes(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """Decodes audio bytes in-memory to 16kHz mono float32 numpy array."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate(input=audio_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to decode audio: {err.decode('utf-8', errors='ignore')}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_models():
    """Initializes and caches Whisper and Alignment models once."""
    global _WHISPER_MODEL, _ALIGN_MODEL, _ALIGN_METADATA

    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisperx.load_model(
            "large-v3-turbo",
            device="cuda",
            compute_type="int8",
            language="en",
        )

    if _ALIGN_MODEL is None:
        _ALIGN_MODEL, _ALIGN_METADATA = whisperx.load_align_model(
            language_code="en",
            device="cpu",
        )

    return _WHISPER_MODEL, _ALIGN_MODEL, _ALIGN_METADATA


def warmup_transcription():
    """Pre-loads models and executes a dummy forward pass to eliminate cold-start lag."""
    logger.info("Warming up WhisperX and Alignment models...")
    whisper_model, align_model, align_metadata = get_models()
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    try:
        res = whisper_model.transcribe(dummy_audio, batch_size=1)
        whisperx.align(res["segments"], align_model, align_metadata, dummy_audio, device="cpu")
        logger.info("WhisperX warmup complete.")
    except Exception as e:
        logger.warning("WhisperX warmup encountered an issue: %s", e)


def transcribe(audio_bytes: bytes) -> dict:
    """Fast transcription and word-level alignment (Diarization removed for speed/safety)."""
    whisper_model, align_model, align_metadata = get_models()
    audio = load_audio_from_bytes(audio_bytes)

    # 1. Transcribe (fast with int8 large-v3-turbo)
    result = whisper_model.transcribe(audio, batch_size=8)

    # 2. Align on CPU to get exact word-level start/end timestamps
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device="cpu",
        return_char_alignments=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result