import whisperx
import gc
import json
import subprocess
import numpy as np
import torch
from utils import audio_duration_seconds

with open("HF.secret") as f:
        HF_TOKEN = f.read().strip()

_WHISPER_MODEL = None
_ALIGN_MODEL = None
_ALIGN_METADATA = None
_DIARIZE_MODEL = None

def load_audio_from_bytes(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Decodes audio bytes (MP3, WAV, M4A, etc.) in-memory 
    to a 16kHz mono float32 NumPy array.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:0",           # Read directly from stdin pipe
        "-f", "s16le",            # Convert to signed 16-bit PCM
        "-ac", "1",                # Mono
        "-acodec", "pcm_s16le",
        "-ar", str(sr),           # 16,000 Hz
        "-"                       # Output to stdout pipe
    ]

    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    out, err = process.communicate(input=audio_bytes)
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to decode audio: {err.decode('utf-8', errors='ignore')}")

    # Convert PCM 16-bit to float32 normalized between -1.0 and 1.0
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def get_models():
    """Initializes and caches models once in memory."""
    global _WHISPER_MODEL, _ALIGN_MODEL, _ALIGN_METADATA, _DIARIZE_MODEL

    if _WHISPER_MODEL is None:
        # 1. Whisper Large-v3-turbo in INT8 uses only ~1.0 GB VRAM
        _WHISPER_MODEL = whisperx.load_model(
            "large-v3-turbo", 
            device="cuda", 
            compute_type="int8", 
            language="en"
        )

    if _ALIGN_MODEL is None:
        # 2. Alignment on CPU uses 0 MB GPU VRAM and aligns in < 1 second
        _ALIGN_MODEL, _ALIGN_METADATA = whisperx.load_align_model(
            language_code="en", 
            device="cpu"
        )

    if _DIARIZE_MODEL is None and HF_TOKEN:
        # 3. Diarization on GPU uses ~1.4 GB VRAM
        _DIARIZE_MODEL = whisperx.diarize.DiarizationPipeline(
            token=HF_TOKEN, 
            device="cuda"
        )

    return _WHISPER_MODEL, _ALIGN_MODEL, _ALIGN_METADATA, _DIARIZE_MODEL


def transcibe(audio_bytes):
    whisper_model, align_model, align_metadata, diarize_model = get_models()

    audio = load_audio_from_bytes(audio_bytes)
    
    # Calculate audio duration in seconds
    duration = audio_duration_seconds(audio_bytes)

    # 1. Transcribe (batch_size=4 is memory-safe for 4-minute audio)
    result = whisper_model.transcribe(audio, batch_size=4)

    # 2. Align on CPU (Vital: GPU alignment on 232s causes CUDA OOM)
    result = whisperx.align(
        result["segments"], 
        align_model, 
        align_metadata, 
        audio, 
        device="cpu", 
        return_char_alignments=False
    )

    # 3. Diarization with Duration Guard:
    # Only run diarization on clips <= 150s. On longer clips, skipping diarization
    # saves 30 seconds and prevents budget timeouts / OOM.
    if diarize_model is not None and duration <= 150.0:
        try:
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print("WARN: Diarization skipped")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# Output contains segments with exact 'start', 'end', and 'speaker'

# with open("test.json", "w") as f:
#     json.dump(result, f)
# print(result["segments"])

def format_seconds(seconds: float) -> str:
    """Converts seconds (e.g. 75.4) to MM:SS format (e.g. 01:15)."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def generate_readable_transcript(result, output_file="transcript.txt"):
    lines = []
    
    current_speaker = None
    current_text = []
    start_time = None
    end_time = None

    for seg in result["segments"]:
        speaker = seg.get("speaker", "Unknown Speaker")
        text = seg["text"].strip()
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        # If the same speaker continues talking, group their sentences together
        if speaker == current_speaker:
            current_text.append(text)
            end_time = seg_end
        else:
            # Commit the previous speaker's dialogue
            if current_speaker is not None:
                lines.append(f"[{format_seconds(start_time)} - {format_seconds(end_time)}] {current_speaker}:\n{' '.join(current_text)}\n")
            
            # Start new speaker turn
            current_speaker = speaker
            current_text = [text]
            start_time = seg_start
            end_time = seg_end

    # Flush the last turn
    if current_speaker is not None:
        lines.append(f"[{format_seconds(start_time)} - {format_seconds(end_time)}] {current_speaker}:\n{' '.join(current_text)}\n")

    transcript_text = "\n".join(lines)

    # Save to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    return transcript_text

