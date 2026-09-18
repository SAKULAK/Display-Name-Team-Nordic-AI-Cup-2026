import whisperx
import gc
import json
import subprocess
import numpy as np

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

def transcibe(audio_bytes):
    device = "cuda" # or "cpu"
    batch_size = 8
    compute_type = "float16" # use "int8" if low on VRAM

    with open("HF.secret") as f:
        HF_TOKEN = f.read().strip()

    # 1. Transcribe with Whisper Large-v3-turbo
    model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type, language="en")
    audio = load_audio_from_bytes(audio_bytes)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align transcript for precise word/sentence timestamps
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Free VRAM
    del model
    del model_a
    gc.collect()

    # 3. Speaker Diarization (requires free HuggingFace token for pyannote)
    diarize_model = whisperx.diarize.DiarizationPipeline(token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
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

