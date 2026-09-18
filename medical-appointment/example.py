"""ASR Question Answering pipeline using Ollama and Llama-3.3-70b."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import csv
import os

import ollama

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import Span, audio_duration_seconds, decode_audio
from transcribe_audio import transcibe, generate_readable_transcript

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OLLAMA_HOST = "http://127.0.0.1:13018"
OLLAMA_MODEL = "qwen2.5:14b"
# Set to a filename to save answers, or None to disable CSV export
CSV_OUTPUT_PATH = "answers.csv"

# TOGGLE FOR EXPERIMENTATION:
# False -> Uses segment timestamps with full float precision (recommended)
# True  -> Uses generate_readable_transcript() [MM:SS format]
USE_READABLE_TRANSCRIPT = False

# Initialize the Ollama client
ollama_client = ollama.Client(host=OLLAMA_HOST)

# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================
SYSTEM_PROMPT = """You are a strict clinical evidence verification assistant. Your task is to verify whether a given question/statement is explicitly supported by direct evidence in the dialogue.

EVIDENCE SPAN RULES:
1. DIALOGUE CONTEXT: If an answer depends on a question-and-answer exchange (e.g., patient asks about treatment, doctor says "no changes"), include BOTH the question segment and the answer segment.
2. PROCEDURES / ACTIONS: If verifying an action/examination (e.g., listening with a stethoscope), select all segments from the start of the action to the report of the findings.
3. COMPLETENESS: Always include complete statement/sentence units. Never select isolated fragmentary words.
4. If no direct evidence exists, has_evidence is false, and start_seg_id/end_seg_id must be null.

CRITICAL VERIFICATION RULES:
- POLARITY & NEGATION: Ensure the transcript AFFIRMS the claim. If the transcript negates the claim (e.g., statement asks "Have complications been identified?" and the transcript says "no complications"), answer has_evidence: false.
- SPAN COMPLETENESS: Always return the full segment ID(s) containing the complete speaker utterance. Never return partial sub-phrases.

OUTPUT FORMAT (strictly JSON):
{
  "reasoning": "Brief explanation",
  "has_evidence": true or false,
  "start_seg_id": <integer segment number, e.g. 5, or null>,
  "end_seg_id": <integer segment number, e.g. 7, or null>
}
"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def parse_raw_data(transcribed_raw: Union[bytes, str, dict]) -> dict:
    """Ensure transcribed_raw is parsed into a Python dictionary."""
    if isinstance(transcribed_raw, dict):
        return transcribed_raw
    if isinstance(transcribed_raw, bytes):
        return json.loads(transcribed_raw.decode("utf-8"))
    if isinstance(transcribed_raw, str):
        return json.loads(transcribed_raw)
    raise ValueError(f"Unsupported transcribed_raw format: {type(transcribed_raw)}")

def append_answer_to_csv(
    csv_path: str,
    transcript_id: str,
    question: str,
    has_evidence: bool,
    start: Optional[float],
    end: Optional[float],
) -> None:
    """Append a single question-answer result to a CSV file."""
    fieldnames = [
        "transcript_id",
        "question",
        "answer",
        "label",
        "evidence_start",
        "evidence_end",
    ]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "transcript_id": transcript_id,
            "question": question,
            "answer": "yes" if has_evidence else "no",
            "label": 1 if has_evidence else 0,
            "evidence_start": f"{start:.3f}" if (has_evidence and start is not None) else "",
            "evidence_end": f"{end:.3f}" if (has_evidence and end is not None) else "",
        })


def format_precise_transcript(raw_data: dict) -> str:
    """Format segments with full float seconds precision."""
    lines = []
    for seg in raw_data.get("segments", []):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "SPEAKER")
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.3f} - {end:.3f}] {speaker}: {text}")
    return "\n".join(lines)

def format_indexed_transcript(raw_data: dict) -> Tuple[str, Dict[int, dict]]:
    """Format segments with unique IDs for discrete LLM selection."""
    lines = []
    segment_map = {}

    for idx, seg in enumerate(raw_data.get("segments", [])):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "SPEAKER")
        text = seg.get("text", "").strip()

        segment_map[idx] = {"start": start, "end": end, "text": text}
        lines.append(f"[SEG_{idx}] {speaker}: {text}")

    return "\n".join(lines), segment_map


def parse_timestamp(val: Any) -> Optional[float]:
    """Parse float, int, MM:SS, or HH:MM:SS timestamps to seconds as a float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().lower().rstrip("s")
        if not val or val in ("null", "none"):
            return None
        if ":" in val:
            parts = val.split(":")
            try:
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return None
        try:
            return float(val)
        except ValueError:
            return None
    return None


def refine_span_with_words(quote: Optional[str], raw_data: dict) -> Optional[Tuple[float, float]]:
    """Snap a quote to the exact word timestamps in transcribed_raw if available."""
    if not quote or not quote.strip():
        return None

    clean_quote_words = re.findall(r"\w+", quote.lower())
    if not clean_quote_words:
        return None

    # Flatten all word objects from segments
    all_words: List[Dict[str, Any]] = []
    for seg in raw_data.get("segments", []):
        for w in seg.get("words", []):
            word_str = re.sub(r"[^\w]", "", w.get("word", "")).lower()
            if word_str:
                all_words.append({
                    "word": word_str,
                    "start": w.get("start"),
                    "end": w.get("end"),
                })

    if not all_words:
        return None

    # Sliding window search over word sequence
    q_len = len(clean_quote_words)
    best_match = None
    best_score = 0.0

    for i in range(len(all_words) - q_len + 1):
        window = [w["word"] for w in all_words[i : i + q_len]]
        score = sum(1 for a, b in zip(clean_quote_words, window) if a == b) / q_len
        if score > 0.8 and score > best_score:
            best_score = score
            start_val = all_words[i].get("start")
            end_val = all_words[i + q_len - 1].get("end")
            if start_val is not None and end_val is not None:
                best_match = (float(start_val), float(end_val))

    return best_match


# ==============================================================================
# MAIN ANSWER FUNCTION
# ==============================================================================
def answer_question(
    transcribed_raw: Union[bytes, str, dict],
    audio_filename: str,
    question: str,
    csv_path: Optional[str] = CSV_OUTPUT_PATH,
) -> Tuple[bool, Optional[Span]]:
    raw_data = parse_raw_data(transcribed_raw)
    transcript_id = os.path.splitext(os.path.basename(audio_filename))[0]

    # Format transcript with discrete segment IDs
    transcript_text, segment_map = format_indexed_transcript(raw_data)

    user_prompt = (
        f"Transcript:\n\"\"\"\n{transcript_text}\n\"\"\"\n\n"
        f"Question to verify: \"{question}\"\n\n"
        "Identify if explicit evidence exists and return the segment range [start_seg_id, end_seg_id]."
    )

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0.0},
            keep_alive=-1,
        )
        data = json.loads(response.message.content)
    except Exception:
        logger.exception("Failed to query Ollama for: %s", question)
        if csv_path:
            append_answer_to_csv(csv_path, transcript_id, question, False, None, None)
        return False, None

    has_evidence = data.get("has_evidence")
    if not isinstance(has_evidence, bool):
        has_evidence = str(has_evidence).strip().lower() == "true"

    if not has_evidence:
        if csv_path:
            append_answer_to_csv(csv_path, transcript_id, question, False, None, None)
        return False, None

    start_id = data.get("start_seg_id")
    end_id = data.get("end_seg_id")

    # Fallback / validation for segment IDs
    if start_id is None or end_id is None:
        if csv_path:
            append_answer_to_csv(csv_path, transcript_id, question, False, None, None)
        return False, None

    try:
        start_id = int(start_id)
        end_id = int(end_id)
    except (ValueError, TypeError):
        return False, None

    if start_id > end_id:
        start_id, end_id = end_id, start_id

    # Resolve bounds from WhisperX segments
    if start_id in segment_map and end_id in segment_map:
        start = segment_map[start_id]["start"]
        end = segment_map[end_id]["end"]
    else:
        return False, None

    if csv_path:
        append_answer_to_csv(csv_path, transcript_id, question, True, start, end)

    return True, (start, end)


# ==============================================================================
# PIPELINE ENTRY POINT
# ==============================================================================
def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation."""
    audio_bytes = decode_audio(request.audio_base64)

    duration = audio_duration_seconds(audio_bytes)
    logger.info(
        "%s (%.1f s, %.1f MB): %d questions",
        request.audio_filename,
        duration if duration is not None else float("nan"),
        len(audio_bytes) / 1e6,
        len(request.questions),
    )

    answers = []
    evidence_start = []
    evidence_end = []

    transcibed_raw = transcibe(audio_bytes)
    # transcibed_raw = {"segments": []}

    for question in request.questions:
        try:
            answer, span = answer_question(
                transcibed_raw, request.audio_filename, question
            )
        except Exception:
            logger.exception("Falling back to False for: %s", question)
            answer, span = False, None

        answers.append(answer)
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )
