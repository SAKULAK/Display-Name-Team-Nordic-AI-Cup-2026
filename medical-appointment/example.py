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
SYSTEM_PROMPT = """You are a strict evidence verification assistant. Your task is to verify whether a given question/claim is explicitly supported by direct evidence in the provided audio transcript.

CRITICAL INSTRUCTIONS:
1. FOCUS ON EVIDENCE FIRST: Do not guess, speculate, or answer using outside knowledge. Your sole objective is to search the transcript for explicit, factual evidence supporting the statement.
2. IF DIRECT EVIDENCE IS FOUND:
   - "has_evidence" must be true.
   - "quote": Extract the exact verbatim quote from the transcript that proves the claim.
   - "start": The starting timestamp of the supporting evidence segment.
   - "end": The ending timestamp of the supporting evidence segment.
3. IF NO DIRECT EVIDENCE IS FOUND:
   - "has_evidence" must be false.
   - "quote": null
   - "start": null
   - "end": null
   - This applies if:
     a) The transcript contradicts or disputes the statement.
     b) The statement is not mentioned in the transcript.
     c) The statement is plausible in reality, but cannot be proven directly from the transcript.
4. Output strictly valid JSON with this exact schema:
{
  "reasoning": "Briefly describe the evidence search and whether explicit proof exists",
  "has_evidence": true or false,
  "quote": "verbatim quote or null",
  "start": <number or timestamp or null>,
  "end": <number or timestamp or null>
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
    use_readable_format: Optional[bool] = None,
    csv_path: Optional[str] = CSV_OUTPUT_PATH,
) -> Tuple[bool, Optional[Span]]:
    """Determine whether the question is supported by evidence in the transcript."""
    if use_readable_format is None:
        use_readable_format = USE_READABLE_TRANSCRIPT

    raw_data = parse_raw_data(transcribed_raw)
    transcript_id = os.path.splitext(os.path.basename(audio_filename))[0]

    # Format transcript according to test option
    if use_readable_format:
        transcript_text = generate_readable_transcript(transcribed_raw)
    else:
        transcript_text = format_precise_transcript(raw_data)

    user_prompt = (
        f"Transcript:\n\"\"\"\n{transcript_text}\n\"\"\"\n\n"
        f"Statement/Question to verify:\n\"{question}\"\n\n"
        "Search the transcript for explicit evidence supporting this statement. "
        "Return the response in the specified JSON format."
    )

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={
                "temperature": 0.0,
            },
            keep_alive=-1,  # Keeps model pinned in GPU VRAM
        )
        content = response.message.content
        data = json.loads(content)
    except Exception:
        logger.exception("Failed to query Ollama or parse JSON output for: %s", question)
        if csv_path:
            append_answer_to_csv(csv_path, transcript_id, question, False, None, None)
        return False, None

    # Extract decision
    has_evidence = data.get("has_evidence")
    if not isinstance(has_evidence, bool):
        has_evidence = str(has_evidence).strip().lower() == "true"

    if not has_evidence:
        if csv_path:
            append_answer_to_csv(csv_path, transcript_id, question, False, None, None)
        return False, None

    # Parse timestamps
    start = parse_timestamp(data.get("start"))
    end = parse_timestamp(data.get("end"))

    # Attempt to snap quote to exact word timestamps
    quote = data.get("quote")
    refined_span = refine_span_with_words(quote, raw_data)
    if refined_span is not None:
        start, end = refined_span

    # Validate timestamps: if true, valid start and end are strictly required
    if start is None or end is None:
        logger.warning("Evidence claimed as True, but timestamps missing. Setting False.")
        if csv_path:
            append_answer_to_csv(csv_path, transcript_id, question, False, None, None)
        return False, None

    if start > end:
        start, end = end, start

    # Save positive match to CSV
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
