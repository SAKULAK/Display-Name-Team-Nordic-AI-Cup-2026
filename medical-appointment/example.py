"""ASR Question Answering pipeline using Ollama and word-level temporal snapping."""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama
import torch

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from transcribe_audio import transcribe, warmup_transcription
from utils import Span, audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OLLAMA_HOST = "http://127.0.0.1:13018"
OLLAMA_MODEL = "phi4"
CSV_OUTPUT_PATH = "answers.csv"

ollama_client = ollama.Client(host=OLLAMA_HOST)

# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================
SYSTEM_PROMPT = """You are a strict clinical evidence verifier. Your task is to verify if a claim is factually TRUE and explicitly CONFIRMED by the transcript.

CRITICAL RULES FOR VERIFICATION:
1. DOCTOR-PATIENT NEGATION (MUST be CONTRADICTED / has_evidence: false):
   - If a doctor asks about a symptom or event ("Any fever?", "Does the pain radiate down your leg?", "Any known COVID exposure?"), and the patient answers "No", "None", or "Nothing like that" -> The claim that the patient had this symptom is CONTRADICTED.
   - If an exam finding is absent ("no pus", "no sores", "no redness", "no swelling", "no foreign body") -> Claims asserting the finding was present or seen are CONTRADICTED.
   - If a test was negative ("strep test was negative", "COVID test was negative") -> Claims asserting infection/bacteria found are CONTRADICTED.

2. ANTONYMS & OPPOSITE FINDINGS (MUST be CONTRADICTED / has_evidence: false):
   - "Normal", "stable", "good", or "unchanged" CONTRADICTS claims of "abnormal", "impaired", "elevated", "unstable", or "worsened".

3. UNMENTIONED SPECIFIC ENTITIES (MUST be NOT_MENTIONED / has_evidence: false):
   - If a specific symptom, diagnosis, or entity was never named, it is NOT_MENTIONED.
   - Strict matching: "holiday" is not "weekend"; watching a television series is not a "specific hobby".

4. EVIDENCE LOCALIZATION (Only if claim_status is CONFIRMED):
   - Identify the single sentence [S_id] containing the direct proof.
   - Extract the exact short quote (3 to 6 words) from that sentence directly stating the fact.

OUTPUT STRICT VALID JSON:
{
  "rationale": "<brief 3-7 words explaining if affirmed, denied with no, normal vs abnormal, or unmentioned>",
  "claim_status": "CONFIRMED" | "CONTRADICTED" | "NOT_MENTIONED",
  "has_evidence": true or false,
  "seg_id": <int or null>,
  "quote": "<exact short quote from transcript or null>"
}
"""

# ==============================================================================
# WARMUP FUNCTION
# ==============================================================================
def warmup_pipeline():
    """Warms up WhisperX and Ollama at startup so Conversation 1 never times out."""
    warmup_transcription()
    logger.info("Warming up Ollama (%s)...", OLLAMA_MODEL)
    try:
        ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_predict": 1},
            keep_alive=-1,
        )
        logger.info("Ollama warmup complete.")
    except Exception as e:
        logger.warning("Ollama warmup encountered an issue: %s", e)


# Run warmup immediately on import
warmup_pipeline()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def parse_raw_data(transcribed_raw: Union[bytes, str, dict]) -> dict:
    if isinstance(transcribed_raw, dict):
        return transcribed_raw
    if isinstance(transcribed_raw, bytes):
        return json.loads(transcribed_raw.decode("utf-8"))
    if isinstance(transcribed_raw, str):
        return json.loads(transcribed_raw)
    raise ValueError(f"Unsupported format: {type(transcribed_raw)}")


def format_indexed_transcript(raw_data: dict) -> Tuple[str, Dict[int, dict]]:
  """Splits Whisper segments into distinct sentences using word timestamps.

  Returns:
      transcript_text: Formatted string of [S_0], [S_1], ... for the prompt.
      sentence_map: Dict mapping int -> {"start": float, "end": float, "text":
      str, "words": list}
  """
  lines = []
  sentence_map = {}
  sent_id = 0

  for seg in raw_data.get("segments", []):
    words = seg.get("words", [])

    # Fallback: if WhisperX alignment didn't produce words for this segment,
    # keep the raw segment as a single unit so nothing is lost.
    if not words:
      text = seg.get("text", "").strip()
      if text:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        sentence_map[sent_id] = {
            "start": start,
            "end": end,
            "text": text,
            "words": [],
        }
        lines.append(f"[S_{sent_id}] {text}")
        sent_id += 1
      continue

    curr_words = []
    for w in words:
      if "start" not in w or "end" not in w:
        continue
      curr_words.append(w)

      # Break into a sentence on terminal punctuation (. ? !) or after 18 words
      clean_token = w.get("word", "").strip()
      is_terminal = (
          bool(clean_token)
          and clean_token[-1] in ".?!"
          or len(curr_words) >= 18
      )

      if is_terminal and curr_words:
        sent_text = " ".join([cw.get("word", "").strip() for cw in curr_words])
        sentence_map[sent_id] = {
            "start": float(curr_words[0]["start"]),
            "end": float(curr_words[-1]["end"]),
            "text": sent_text,
            "words": curr_words,
        }
        lines.append(f"[S_{sent_id}] {sent_text}")
        sent_id += 1
        curr_words = []

    # Catch any leftover words in the segment
    if curr_words:
      sent_text = " ".join([cw.get("word", "").strip() for cw in curr_words])
      sentence_map[sent_id] = {
          "start": float(curr_words[0]["start"]),
          "end": float(curr_words[-1]["end"]),
          "text": sent_text,
          "words": curr_words,
      }
      lines.append(f"[S_{sent_id}] {sent_text}")
      sent_id += 1

  return "\n".join(lines), sentence_map

def refine_span_with_words(
    quote: Optional[str],
    target_seg: dict,
    raw_data: dict,
) -> Optional[Tuple[float, float]]:
  """Snaps an LLM quote to exact word-level timestamps from WhisperX alignment.

  Returns (start, end) in seconds, or None if no reliable match is found.
  """
  if not quote or not quote.strip():
    return None

  # Clean quote tokens: lowercase, strip all non-alphanumeric characters
  clean_quote = [re.sub(r"[^\w]", "", w).lower() for w in quote.split()]
  clean_quote = [w for w in clean_quote if w]
  if not clean_quote:
    return None

  def search_word_list(words_list: List[dict]) -> Optional[Tuple[float, float]]:
    """Searches a list of WhisperX word dicts for the best quote match."""
    # Extract only words that have valid start and end timestamps
    cleaned_words = []
    for w in words_list:
      cw = re.sub(r"[^\w]", "", w.get("word", "")).lower()
      if cw and "start" in w and "end" in w:
        cleaned_words.append((cw, float(w["start"]), float(w["end"])))

    if not cleaned_words:
      return None

    q_len = len(clean_quote)

    # Strategy A: Sliding window over words
    if len(cleaned_words) >= q_len:
      best_score = 0.0
      best_span = None

      for i in range(len(cleaned_words) - q_len + 1):
        window = [cleaned_words[i + k][0] for k in range(q_len)]
        score = sum(1 for a, b in zip(clean_quote, window) if a == b) / q_len
        if score > best_score and score >= 0.60:
          best_score = score
          best_span = (cleaned_words[i][1], cleaned_words[i + q_len - 1][2])
          if score == 1.0:
            return best_span

      if best_span is not None and best_score >= 0.70:
        return best_span

    # Strategy B: Anchor on first and last word of the quote
    first_tok = clean_quote[0]
    last_tok = clean_quote[-1]

    cand_starts = [s for (tok, s, e) in cleaned_words if tok == first_tok]
    cand_ends = [e for (tok, s, e) in cleaned_words if tok == last_tok]

    # Find the tightest valid span (must be > 0s and < 30s)
    valid_spans = [
        (s, e)
        for s in cand_starts
        for e in cand_ends
        if 0.3 <= (e - s) <= 30.0 and s < e
    ]
    if valid_spans:
      # Return the shortest valid span that fits
      valid_spans.sort(key=lambda span: span[1] - span[0])
      return valid_spans[0]

    return None

  # 1. Search inside the target segment first (avoids duplicate matches elsewhere)
  target_words = target_seg.get("words", [])
  if target_words:
    span = search_word_list(target_words)
    if span and span[0] < span[1]:
      return span

  # 2. Fallback: Search all words in the entire transcript
  all_words = []
  for seg in raw_data.get("segments", []):
    all_words.extend(seg.get("words", []))

  if all_words:
    span = search_word_list(all_words)
    if span and span[0] < span[1]:
      return span

  return None


# ==============================================================================
# MAIN ANSWER FUNCTION
# ==============================================================================
def answer_question(
    raw_data: dict,
    question: str,
) -> Tuple[bool, Optional[Span]]:
    transcript_text, segment_map = format_indexed_transcript(raw_data)

    user_prompt = (
        f"Transcript:\n\"\"\"\n{transcript_text}\n\"\"\"\n\n"
        f"Claim to verify: \"{question}\"\n\n"
        "State if explicitly confirmed, contradicted, or not mentioned. Return JSON."
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
                "num_predict": 80,
                "num_ctx": 4096,
            },
            keep_alive=-1,
        )
        data = json.loads(response.message.content)
    except Exception:
        logger.exception("Ollama query failed for question: %s", question)
        return False, None

    claim_status = str(data.get("claim_status", "")).strip().upper()
    raw_has_evidence = data.get("has_evidence")
    if not isinstance(raw_has_evidence, bool):
        raw_has_evidence = str(raw_has_evidence).strip().lower() == "true"

    # Strict confirmation check
    if not (raw_has_evidence and claim_status == "CONFIRMED"):
        return False, None

    # Handle segment ID
    sent_id = data.get("sentence_id")
    if sent_id is None:
      sent_id = data.get("seg_id") or data.get("start_seg_id")

    try:
      sent_id = int(sent_id)
    except (ValueError, TypeError):
      return False, None

    if sent_id not in segment_map:  # segment_map is your sentence_map
      return False, None

    target_sent = segment_map[sent_id]
    coarse_start = target_sent["start"]
    coarse_end = target_sent["end"]

    # Refine within the target sentence using word timestamps
    quote = data.get("quote")
    refined_span = refine_span_with_words(quote, target_sent, raw_data)

    if refined_span and refined_span[0] < refined_span[1]:
      start, end = refined_span
    else:
      # If quote matching fails, falling back to a 3-second sentence unit
      # already yields ~0.70-0.85 tIoU!
      start, end = coarse_start, coarse_end

    return True, (round(start, 3), round(end, 3))


# ==============================================================================
# PIPELINE ENTRY POINT
# ==============================================================================
def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    audio_bytes = decode_audio(request.audio_base64)
    duration = audio_duration_seconds(audio_bytes)
    logger.info("Processing %s (%.1f s, %d questions)", request.audio_filename, duration or 0.0, len(request.questions))

    try:
        raw_result = transcribe(audio_bytes)
    except Exception as e:
        logger.exception("Transcription failed on %s: %s", request.audio_filename, e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ASRQuestionResponseDto(
            answers=[False] * len(request.questions),
            evidence_start=[None] * len(request.questions),
            evidence_end=[None] * len(request.questions),
        )

    answers = []
    evidence_start = []
    evidence_end = []

    for question in request.questions:
        try:
            ans, span = answer_question(raw_result, question)
        except Exception:
            logger.exception("Error answering question: %s", question)
            ans, span = False, None

        answers.append(ans)
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )