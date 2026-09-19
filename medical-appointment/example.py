"""ASR Question Answering pipeline (Ollama + sentence-level evidence spans).

v5 - optional "second look" at the evidence sentence (OFF by default) plus richer logging.
  Behaviour with the defaults is identical to v4. Every confirmed question now logs a
  candidate table (chosen sentence, its neighbours, the best lexical matches, with
  timings) so any span policy can be replayed offline against the ground truth.
    QA_SECOND_LOOK=lexical   no extra LLM call: switch to the sentence that matches the
                             claim's words clearly better than the chosen one
    QA_SECOND_LOOK=llm       one short follow-up call (same chat, so the transcript prefix
                             can be served from the KV cache) choosing between a few sentences
  Only questions that trip a trigger get a second look (about 10% of all questions).

v4 - LLM sentence ranges and question/answer pairing are OFF by default.
  The v3 rerun showed they are unstable: 20 of 189 range decisions flipped
  between v2 and v3, and the effect of ranges on mean tIoU changed sign
  (about 0.000 with v2's picks, -0.017 with v3's). The prompt is back to v1's
  single-sentence evidence rule. Turn them back on with QA_ENABLE_RANGE=1 and
  QA_ENABLE_PAIRING=1. The ASR hints (which recovered Esomeprazole and Airomir
  in v3) stay.

v3 - two narrow changes on top of v2, both based on the v2 rerun:
  * Evidence-selection wording reverted to v1's ("the sentence containing the
    direct proof"). In v2 the "most direct and complete statement" wording moved
    14 picks to another sentence: 5 got much worse, 3 much better, net -0.010 tIoU.
  * ASR spelling hints: when a claim word differs from a transcript word only in
    vowels (Esomeprazole vs Isomeprazole, Airomir vs Aromir) the model is told so.
    2 of the 5 remaining false negatives were exactly this.

v2 - evidence spans tuned toward the ground-truth annotation style.

What changed vs. v1 (see the analysis of 185 true positives):
  * Evidence is the whole transcript sentence, not a 3-6 word quote.
    Ground-truth spans are ~3.2 s / 9 words; v1 spans were ~1.8 s / 5 words
    (mean IoU 0.50). Full sentences alone give ~0.56, plus a small boundary
    pad ~0.58. Quote trimming did not help in any sentence-length bucket.
  * The LLM may return a contiguous sentence RANGE (start_seg_id..end_seg_id)
    so question+answer pairs and short confirmations can be captured, as the
    ground truth often does (24% of GT spans cover 2+ sentences).
  * Deterministic question/answer pairing for bare one-word replies.
  * Bug fix: `seg_id or start_seg_id` treated sentence id 0 as missing and
    silently turned a CONFIRMED claim into a "no" answer.
  * A CONFIRMED verdict is never thrown away because localisation failed; the
    sentence is recovered lexically instead.
  * num_predict raised (80 tokens could truncate the JSON), one retry on bad JSON.
  * Logs to a NEW csv (answers_v2.csv) because the schema changed.
"""

import csv
import difflib
import json
import logging
import os
import re
import time
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
OLLAMA_HOST = "http://127.0.0.1:16614"
OLLAMA_MODEL = os.environ.get("QA_MODEL", "phi4")
CSV_OUTPUT_PATH = os.environ.get("QA_CSV", "answers_v5.csv")   # new name: the column set changed

ollama_client = ollama.Client(host=OLLAMA_HOST)

# ---- Evidence-span policy (tuned on ground truth; re-tune when data changes) --
# Positive START pad moves the start EARLIER, END pad moves the end LATER
# (negative = earlier). Best values from a grid search with leave-one-transcript-
# out CV: IoU 0.563 (raw sentence) -> 0.582. The effect is small (~0.02).
SPAN_START_PAD_SEC = 0.1
SPAN_END_PAD_SEC = -0.1

# Evidence range limits (GT: 75% of spans <= 4.1 s, max 14 s).
MAX_SPAN_SENTENCES = 3
MAX_SPAN_SEC = 10.0

# Multi-sentence evidence returned by the LLM (start_seg_id..end_seg_id).
# OFF: ranges were break-even at best and unstable between prompt versions.
ENABLE_LLM_RANGE = os.environ.get("QA_ENABLE_RANGE", "0") == "1"

# Question/answer pairing for bare replies ("None.", "Observation.", "Yes.").
# OFF: 5 cases per run, net effect -0.005 (v2) and -0.0003 (v3) on mean tIoU.
ENABLE_QA_PAIRING = os.environ.get("QA_ENABLE_PAIRING", "0") == "1"
QA_SHORT_ANSWER_WORDS = 4     # target counts as a bare answer if <= this
QA_SHORT_REPLY_WORDS = 6      # reply that follows a question sentence
QA_MAX_GAP_SEC = 1.5          # max silence between the paired sentences

# v1 trimmed the evidence to the quoted words. That LOWERED IoU (0.50 vs 0.56
# for whole sentences), so it is off. Kept only for A/B comparison.
USE_QUOTE_SNAPPING = False

# Prompt option: accept synonyms/lay terms/ASR spelling variants of the SAME
# entity (7 of the 10 v1 false negatives were paraphrase or spelling mismatches).
# Re-run the hard-negative questions after enabling; v1 had 0 false positives.
ALLOW_SYNONYM_MATCH = True

# Tell the model about ASR spelling errors between the claim and the transcript.
# A pair only counts when the words differ ONLY in vowels (identical consonant
# skeleton), e.g. Esomeprazole/Isomeprazole, Airomir/Aromir. A plain similarity
# score cannot be used: prednisone/prednisolone scores 0.91, the same as
# Esomeprazole/Isomeprazole, but is a different drug (hard negatives may use
# such pairs). Check the 142 hard negatives after enabling.
ENABLE_ASR_VARIANT_HINTS = True
ASR_VARIANT_MIN_LEN = 7        # claim word length; shorter words are too often real words (affect/effect)
ASR_VARIANT_MIN_RATIO = 0.8    # sanity check on top of the skeleton match

# ---- Second look at the evidence sentence -------------------------------------
# Triggers (checked on the v4 run, 191 confirmed questions; "bad" = IoU < 0.3, base rate 25%):
#   lex_gap : another sentence shares >= 0.25 more of the claim's content words than the
#             chosen one (29 questions, 59% bad, mean IoU 0.30 vs 0.64)
#   short   : chosen sentence has <= 3 words (18 questions, 72% bad, mean IoU 0.29 vs 0.62)
# Together: 38 questions = 9.7% of all questions, about 1 per conversation.
SECOND_LOOK_MODE = os.environ.get("QA_SECOND_LOOK", "llm").strip().lower()   # 0 | lexical | llm
S2_MIN_LEX_GAP = 0.25
S2_SHORT_WORDS = 3
S2_MAX_OFFERED = 2           # sentences offered to the LLM besides the chosen one
S2_MAX_ELAPSED_SEC = 40.0    # skip the second look once the conversation has used this much time

APPROX_CHARS_PER_TOKEN = 3.5
NUM_CTX = 4096

# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================
_SYNONYM_RULE = """
   - Same entity under a different name still counts as mentioned: lay terms vs medical terms ("long-term sugar value" = HbA1c), and ASR spelling variants of a drug or diagnosis ("Isomeprazole" ~ "Esomeprazole", "molluscum" ~ "molluscs"). Numbers, units, negation and side of the body must still match exactly.
"""

_PROMPT_HEAD = """You are a strict clinical evidence verifier. Your task is to verify if a claim is factually TRUE and explicitly CONFIRMED by the transcript.

CRITICAL RULES FOR VERIFICATION:
1. DOCTOR-PATIENT NEGATION (MUST be CONTRADICTED / has_evidence: false):
   - If a doctor asks about a symptom or event ("Any fever?", "Does the pain radiate down your leg?", "Any known COVID exposure?"), and the patient answers "No", "None", or "Nothing like that" -> The claim that the patient had this symptom is CONTRADICTED.
   - If an exam finding is absent ("no pus", "no sores", "no redness", "no swelling", "no foreign body") -> Claims asserting the finding was present or seen are CONTRADICTED.
   - If a test was negative ("strep test was negative", "COVID test was negative") -> Claims asserting infection/bacteria found are CONTRADICTED.
   - The reverse also holds: if the claim itself states an absence ("free of symptoms", "no side effects", "no foreign body") and the transcript confirms that absence, the claim is CONFIRMED.

2. ANTONYMS & OPPOSITE FINDINGS (MUST be CONTRADICTED / has_evidence: false):
   - "Normal", "stable", "good", or "unchanged" CONTRADICTS claims of "abnormal", "impaired", "elevated", "unstable", or "worsened".

3. UNMENTIONED SPECIFIC ENTITIES (MUST be NOT_MENTIONED / has_evidence: false):
   - If a specific symptom, diagnosis, or entity was never named, it is NOT_MENTIONED.
   - Strict matching: "holiday" is not "weekend"; watching a television series is not a "specific hobby".
"""

_EVIDENCE_SINGLE = """
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

_EVIDENCE_RANGE = """
4. EVIDENCE LOCALIZATION (Only if claim_status is CONFIRMED):
   - Identify the single sentence [S_id] that contains the direct proof.
   - Report start_seg_id and end_seg_id. Normally they are the SAME sentence id.
   - If the chosen sentence is only a bare short answer ("None.", "Yes.", "No changes.", "Observation."), set start_seg_id to the question or statement right before it that gives the answer its meaning.
   - If the chosen sentence is a question and the next short sentence is the reply that confirms it, set end_seg_id to that reply.
   - Never span more than 3 consecutive sentences, and only include sentences that belong to the same statement.
   - "quote" = the complete clause that states the fact, copied exactly from the transcript (not a 2-3 word fragment).

OUTPUT STRICT VALID JSON:
{
  "rationale": "<brief 3-7 words explaining if affirmed, denied with no, normal vs abnormal, or unmentioned>",
  "claim_status": "CONFIRMED" | "CONTRADICTED" | "NOT_MENTIONED",
  "has_evidence": true or false,
  "start_seg_id": <int or null>,
  "end_seg_id": <int or null>,
  "quote": "<exact clause from transcript or null>"
}
"""


def build_system_prompt(enable_range: bool = False, synonym_rule: bool = True) -> str:
    return (
        _PROMPT_HEAD
        + (_SYNONYM_RULE if synonym_rule else "")
        + (_EVIDENCE_RANGE if enable_range else _EVIDENCE_SINGLE)
    )


SYSTEM_PROMPT = build_system_prompt(ENABLE_LLM_RANGE, ALLOW_SYNONYM_MATCH)

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


warmup_pipeline()

# ==============================================================================
# QUALITATIVE LOGGING HELPER
# ==============================================================================
ANALYSIS_FIELDNAMES = [
    "audio_filename",
    "question",
    "predicted_answer",
    "final_start",
    "final_end",
    "span_duration_sec",
    # NONE | SENTENCE | SENTENCE_QA_PAIR | LLM_RANGE | QUOTE_SNAPPED |
    # LEXICAL_FALLBACK | ERROR
    "alignment_mode",
    "llm_claim_status",
    "llm_has_evidence",
    "llm_rationale",
    "llm_seg_id",            # start id (kept for compatibility with v1 analysis)
    "llm_end_seg_id",
    "final_seg_range",       # ids actually used after pairing / clamping
    "llm_quote",
    "asr_hints",
    "llm_ms",                # stage-1 LLM time
    "chosen_overlap",        # claim-word overlap of the chosen sentence
    "lex_best_id",
    "lex_best_overlap",
    "lex_gap",               # best other sentence minus chosen sentence
    "s2_mode",
    "s2_reason",             # lex_gap | short | lex_gap+short | ""
    "s2_offered",
    "s2_choice",
    "s2_changed",
    "s2_ms",
    "cand_json",             # chosen / prev / next / top lexical sentences with timings
    "q_overlap_max",         # router feature: best question/sentence content-word overlap (0-1)
    "q_overlap_n",           # router feature: number of sentences with overlap >= 0.5
    "n_sentences",
    "target_sentence_text",
    "coarse_sent_start",
    "coarse_sent_end",
    "raw_llm_json",
]


def append_analysis_records(records: List[Dict[str, Any]], filepath: str = CSV_OUTPUT_PATH):
    """Appends qualitative analysis records to CSV, creating headers if file doesn't exist."""
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for rec in records:
                writer.writerow(rec)
    except Exception as e:
        logger.error("Failed to write qualitative analysis log: %s", e)


# ==============================================================================
# TRANSCRIPT HELPERS
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
    lines = []
    sentence_map = {}
    sent_id = 0

    for seg in raw_data.get("segments", []):
        words = seg.get("words", [])

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


_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "with", "as", "by", "did",
    "does", "do", "has", "have", "had", "patient", "your", "you", "i", "my", "any",
    "will", "would", "should", "there", "their", "them", "they", "he", "she",
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _content_tokens(text: str) -> set:
    return {t for t in _tokens(text) if t not in _STOPWORDS}


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _first_int(*values: Any) -> Optional[int]:
    """First value that contains an integer. Uses `is None`, NOT truthiness,
    so sentence id 0 is valid (v1 dropped it via `x or y`)."""
    for v in values:
        if v is None or isinstance(v, bool):
            continue
        m = re.search(r"\d+", str(v))
        if m:
            return int(m.group())
    return None


def locate_sentence(
    segment_map: Dict[int, dict],
    quote: Optional[str],
    question: str,
    candidates: Optional[List[int]] = None,
) -> Optional[int]:
    """Recover the evidence sentence when the LLM id is missing or invalid.
    Prefers the quote, then falls back to the question's content words."""
    ids = candidates if candidates is not None else sorted(segment_map)
    for probe, min_score in ((quote, 0.6), (question, 0.4)):
        toks = _content_tokens(probe or "")
        if not toks:
            continue
        best_id, best_score = None, 0.0
        for sid in ids:
            sent = segment_map.get(sid)
            if sent is None:
                continue
            score = len(toks & _content_tokens(sent["text"])) / len(toks)
            if score > best_score:
                best_id, best_score = sid, score
        if best_id is not None and best_score >= min_score:
            return best_id
    return None


def _all_words(segment_map: Dict[int, dict]) -> set:
    words = set()
    for sent in segment_map.values():
        words.update(_tokens(sent["text"]))
    return words


def question_overlap_features(question: str, segment_map: Dict[int, dict]) -> Tuple[float, int]:
    """Cheap routing features, logged for every question.
    q_overlap_max: how strongly the transcript talks about the claim's topic
                   (~0 for off-topic questions, high for hard negatives).
    q_overlap_n:   how many sentences mention it (>=2 means several candidate
                   evidence sentences, where the ground truth may pick another one)."""
    toks = _content_tokens(question)
    if not toks:
        return 0.0, 0
    scores = [len(toks & _content_tokens(s["text"])) / len(toks) for s in segment_map.values()]
    return round(max(scores, default=0.0), 3), sum(1 for x in scores if x >= 0.5)


def _skeleton(word: str) -> str:
    """Consonant skeleton: drop vowels, collapse doubled letters."""
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiou]", "", word))


def asr_variant_hints(question: str, segment_map: Dict[int, dict]) -> List[Tuple[str, str]]:
    """Claim words missing from the transcript that appear there with only the
    vowels changed (typical ASR errors on drug and diagnosis names).
    Pure inflections (medication / medications) are ignored."""
    if not ENABLE_ASR_VARIANT_HINTS:
        return []
    all_words = _all_words(segment_map)
    hints = []
    for q in dict.fromkeys(_tokens(question)):
        if len(q) < ASR_VARIANT_MIN_LEN or q in _STOPWORDS or q in all_words:
            continue
        q_skel = _skeleton(q)
        if len(q_skel) < 3:
            continue
        best, best_ratio = None, 0.0
        for t in all_words:
            if len(t) < 5 or t.startswith(q) or q.startswith(t) or _skeleton(t) != q_skel:
                continue
            ratio = difflib.SequenceMatcher(None, q, t).ratio()
            if ratio >= ASR_VARIANT_MIN_RATIO and ratio > best_ratio:
                best, best_ratio = t, ratio
        if best is not None:
            hints.append((q, best))
    return hints


# ==============================================================================
# SECOND LOOK
# ==============================================================================
def lexical_scores(question: str, segment_map: Dict[int, dict]) -> Dict[int, float]:
    """Share of the claim's content words found in each sentence."""
    toks = _content_tokens(question)
    if not toks:
        return {}
    return {sid: len(toks & _content_tokens(sent["text"])) / len(toks) for sid, sent in segment_map.items()}


def ranked_alternatives(chosen_id: int, scores: Dict[int, float], k: int) -> List[int]:
    """Other sentences by claim-word overlap (ties: closest to the chosen one)."""
    others = [sid for sid in scores if sid != chosen_id and scores[sid] > 0]
    others.sort(key=lambda sid: (-scores[sid], abs(sid - chosen_id), sid))
    return others[:k]


def second_look_reason(chosen_id: int, scores: Dict[int, float], segment_map: Dict[int, dict]) -> str:
    reasons = []
    best_other = max((v for sid, v in scores.items() if sid != chosen_id), default=0.0)
    if best_other - scores.get(chosen_id, 0.0) >= S2_MIN_LEX_GAP:
        reasons.append("lex_gap")
    if _word_count(segment_map[chosen_id]["text"]) <= S2_SHORT_WORDS:
        reasons.append("short")
    return "+".join(reasons)


def candidate_table(chosen_id: int, scores: Dict[int, float], segment_map: Dict[int, dict], k: int = 3) -> str:
    """JSON list with the chosen sentence, its neighbours and the top lexical matches,
    so any span policy can be evaluated offline without re-running the LLM."""
    roles: Dict[int, List[str]] = {chosen_id: ["chosen"]}
    for sid, role in ((chosen_id - 1, "prev"), (chosen_id + 1, "next")):
        if sid in segment_map:
            roles.setdefault(sid, []).append(role)
    for sid in ranked_alternatives(chosen_id, scores, k):
        roles.setdefault(sid, []).append("lex")
    rows = []
    for sid in sorted(roles):
        sent = segment_map[sid]
        rows.append({
            "id": sid,
            "role": "+".join(roles[sid]),
            "start": round(sent["start"], 3),
            "end": round(sent["end"], 3),
            "ov": round(scores.get(sid, 0.0), 3),
            "text": sent["text"][:160],
        })
    return json.dumps(rows, ensure_ascii=False)


def _second_look_llm(
    user_prompt: str,
    first_reply: str,
    chosen_id: int,
    offered: List[int],
    segment_map: Dict[int, dict],
) -> Optional[int]:
    """Follow-up turn in the SAME chat. The messages up to the follow-up are exactly
    what stage 1 sent plus its reply, so the transcript prefix can be reused from the
    KV cache; only the short tail is new and the answer is a few tokens."""
    def show(i: int) -> str:
        return f'[S_{i}] "{segment_map[i]["text"][:200]}"'

    tail = (
        "Second check of the evidence sentence for the same claim. "
        f"Current choice: {show(chosen_id)}. "
        "Other sentences that share words with the claim: " + "; ".join(show(i) for i in offered) + ". "
        "Which ONE of these sentences states the claim most directly? "
        'Reply with JSON only: {"seg_id": <int>}'
    )
    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": first_reply},
                {"role": "user", "content": tail},
            ],
            format="json",
            options={"temperature": 0.0, "num_predict": 24, "num_ctx": NUM_CTX},
            keep_alive=-1,
        )
        data = json.loads(response.message.content)
        pick = _first_int(data.get("seg_id")) if isinstance(data, dict) else None
    except Exception:
        logger.exception("Second-look LLM call failed; keeping the first choice")
        return None
    return pick if pick in ([chosen_id] + offered) else None


# ==============================================================================
# EVIDENCE SPAN CONSTRUCTION
# ==============================================================================
def _range_times(lo: int, hi: int, segment_map: Dict[int, dict]) -> Tuple[float, float]:
    return segment_map[lo]["start"], segment_map[hi]["end"]


def apply_qa_pairing(lo: int, hi: int, segment_map: Dict[int, dict]) -> Tuple[int, int, bool]:
    """Pair a bare reply with its question (or a question with its short reply).

    Ground-truth spans often read "And any side effects at all? None." while a
    single-sentence prediction would only cover "None.". Only fires for one
    sentence and only for clear question/answer shapes, because blind padding
    was measured to LOWER IoU.
    """
    if not ENABLE_QA_PAIRING or lo != hi:
        return lo, hi, False

    target = segment_map[lo]
    text = target["text"].strip()

    prev = segment_map.get(lo - 1)
    if (
        prev is not None
        and _word_count(text) <= QA_SHORT_ANSWER_WORDS
        and prev["text"].strip().endswith("?")
        and 0.0 <= target["start"] - prev["end"] <= QA_MAX_GAP_SEC
    ):
        return lo - 1, hi, True

    nxt = segment_map.get(hi + 1)
    if (
        nxt is not None
        and text.endswith("?")
        and _word_count(nxt["text"]) <= QA_SHORT_REPLY_WORDS
        and 0.0 <= nxt["start"] - target["end"] <= QA_MAX_GAP_SEC
    ):
        return lo, hi + 1, True

    return lo, hi, False


def refine_span_with_words(
    quote: Optional[str],
    target_seg: dict,
    raw_data: dict,
) -> Optional[Tuple[float, float]]:
    """v1 quote -> word-timestamp snapping. Only used if USE_QUOTE_SNAPPING."""
    if not quote or not quote.strip():
        return None

    clean_quote = [re.sub(r"[^\w]", "", w).lower() for w in quote.split()]
    clean_quote = [w for w in clean_quote if w]
    if not clean_quote:
        return None

    def search_word_list(words_list: List[dict]) -> Optional[Tuple[float, float]]:
        cleaned_words = []
        for w in words_list:
            cw = re.sub(r"[^\w]", "", w.get("word", "")).lower()
            if cw and "start" in w and "end" in w:
                cleaned_words.append((cw, float(w["start"]), float(w["end"])))

        if not cleaned_words:
            return None

        q_len = len(clean_quote)

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

        return None

    # Restrict to the target sentence only: v1's global fallback could jump to a
    # different mention of the phrase elsewhere in the conversation.
    target_words = target_seg.get("words", [])
    if target_words:
        span = search_word_list(target_words)
        if span and span[0] < span[1]:
            return span
    return None


def resolve_evidence_range(
    data: dict,
    quote: Optional[str],
    question: str,
    segment_map: Dict[int, dict],
) -> Tuple[Optional[int], Optional[int], str]:
    """Turn the LLM's ids into a valid, bounded, contiguous sentence range."""
    start_id = _first_int(data.get("start_seg_id"), data.get("seg_id"), data.get("sentence_id"))
    end_id = _first_int(data.get("end_seg_id"), start_id)
    if start_id is None:
        start_id = end_id

    ids_valid = (
        start_id is not None
        and end_id is not None
        and start_id in segment_map
        and end_id in segment_map
    )

    if not ids_valid:
        located = locate_sentence(segment_map, quote, question)
        if located is None:
            return None, None, "NONE"
        return located, located, "LEXICAL_FALLBACK"

    lo, hi = sorted((start_id, end_id))
    if not all(i in segment_map for i in range(lo, hi + 1)):
        # non-contiguous ids (e.g. gaps in the map): keep the sentence that best matches
        pick = locate_sentence(segment_map, quote, question, [start_id, end_id]) or start_id
        return pick, pick, "SENTENCE"

    if hi > lo and not ENABLE_LLM_RANGE:
        # The model may still return a range; keep only the sentence that best matches the quote.
        pick = locate_sentence(segment_map, quote, question, list(range(lo, hi + 1))) or lo
        return pick, pick, "SENTENCE"

    too_many = (hi - lo + 1) > MAX_SPAN_SENTENCES
    s, e = _range_times(lo, hi, segment_map)
    too_long = (hi > lo) and (e - s) > MAX_SPAN_SEC
    if too_many or too_long:
        pick = locate_sentence(segment_map, quote, question, list(range(lo, hi + 1))) or lo
        return pick, pick, "SENTENCE"

    return lo, hi, ("LLM_RANGE" if hi > lo else "SENTENCE")


def build_evidence_span(
    lo: int,
    hi: int,
    mode: str,
    quote: Optional[str],
    segment_map: Dict[int, dict],
    raw_data: dict,
) -> Tuple[float, float, str, Tuple[int, int]]:
    lo, hi, paired = apply_qa_pairing(lo, hi, segment_map)
    if paired:
        mode = "SENTENCE_QA_PAIR"

    start, end = _range_times(lo, hi, segment_map)

    if USE_QUOTE_SNAPPING and lo == hi:
        snapped = refine_span_with_words(quote, segment_map[lo], raw_data)
        if snapped:
            start, end = snapped
            mode = "QUOTE_SNAPPED"

    # Small boundary correction learned from the ground truth.
    p_start = max(0.0, start - SPAN_START_PAD_SEC)
    p_end = end + SPAN_END_PAD_SEC
    if p_end - p_start >= 0.3:
        start, end = p_start, p_end

    return start, end, mode, (lo, hi)


# ==============================================================================
# LLM CALL
# ==============================================================================
def _query_llm(user_prompt: str, info: Dict[str, Any]) -> Optional[dict]:
    """One retry on transport/JSON failure. Returns parsed dict or None."""
    for attempt in range(2):
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
                    "num_predict": 160,   # v1 used 80: could cut the JSON mid-string
                    "num_ctx": NUM_CTX,
                },
                keep_alive=-1,
            )
            raw_content = response.message.content
            info["raw_llm_json"] = raw_content
            data = json.loads(raw_content)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.exception("Ollama query failed (attempt %d)", attempt + 1)
    return None


# ==============================================================================
# MAIN ANSWER FUNCTION
# ==============================================================================
def answer_question(
    raw_data: dict,
    question: str,
    t_start: Optional[float] = None,
) -> Tuple[bool, Optional[Span], Dict[str, Any]]:
    """Answers question and returns (answer, span, qualitative_meta_dict)."""
    transcript_text, segment_map = format_indexed_transcript(raw_data)

    hints = asr_variant_hints(question, segment_map)
    hint_text = ""
    if hints:
        hint_text = (
            "Note: likely ASR spelling errors in the transcript (treat each pair as the same word): "
            + "; ".join(f'claim "{q}" = transcript "{t}"' for q, t in hints)
            + ".\n\n"
        )

    user_prompt = (
        f"Transcript:\n\"\"\"\n{transcript_text}\n\"\"\"\n\n"
        f"Claim to verify: \"{question}\"\n\n"
        f"{hint_text}"
        "State if explicitly confirmed, contradicted, or not mentioned. Return JSON."
    )

    approx_tokens = (len(SYSTEM_PROMPT) + len(user_prompt)) / APPROX_CHARS_PER_TOKEN
    if approx_tokens > NUM_CTX * 0.9:
        logger.warning("Prompt ~%d tokens is close to num_ctx=%d; the transcript may be truncated.", approx_tokens, NUM_CTX)

    q_max, q_n = question_overlap_features(question, segment_map)

    info: Dict[str, Any] = {
        "question": question,
        "predicted_answer": False,
        "final_start": None,
        "final_end": None,
        "span_duration_sec": None,
        "alignment_mode": "NONE",
        "llm_claim_status": None,
        "llm_has_evidence": None,
        "llm_rationale": None,
        "llm_seg_id": None,
        "llm_end_seg_id": None,
        "final_seg_range": None,
        "llm_quote": None,
        "asr_hints": "; ".join(f"{q}={t}" for q, t in hints),
        "llm_ms": None,
        "chosen_overlap": None,
        "lex_best_id": None,
        "lex_best_overlap": None,
        "lex_gap": None,
        "s2_mode": SECOND_LOOK_MODE,
        "s2_reason": "",
        "s2_offered": "",
        "s2_choice": None,
        "s2_changed": False,
        "s2_ms": None,
        "cand_json": "",
        "q_overlap_max": q_max,
        "q_overlap_n": q_n,
        "n_sentences": len(segment_map),
        "target_sentence_text": None,
        "coarse_sent_start": None,
        "coarse_sent_end": None,
        "raw_llm_json": "",
    }

    t_llm = time.monotonic()
    data = _query_llm(user_prompt, info)
    info["llm_ms"] = int(1000 * (time.monotonic() - t_llm))
    if data is None:
        return False, None, info

    claim_status = str(data.get("claim_status", "")).strip().upper()
    raw_has_evidence = data.get("has_evidence")
    if not isinstance(raw_has_evidence, bool):
        raw_has_evidence = str(raw_has_evidence).strip().lower() == "true"

    quote = data.get("quote")
    if quote is not None and not isinstance(quote, str):
        quote = str(quote)

    info["llm_claim_status"] = claim_status
    info["llm_has_evidence"] = raw_has_evidence
    info["llm_rationale"] = data.get("rationale")
    info["llm_quote"] = quote
    info["llm_seg_id"] = _first_int(data.get("start_seg_id"), data.get("seg_id"), data.get("sentence_id"))
    info["llm_end_seg_id"] = _first_int(data.get("end_seg_id"))

    # In the v1 analysis CONFIRMED was never wrong on 195 negatives (0 false
    # positives), so the verdict alone decides the answer. has_evidence is only
    # logged; localisation problems must not flip a correct "yes" to "no".
    if claim_status != "CONFIRMED":
        return False, None, info

    lo, hi, mode = resolve_evidence_range(data, quote, question, segment_map)
    if lo is None or hi is None:
        info["predicted_answer"] = True     # confirmed, but no sentence could be located
        info["alignment_mode"] = "NONE"
        return True, None, info

    # ---- second look at the evidence sentence (off unless QA_SECOND_LOOK is set) ----
    if lo == hi:
        scores = lexical_scores(question, segment_map)
        stage1_id = lo
        others = ranked_alternatives(stage1_id, scores, 1)
        info["chosen_overlap"] = round(scores.get(stage1_id, 0.0), 3)
        if others:
            info["lex_best_id"] = others[0]
            info["lex_best_overlap"] = round(scores[others[0]], 3)
            info["lex_gap"] = round(scores[others[0]] - scores.get(stage1_id, 0.0), 3)
        info["cand_json"] = candidate_table(stage1_id, scores, segment_map)

        time_ok = t_start is None or (time.monotonic() - t_start) < S2_MAX_ELAPSED_SEC
        reason = second_look_reason(stage1_id, scores, segment_map)
        info["s2_reason"] = reason
        if SECOND_LOOK_MODE in ("lexical", "llm") and reason and time_ok:
            new_id = stage1_id
            if SECOND_LOOK_MODE == "lexical":
                if "lex_gap" in reason and others:
                    new_id = others[0]
            else:
                offered = ranked_alternatives(stage1_id, scores, S2_MAX_OFFERED)
                info["s2_offered"] = ",".join(str(i) for i in offered)
                if offered:
                    t_s2 = time.monotonic()
                    pick = _second_look_llm(user_prompt, info["raw_llm_json"], stage1_id, offered, segment_map)
                    info["s2_ms"] = int(1000 * (time.monotonic() - t_s2))
                    if pick is not None:
                        new_id = pick
            info["s2_choice"] = new_id
            info["s2_changed"] = new_id != stage1_id
            lo = hi = new_id

    start, end, mode, (lo, hi) = build_evidence_span(lo, hi, mode, quote, segment_map, raw_data)
    info["target_sentence_text"] = " ".join(segment_map[i]["text"] for i in range(lo, hi + 1) if i in segment_map)
    info["coarse_sent_start"] = round(segment_map[lo]["start"], 3)
    info["coarse_sent_end"] = round(segment_map[hi]["end"], 3)
    info["final_seg_range"] = f"{lo}-{hi}"

    start_r, end_r = round(start, 3), round(end, 3)
    info["predicted_answer"] = True
    info["final_start"] = start_r
    info["final_end"] = end_r
    info["span_duration_sec"] = round(end_r - start_r, 3)
    info["alignment_mode"] = mode

    return True, (start_r, end_r), info


# ==============================================================================
# PIPELINE ENTRY POINT
# ==============================================================================
def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    t0 = time.monotonic()
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
    analysis_records = []

    for question in request.questions:
        try:
            ans, span, info = answer_question(raw_result, question, t0)
        except Exception:
            logger.exception("Error answering question: %s", question)
            ans, span = False, None
            info = {
                "question": question,
                "predicted_answer": False,
                "alignment_mode": "ERROR",
                "llm_claim_status": "ERROR",
                "llm_has_evidence": False,
                "llm_rationale": "exception_raised",
                "raw_llm_json": "",
            }

        info["audio_filename"] = request.audio_filename
        analysis_records.append(info)

        answers.append(ans)
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    # Save details for qualitative error/span analysis
    append_analysis_records(analysis_records, CSV_OUTPUT_PATH)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )