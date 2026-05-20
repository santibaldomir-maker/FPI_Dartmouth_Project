"""
classifier.py — FPI incident classification engine.

Classification pipeline (in order):
  1. Exact keyword match against vocabulary.yaml (case-insensitive substring)
  2. Fuzzy match via rapidfuzz if no exact hit (configurable threshold)
  3. LLM fallback if fuzzy also fails (optional, disabled by default)

Unclassified records are logged to unclassified_log.jsonl.
"""

import os
import re
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_PATH          = Path(__file__).parent / "vocabulary.yaml"
UNCLASSIFIED_LOG    = Path(__file__).parent / "unclassified_log.jsonl"
FUZZY_THRESHOLD     = int(os.getenv("FPI_FUZZY_THRESHOLD", "75"))   # 0-100
FUZZY_MIN_LEN       = 4    # don't fuzzy-match tokens shorter than this
LLM_FALLBACK        = os.getenv("FPI_LLM_FALLBACK", "false").lower() == "true"

# ── Load vocabulary ────────────────────────────────────────────────────────────
def _load_vocab() -> list[dict]:
    with open(VOCAB_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data["modes"]

_MODES: list[dict] = _load_vocab()

def reload_vocab():
    """Reload vocabulary from disk (call after keyword_expander updates the YAML)."""
    global _MODES
    _MODES = _load_vocab()


# ── Pre-build keyword index ────────────────────────────────────────────────────
def _build_index(modes: list[dict]) -> list[tuple[str, str, str, str, int]]:
    """Return list of (phrase, code, label, energy, base_score) sorted longest-first."""
    index = []
    for m in modes:
        for kw in m.get("keywords", []):
            phrase = kw["phrase"].lower().strip()
            index.append((phrase, m["code"], m["label"], m["energy"], m["base_score"]))
    # longest phrase first → prevents short phrases from blocking longer ones
    return sorted(index, key=lambda x: len(x[0]), reverse=True)

_INDEX = _build_index(_MODES)
_PHRASES: list[str] = [row[0] for row in _INDEX]
_PHRASE_TO_IDX: dict[str, int] = {row[0]: i for i, row in enumerate(_INDEX)}

# Pre-import rapidfuzz once at module level (avoids repeated import overhead)
try:
    from rapidfuzz import fuzz as _fuzz, process as _rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


_PUNCT_RE = re.compile(r"[,;:()\-]+")

def _normalize(text: str) -> str:
    """Lowercase and collapse punctuation clusters to a space for matching."""
    return _PUNCT_RE.sub(" ", text.lower())


# ── Exact match ───────────────────────────────────────────────────────────────
def _exact_match(text: str) -> list[tuple[str, str, str]]:
    """Return list of (code, label, energy) for all exact keyword hits in text."""
    t = _normalize(text)
    hits = []
    seen_codes = set()
    for phrase, code, label, energy, _ in _INDEX:
        if code not in seen_codes and phrase in t:
            hits.append((code, label, energy))
            seen_codes.add(code)
    return hits


# ── Fuzzy match ────────────────────────────────────────────────────────────────
def _fuzzy_match(text: str, threshold: int = FUZZY_THRESHOLD) -> Optional[tuple[str, str, str, float]]:
    """
    Single-call fuzzy match using token_set_ratio on the full text.
    Returns (code, label, energy, score) for best match above threshold, or None.
    token_set_ratio is designed for variable-length texts and handles tokenization
    internally — no need to manually iterate n-grams.
    """
    if not _HAS_RAPIDFUZZ:
        return None

    result = _rf_process.extractOne(
        _normalize(text), _PHRASES, scorer=_fuzz.token_set_ratio
    )
    if result and result[1] >= threshold:
        idx = _PHRASE_TO_IDX[result[0]]
        _, code, label, energy, _ = _INDEX[idx]
        return code, label, energy, float(result[1])

    return None


# ── LLM fallback ──────────────────────────────────────────────────────────────
_LLM_CALL = None

def _init_llm():
    global _LLM_CALL
    if _LLM_CALL is not None:
        return
    dart_key = os.getenv("DARTMOUTH_API_KEY", "")
    oai_key  = os.getenv("OPENAI_API_KEY", "")
    if dart_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="anthropic.claude-haiku-4-5-20251001",
            api_key=dart_key,
            base_url="https://chat.dartmouth.edu/api",
            temperature=0.0,
            max_tokens=50,
        )
        _LLM_CALL = lambda p: llm.invoke(p).content.strip()
    elif oai_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=oai_key, temperature=0.0, max_tokens=50)
        _LLM_CALL = lambda p: llm.invoke(p).content.strip()


def _llm_classify(text: str) -> Optional[tuple[str, str, str]]:
    """Ask LLM to pick the best mode code. Returns (code, label, energy) or None."""
    if not LLM_FALLBACK:
        return None
    try:
        _init_llm()
        if _LLM_CALL is None:
            return None
        codes = ", ".join(m["code"] for m in _MODES)
        prompt = (
            f"Classify this industrial incident into exactly one of these codes: {codes}\n\n"
            f"Incident: {text[:400]}\n\n"
            f"Reply with ONLY the code, nothing else."
        )
        code = _LLM_CALL(prompt).strip().upper()
        for m in _MODES:
            if m["code"] == code:
                return code, m["label"], m["energy"]
    except Exception as e:
        logger.warning(f"LLM fallback failed: {e}")
    return None


# ── Unclassified logger ───────────────────────────────────────────────────────
def _log_unclassified(text: str, method_tried: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "method_tried": method_tried,
        "text_snippet": text[:300],
    }
    with open(UNCLASSIFIED_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Public API ────────────────────────────────────────────────────────────────
def classify(event_title: str = "", narrative: str = "",
             fuzzy: bool = True) -> list[tuple[str, str, str]]:
    """
    Classify an incident. Returns list of (code, label, energy) — primary match first.
    Falls back to [('OTHER', 'Other', 'Other')] if nothing matches.

    Args:
        event_title: short event title / event type field
        narrative:   full incident description
        fuzzy:       enable fuzzy matching fallback (default True)
    """
    text = (str(event_title) + " " + str(narrative)).strip()
    if not text:
        return [("OTHER", "Other", "Other")]

    # Stage 1: exact keyword match
    hits = _exact_match(text)
    if hits:
        return hits

    # Stage 2: fuzzy match
    if fuzzy:
        result = _fuzzy_match(text)
        if result:
            code, label, energy, score = result
            logger.debug(f"Fuzzy match: {code} @ {score:.0f}%  → '{text[:60]}'")
            return [(code, label, energy)]

    # Stage 3: LLM fallback (disabled by default)
    llm_result = _llm_classify(text)
    if llm_result:
        return [llm_result]

    # Unclassified
    method = "exact+fuzzy" if fuzzy else "exact"
    if LLM_FALLBACK:
        method += "+llm"
    _log_unclassified(text, method)
    return [("OTHER", "Other", "Other")]


def get_mode_base_score(code: str) -> int:
    """Return the base physics score (1-3) for a given mode code."""
    for m in _MODES:
        if m["code"] == code:
            return m["base_score"]
    return 2  # default mid


def get_mode_info(code: str) -> Optional[dict]:
    """Return full mode dict for a given code."""
    for m in _MODES:
        if m["code"] == code:
            return m
    return None


def vocab_summary() -> list[dict]:
    """Return per-mode keyword counts for reporting."""
    return [
        {"code": m["code"], "label": m["label"], "keyword_count": len(m.get("keywords", []))}
        for m in _MODES
    ]
