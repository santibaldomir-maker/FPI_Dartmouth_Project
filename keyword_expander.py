"""
keyword_expander.py — AI-assisted vocabulary expansion for the FPI classifier.

Usage:
  # Expand all modes (auto-merges into vocabulary.yaml):
  python keyword_expander.py

  # Expand a single mode:
  python keyword_expander.py --mode FET

  # Dry-run (print suggestions, don't save):
  python keyword_expander.py --dry-run

Reads DARTMOUTH_API_KEY (or OPENAI_API_KEY) from environment.
Falls back to ANTHROPIC_API_KEY if set.
"""

import os
import sys
import argparse
import json
import yaml
from datetime import date
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_PATH = Path(__file__).parent / "vocabulary.yaml"
TODAY = date.today().isoformat()

# ── LLM setup (reuse existing project endpoint) ───────────────────────────────
def _build_llm():
    """Return a callable(prompt) -> str using whatever API key is available."""
    dart_key = os.getenv("DARTMOUTH_API_KEY", "")
    oai_key  = os.getenv("OPENAI_API_KEY", "")
    anth_key = os.getenv("ANTHROPIC_API_KEY", "")

    if dart_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="anthropic.claude-haiku-4-5-20251001",
            api_key=dart_key,
            base_url="https://chat.dartmouth.edu/api",
            temperature=0.4,
            max_tokens=1200,
        )
        def call(prompt):
            return llm.invoke(prompt).content
        print("LLM: Dartmouth/Claude Haiku")
        return call

    if oai_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=oai_key, temperature=0.4, max_tokens=1200)
        def call(prompt):
            return llm.invoke(prompt).content
        print("LLM: OpenAI gpt-4o-mini")
        return call

    if anth_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anth_key)
        def call(prompt):
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        print("LLM: Anthropic SDK")
        return call

    raise RuntimeError(
        "No API key found. Set DARTMOUTH_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
    )

# ── Lemmatizer (lightweight, no NLTK required) ────────────────────────────────
def _simple_normalize(phrase: str) -> str:
    """Lowercase + strip punctuation for deduplication."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()

# ── Core expansion logic ──────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a safety incident classification expert specializing in construction and industrial incidents.

Fatality mode: {label} ({code})
Energy type: {energy}
Existing keywords: {existing}

Generate 20–30 NEW keyword phrases that workers or supervisors might write in incident reports to describe this fatality mode. Include:
- Industry jargon and slang
- Common misspellings or shorthand (e.g. "electrocuted" vs "electric shock")
- Passive and active voice variants
- Regional/trade-specific terminology
- Equipment brand names commonly used as generic terms (e.g. "JLG" for aerial lift)
- Abbreviated forms (e.g. "CS entry" for confined space entry)

CRITICAL rules:
- Each phrase must be 1–6 words
- Do NOT repeat any existing keyword (listed above)
- Do NOT include full sentences
- Return ONLY a JSON array of strings, no explanation, no markdown

Example output format:
["phrase one", "phrase two", "phrase three"]
"""

def expand_mode(mode: dict, call_llm) -> list[dict]:
    """Call LLM to get new keywords for one mode. Returns list of new keyword dicts."""
    existing = [k["phrase"] for k in mode.get("keywords", [])]
    existing_normalized = {_simple_normalize(p) for p in existing}

    prompt = PROMPT_TEMPLATE.format(
        label=mode["label"],
        code=mode["code"],
        energy=mode["energy"],
        existing=", ".join(existing[:30]),  # send first 30 to keep prompt small
    )

    try:
        raw = call_llm(prompt)
        # Extract JSON array from response (handle markdown code blocks)
        import re
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not json_match:
            print(f"  WARNING: Could not parse JSON for {mode['code']}")
            return []
        candidates = json.loads(json_match.group())
    except Exception as e:
        print(f"  ERROR expanding {mode['code']}: {e}")
        return []

    new_keywords = []
    for phrase in candidates:
        phrase = str(phrase).strip().lower()
        if not phrase or len(phrase) > 80:
            continue
        if _simple_normalize(phrase) in existing_normalized:
            continue  # deduplicate
        existing_normalized.add(_simple_normalize(phrase))
        new_keywords.append({
            "phrase": phrase,
            "source": "ai_generated",
            "added": TODAY,
        })

    return new_keywords

# ── YAML helpers ──────────────────────────────────────────────────────────────
def load_vocab() -> dict:
    with open(VOCAB_PATH, "r") as f:
        return yaml.safe_load(f)

def save_vocab(vocab: dict):
    with open(VOCAB_PATH, "w") as f:
        yaml.dump(vocab, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI-assisted vocabulary expansion")
    parser.add_argument("--mode", help="Expand only this mode code (e.g. FET)")
    parser.add_argument("--dry-run", action="store_true", help="Print suggestions, don't save")
    args = parser.parse_args()

    call_llm = _build_llm()
    vocab    = load_vocab()
    modes    = vocab["modes"]

    print(f"\nVocabulary file: {VOCAB_PATH}")
    print(f"Modes to process: {len(modes)}\n")
    print("=" * 60)

    report = []

    for mode in modes:
        if args.mode and mode["code"].upper() != args.mode.upper():
            continue

        before = len(mode.get("keywords", []))
        print(f"[{mode['code']}] {mode['label']} — {before} keywords → expanding...")

        new_kws = expand_mode(mode, call_llm)
        after   = before + len(new_kws)

        if not args.dry_run:
            mode["keywords"] = mode.get("keywords", []) + new_kws

        report.append({
            "code":    mode["code"],
            "label":   mode["label"],
            "before":  before,
            "added":   len(new_kws),
            "after":   after,
        })

        print(f"  ✓ +{len(new_kws)} new keywords  (total: {after})")
        if args.dry_run and new_kws:
            for k in new_kws[:10]:
                print(f"    • {k['phrase']}")

    if not args.dry_run:
        save_vocab(vocab)
        print(f"\n✓ vocabulary.yaml updated")

    # Summary report
    print("\n" + "=" * 60)
    print(f"{'MODE':<10} {'LABEL':<40} {'BEFORE':>6} {'ADDED':>6} {'AFTER':>6}")
    print("-" * 60)
    total_before = total_added = 0
    for r in report:
        print(f"{r['code']:<10} {r['label'][:38]:<40} {r['before']:>6} {r['added']:>6} {r['after']:>6}")
        total_before += r["before"]
        total_added  += r["added"]
    print("-" * 60)
    print(f"{'TOTAL':<10} {'':<40} {total_before:>6} {total_added:>6} {total_before+total_added:>6}")
    print("=" * 60)

    if args.dry_run:
        print("\n(dry-run: no changes saved)")

if __name__ == "__main__":
    main()
