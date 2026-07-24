"""Load and slice the official annotation guidelines for prompting."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from emotion_cls.config import REPO_ROOT, resolve_path

DEFAULT_GUIDELINES_PATH = REPO_ROOT / "Datasets" / "guidelines" / "Annotation Guidelines.txt"

EMOTION_SECTION_ORDER = [
    "Joy",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger",
    "Anticipation",
]

_EXAMPLE_LINE = re.compile(r"^\[Ex\d+\]", re.IGNORECASE)


@lru_cache(maxsize=4)
def load_guidelines_text(path: str | None = None) -> str:
    p = resolve_path(path) if path else DEFAULT_GUIDELINES_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Annotation guidelines not found at {p}. "
            "Expected Datasets/guidelines/Annotation Guidelines.txt"
        )
    return p.read_text(encoding="utf-8")


def _slice_between(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    chunk = text[i:j] if j >= 0 else text[i:]
    return chunk.strip()


def general_plutchik_definitions(text: str | None = None) -> str:
    text = text or load_guidelines_text()
    return _slice_between(
        text,
        "According to this taxonomy, emotions are defined as follows:",
        "Applying Plutchik",
    )


def annotation_instructions(text: str | None = None) -> str:
    text = text or load_guidelines_text()
    chunk = _slice_between(text, "Annotation instructions", "References")
    return chunk.strip()


def emotion_section(emotion: str, text: str | None = None) -> str:
    """Full domain section for one Plutchik emotion (definitions + guideline examples)."""
    text = text or load_guidelines_text()
    # Sections are headed by the emotion name alone on a line (after "Applying Plutchik...")
    apply = text.find("Applying Plutchik")
    body = text[apply:] if apply >= 0 else text
    # Find this emotion heading and the next emotion / Annotation instructions
    pattern = rf"(?m)^{re.escape(emotion)}\s*$"
    m = re.search(pattern, body)
    if not m:
        return ""
    start = m.start()
    next_starts = []
    for other in EMOTION_SECTION_ORDER:
        if other == emotion:
            continue
        om = re.search(rf"(?m)^{re.escape(other)}\s*$", body[m.end() :])
        if om:
            next_starts.append(m.end() + om.start())
    instr = body.find("Annotation instructions", m.end())
    if instr >= 0:
        next_starts.append(instr)
    end = min(next_starts) if next_starts else len(body)
    return body[start:end].strip()


def strip_guideline_examples(section: str) -> str:
    """Remove [ExN] lines for zero-shot (definition-only) prompting."""
    lines = []
    for line in section.splitlines():
        if _EXAMPLE_LINE.match(line.strip()):
            continue
        lines.append(line)
    # Collapse excess blank lines
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def neutral_definition() -> str:
    return (
        "Neutral → to annotate sentences which are either too short to infer any emotion "
        "or reflect purely descriptive, objective observations. This includes the situation "
        "where a sentence does not include any emotion linked to the current state of the app."
    )


def classification_guideline_block(
    emotions: list[str],
    strategy: str,
    *,
    guidelines_path: str | None = None,
) -> str:
    """Build the guideline text embedded in decoder classification prompts."""
    text = load_guidelines_text(guidelines_path)
    parts: list[str] = []
    parts.append(general_plutchik_definitions(text))
    parts.append("")
    parts.append("Domain-specific definitions for mobile app reviews:")
    for e in emotions:
        if e == "Neutral":
            parts.append(f"### Neutral\n{neutral_definition()}")
            continue
        if e == "Reject":
            continue
        section = emotion_section(e, text)
        if strategy in {"zero_shot", "zeroShoot"}:
            section = strip_guideline_examples(section)
        if section:
            parts.append(f"### {e}\n{section}")
    parts.append("")
    parts.append(annotation_instructions(text))
    return "\n\n".join(p for p in parts if p is not None).strip()


def generation_definition(
    emotion: str,
    strategy: str,
    *,
    guidelines_path: str | None = None,
) -> str:
    """Per-emotion text for synthetic review generation."""
    if emotion == "Neutral":
        return neutral_definition()
    text = load_guidelines_text(guidelines_path)
    section = emotion_section(emotion, text)
    if strategy in {"zero_shot", "zeroShoot"}:
        return strip_guideline_examples(section) or emotion
    return section or emotion
