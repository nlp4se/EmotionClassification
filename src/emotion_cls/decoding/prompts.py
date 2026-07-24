"""Prompt builders for decoder-only classification and generation."""

from __future__ import annotations

from emotion_cls.guidelines import (
    classification_guideline_block,
    generation_definition,
)
from emotion_cls.resources.definitions import STRATEGY_ALIASES


def normalize_strategy(strategy: str) -> str:
    return STRATEGY_ALIASES.get(strategy, strategy)


def classification_messages(
    sentence: str,
    emotions: list[str],
    strategy: str,
    *,
    few_shot_examples: dict[str, list[str]] | None = None,
    guidelines_path: str | None = None,
    review_context: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for multi-label emotion classification.

    Strategies:
      - zero_shot: guidelines without [ExN] examples
      - few_shot_guidelines: full guideline emotion sections (with examples)
      - few_shot_guidelines_dataset: guidelines + labelled sentences from the training fold
    """
    strategy = normalize_strategy(strategy)
    guidelines = classification_guideline_block(
        emotions, strategy, guidelines_path=guidelines_path
    )

    exemplars = ""
    if strategy == "few_shot_guidelines_dataset" and few_shot_examples:
        lines = []
        for e, exs in few_shot_examples.items():
            for ex in exs:
                lines.append(f"- ({e}) {ex}")
        exemplars = (
            "\n\nAdditional labelled examples from the training fold "
            "(use only as illustration; follow the guidelines above):\n"
            + "\n".join(lines)
        )

    context = ""
    if review_context and review_context.strip() and review_context.strip() != sentence.strip():
        context = (
            "\n\nFull review (context only; annotate the sentence, not the whole review):\n"
            f"{review_context.strip()}\n"
        )

    system = (
        "You are an expert annotator of emotions in mobile app reviews. "
        "Follow the annotation guidelines strictly. "
        "Assign zero or more labels from the allowed set. "
        "Annotate only what the user expresses explicitly in the sentence. "
        'Reply with JSON only: {"emotions": ["Label", ...]}.'
    )
    user = (
        f"Allowed labels: {', '.join(emotions)}\n\n"
        f"{guidelines}{exemplars}{context}\n\n"
        f"Sentence to annotate:\n{sentence}\n\n"
        "Return JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generation_messages(
    emotion: str,
    count: int,
    strategy: str,
    *,
    guidelines_path: str | None = None,
) -> list[dict[str, str]]:
    strategy = normalize_strategy(strategy)
    definition = generation_definition(emotion, strategy, guidelines_path=guidelines_path)
    user = (
        f"Generate {count} new mobile app reviews containing a sentence belonging to the emotion {emotion} "
        f"based on the following definition and examples from the annotation guidelines:\n\n{definition}\n\n"
        'Return JSON only: {"reviews": [{"review": "...", "sentence": "..."}, ...]} '
        f"with exactly {count} items."
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate realistic mobile app review text for emotion dataset augmentation. "
                "Follow the provided emotion definitions. Reply with JSON only."
            ),
        },
        {"role": "user", "content": user},
    ]
