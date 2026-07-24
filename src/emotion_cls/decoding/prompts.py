"""Prompt builders for decoder-only classification and generation."""

from __future__ import annotations

from emotion_cls.resources.definitions import DEFINITIONS, STRATEGY_ALIASES


def normalize_strategy(strategy: str) -> str:
    return STRATEGY_ALIASES.get(strategy, strategy)


def emotion_definition(emotion: str, strategy: str) -> str:
    strategy = normalize_strategy(strategy)
    return DEFINITIONS[strategy].get(emotion, f"No definition for {emotion}.")


def classification_messages(
    sentence: str,
    emotions: list[str],
    strategy: str,
    *,
    few_shot_examples: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    strategy = normalize_strategy(strategy)
    blocks = []
    for e in emotions:
        if e == "Neutral":
            blocks.append(
                f"### Neutral\nNeutral expresses absence of clear emotion toward the app."
            )
        else:
            blocks.append(f"### {e}\n{emotion_definition(e, strategy)}")
    guidelines = "\n\n".join(blocks)

    exemplars = ""
    if strategy == "few_shot_guidelines_dataset" and few_shot_examples:
        lines = []
        for e, exs in few_shot_examples.items():
            for ex in exs:
                lines.append(f"- ({e}) {ex}")
        exemplars = "\n\nAdditional labelled examples from the training fold:\n" + "\n".join(lines)

    system = (
        "You are an expert annotator of emotions in mobile app reviews. "
        "Assign zero or more emotion labels from the allowed set. "
        "Follow the guidelines strictly. "
        'Reply with JSON only: {"emotions": ["Label", ...]}.'
    )
    user = (
        f"Allowed labels: {', '.join(emotions)}\n\n"
        f"Guidelines:\n{guidelines}{exemplars}\n\n"
        f"Review sentence:\n{sentence}\n\n"
        "Return JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generation_messages(emotion: str, count: int, strategy: str) -> list[dict[str, str]]:
    strategy = normalize_strategy(strategy)
    definition = emotion_definition(emotion, strategy)
    user = (
        f"Generate {count} new mobile app reviews containing a sentence belonging to the emotion {emotion} "
        f"based on the following definition and examples:\n\n{definition}\n\n"
        'Return JSON only: {"reviews": [{"review": "...", "sentence": "..."}, ...]} '
        f"with exactly {count} items."
    )
    return [
        {
            "role": "system",
            "content": "You generate realistic mobile app review text for emotion dataset augmentation. Reply with JSON only.",
        },
        {"role": "user", "content": user},
    ]
