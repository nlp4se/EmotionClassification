"""Prompt builders for decoder-only classification and generation."""

from __future__ import annotations

import json

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


def classification_messages_batch(
    items: list[dict[str, str | int]],
    emotions: list[str],
    strategy: str,
    *,
    few_shot_examples: dict[str, list[str]] | None = None,
    guidelines_path: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages classifying a BATCH of sentences in one call.

    ``items`` is ``[{"id": <int>, "sentence": <str>, "review_context": <str|None>}, ...]``.
    The whole point of batching is that the (large) guidelines block is sent
    once for the batch instead of once per sentence. Each item carries a
    caller-assigned integer id that the model must echo back with its result
    -- that's what lets the response be validated and matched to inputs
    regardless of ordering, instead of trusting positional alignment.
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

    item_objs = []
    for item in items:
        obj: dict[str, str | int] = {"id": item["id"], "sentence": item["sentence"]}
        ctx = item.get("review_context")
        sentence = str(item["sentence"])
        if ctx and str(ctx).strip() and str(ctx).strip() != sentence.strip():
            obj["review"] = str(ctx).strip()
        item_objs.append(obj)
    items_block = json.dumps(item_objs, ensure_ascii=False, indent=2)

    n = len(items)
    system = (
        "You are an expert annotator of emotions in mobile app reviews. "
        "Follow the annotation guidelines strictly. "
        "You will be given a batch of independent sentences to classify. "
        "Assign zero or more labels from the allowed set to EACH sentence "
        "independently -- never let one sentence's content, context, or "
        "labels influence another's. Annotate only what the user expresses "
        "explicitly in each sentence; a \"review\" field, when present, is "
        "context only, never the annotation target. "
        f'Reply with JSON only: a single array of exactly {n} objects, one '
        'per input item, each shaped {"id": <the input id>, "emotions": '
        '["Label", ...]}. Every input id must appear exactly once in the '
        "output, in any order."
    )
    user = (
        f"Allowed labels: {', '.join(emotions)}\n\n"
        f"{guidelines}{exemplars}\n\n"
        f"Classify each of the following {n} items:\n{items_block}\n\n"
        f"Return the JSON array now (exactly {n} objects, each with its matching id)."
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
