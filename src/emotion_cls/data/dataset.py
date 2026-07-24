"""Dataset loading, label matrices, and CV splits."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from emotion_cls.config import resolve_path

DEFAULT_EMOTIONS = [
    "Joy",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger",
    "Anticipation",
    "Neutral",
]


def load_ground_truth(
    path: str | Path,
    *,
    emotions: list[str] | None = None,
    text_column: str = "sentence",
    drop_reject: bool = True,
) -> pd.DataFrame:
    """Load the human ground-truth CSV (semicolon-separated).

    By default drops rows with Reject=1, matching the experimental protocol
    (Reject is unused as a prediction target).
    """
    emotions = emotions or DEFAULT_EMOTIONS
    df = pd.read_csv(resolve_path(path), sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if text_column not in df.columns:
        raise KeyError(f"Missing text column '{text_column}' in {path}")
    if drop_reject and "Reject" in df.columns:
        reject = pd.to_numeric(df["Reject"], errors="coerce").fillna(0).astype(int)
        df = df.loc[reject == 0].reset_index(drop=True)
    for e in emotions:
        if e not in df.columns:
            raise KeyError(f"Missing emotion column '{e}' in {path}")
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0).astype(int)
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() > 0].reset_index(drop=True)
    df["source"] = "human"
    return df


def label_matrix(df: pd.DataFrame, emotions: list[str]) -> np.ndarray:
    return df[emotions].to_numpy(dtype=np.float32)


def majority_count(df: pd.DataFrame, emotions: list[str]) -> int:
    """Max positive count among generation target emotions (excludes Neutral if absent)."""
    return int(df[emotions].sum().max())


def multilabel_folds(
    y: np.ndarray,
    n_splits: int = 10,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # X is unused by the splitter besides length
    X = np.zeros((y.shape[0], 1))
    yield from mskf.split(X, y)


def binary_folds(
    y_binary: np.ndarray,
    n_splits: int = 10,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros((len(y_binary), 1))
    yield from skf.split(X, y_binary)


def load_synthetic_dir(
    root: str | Path,
    emotions: list[str],
) -> pd.DataFrame:
    """Load synthetic review CSVs under Datasets/{Provider}/{Strategy}/."""
    root = resolve_path(root)
    frames: list[pd.DataFrame] = []
    provider_map = {"Claude": "Claude", "Gemini": "Gemini", "OpenAi": "GPT", "OpenAI": "GPT"}
    strategy_map = {
        "0Shoot": "zero_shot",
        "FewShoot": "few_shot_guidelines",
        "FewShootExample": "few_shot_guidelines_dataset",
        "FewShootExamples": "few_shot_guidelines_dataset",
    }
    for provider_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        genai = provider_map.get(provider_dir.name)
        if not genai:
            continue
        for strategy_dir in sorted(p for p in provider_dir.iterdir() if p.is_dir()):
            strategy = strategy_map.get(strategy_dir.name)
            if not strategy:
                continue
            for csv_path in sorted(strategy_dir.glob("*.csv")):
                emotion = _emotion_from_name(csv_path.name, emotions)
                if emotion is None:
                    continue
                df = pd.read_csv(csv_path, sep=None, engine="python")
                df.columns = [c.strip() for c in df.columns]
                text_col = "sentence" if "sentence" in df.columns else "review"
                out = pd.DataFrame(
                    {
                        "sentence": df[text_col].astype(str).str.strip().str.strip('"'),
                        "Emotion": emotion,
                        "Strategy": strategy,
                        "GenAI": genai,
                        "source": "synthetic",
                        "source_file": csv_path.name,
                    }
                )
                frames.append(out[out["sentence"].str.len() > 0])
    if not frames:
        return pd.DataFrame(columns=["sentence", "Emotion", "Strategy", "GenAI", "source"])
    return pd.concat(frames, ignore_index=True)


def synthetic_as_multilabel(
    synth: pd.DataFrame,
    emotions: list[str],
    *,
    strategy: str | None = None,
    genai: str | None = None,
    n_per_emotion: int | None = None,
) -> pd.DataFrame:
    """Convert synthetic single-emotion rows into multilabel GT schema."""
    df = synth.copy()
    if strategy:
        df = df[df["Strategy"] == strategy]
    if genai:
        df = df[df["GenAI"] == genai]
    rows = []
    for emotion in emotions:
        sub = df[df["Emotion"] == emotion]
        if n_per_emotion is not None:
            sub = sub.head(n_per_emotion)
        for _, r in sub.iterrows():
            labels = {e: 0 for e in emotions}
            labels[emotion] = 1
            rows.append({"sentence": r["sentence"], "source": "synthetic", **labels})
    return pd.DataFrame(rows)


def _emotion_from_name(name: str, emotions: list[str]) -> str | None:
    lower = name.lower()
    for e in emotions:
        if f"_{e.lower()}_" in lower:
            return e
    return None
