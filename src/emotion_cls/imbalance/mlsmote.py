"""MLSMOTE-style oversampling in embedding space (Charte et al., 2015).

Produces synthetic (embedding, label) pairs for optional embedding-space
training. For end-to-end encoder fine-tuning on text, use generative
augmentation (`genai_aug`) instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class MLSMOTEResult:
    embeddings: np.ndarray
    labels: np.ndarray


def mlsmote_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    k_neighbors: int = 5,
    n_samples: int | None = None,
    seed: int = 42,
) -> MLSMOTEResult:
    """Generate synthetic (embedding, label) pairs for minority labels."""
    rng = np.random.default_rng(seed)
    pos_counts = labels.sum(axis=0)
    mean_count = pos_counts.mean()
    minority = np.where(pos_counts < mean_count)[0]
    if len(minority) == 0:
        return MLSMOTEResult(embeddings=np.empty((0, embeddings.shape[1])), labels=np.empty((0, labels.shape[1])))

    synth_x: list[np.ndarray] = []
    synth_y: list[np.ndarray] = []
    target = n_samples or int(mean_count)

    for c in minority:
        idx = np.where(labels[:, c] == 1)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(idx))).fit(embeddings[idx])
        n_needed = max(0, target - int(pos_counts[c]))
        for _ in range(n_needed):
            i = rng.choice(idx)
            dists, neigh = nn.kneighbors(embeddings[i].reshape(1, -1), return_distance=True)
            # skip self (first neighbour)
            candidates = neigh[0][1:] if neigh.shape[1] > 1 else neigh[0]
            if len(candidates) == 0:
                continue
            j_local = rng.choice(candidates)
            j = idx[j_local]
            gap = rng.random()
            x_new = embeddings[i] + gap * (embeddings[j] - embeddings[i])
            # Intersection of labels (conservative multi-label SMOTE variant)
            y_new = (labels[i] * labels[j]).astype(np.float32)
            y_new[c] = 1.0
            synth_x.append(x_new)
            synth_y.append(y_new)

    if not synth_x:
        return MLSMOTEResult(embeddings=np.empty((0, embeddings.shape[1])), labels=np.empty((0, labels.shape[1])))
    return MLSMOTEResult(embeddings=np.vstack(synth_x), labels=np.vstack(synth_y))


def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
