"""Augmentation-utility ranking (MiniLM + Borda) for RQ4 generator selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from emotion_cls.config import resolve_path
from emotion_cls.data.dataset import load_ground_truth, load_synthetic_dir


def _centroid(x: np.ndarray) -> np.ndarray:
    c = x.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def _metrics(s_emb: np.ndarray, gt_emb: np.ndarray) -> dict[str, float]:
    n = s_emb.shape[0]
    if n < 2 or gt_emb.shape[0] == 0:
        return {k: float("nan") for k in ("diversity_intra", "novelty_vs_synth", "novelty_vs_gt", "on_emotion")}
    d_ss = cosine_distances(s_emb, s_emb)
    iu = np.triu_indices_from(d_ss, k=1)
    diversity_intra = float(d_ss[iu].mean())
    d_nn = d_ss.copy()
    np.fill_diagonal(d_nn, np.inf)
    novelty_vs_synth = float(d_nn.min(axis=1).mean())
    novelty_vs_gt = float(cosine_distances(s_emb, gt_emb).min(axis=1).mean())
    on_emotion = float((s_emb @ _centroid(gt_emb)).mean())
    return {
        "diversity_intra": diversity_intra,
        "novelty_vs_synth": novelty_vs_synth,
        "novelty_vs_gt": novelty_vs_gt,
        "on_emotion": on_emotion,
    }


def rank_augmentation_utility(cfg: dict[str, Any]) -> Path:
    emotions = cfg["data"]["generation_emotions"]
    gt = load_ground_truth(cfg["data"]["ground_truth"], emotions=cfg["data"]["emotions"])
    synth = load_synthetic_dir(cfg["data"]["synthetic_dir"], emotions)
    if len(synth) == 0:
        raise FileNotFoundError(
            f"No synthetic CSVs found under {cfg['data']['synthetic_dir']}. "
            "Expected Datasets/{Claude|Gemini|OpenAi}/{strategy}/*.csv"
        )

    from sentence_transformers import SentenceTransformer

    from emotion_cls.experiment import ExperimentRun, timed

    out_dir = resolve_path(cfg["project"]["output_dir"]) / "augmentation_utility"
    run = ExperimentRun(
        out_dir,
        name="rank_augmentation",
        cfg=cfg,
        resume=False,
        extra_meta={"n_synth": len(synth)},
    )

    embedder_name = cfg["augmentation"]["embedder"]
    t0 = timed()
    encoder = SentenceTransformer(embedder_name)
    gt_emb = {
        e: encoder.encode(
            gt.loc[gt[e] == 1, "sentence"].tolist(),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        for e in emotions
    }
    synth_emb = encoder.encode(
        synth["sentence"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    rows = []
    for (emotion, strategy, genai), idx in synth.groupby(["Emotion", "Strategy", "GenAI"]).groups.items():
        m = _metrics(synth_emb[list(idx)], gt_emb[emotion])
        rows.append({"Emotion": emotion, "Strategy": strategy, "GenAI": genai, "n": len(idx), **m})
    metrics_df = pd.DataFrame(rows)

    metric_cols = ["diversity_intra", "novelty_vs_synth", "novelty_vs_gt", "on_emotion"]
    ranked_parts = []
    winners = []
    for emotion, sub in metrics_df.groupby("Emotion"):
        df_e = sub.copy()
        for c in metric_cols:
            df_e[f"rk_{c}"] = df_e[c].rank(method="min", ascending=True)
        df_e["borda"] = df_e[[f"rk_{c}" for c in metric_cols]].sum(axis=1)
        df_e = df_e.sort_values("borda", ascending=False).reset_index(drop=True)
        df_e["overall_rank"] = np.arange(1, len(df_e) + 1)
        ranked_parts.append(df_e)
        winners.append(df_e.iloc[0][["Emotion", "Strategy", "GenAI", "borda"]].to_dict())

    ranked = pd.concat(ranked_parts, ignore_index=True)
    agg = (
        metrics_df.groupby(["Strategy", "GenAI"], as_index=False)
        .agg(emotions_covered=("Emotion", "nunique"), n_total=("n", "sum"), **{c: (c, "mean") for c in metric_cols})
    )
    for c in metric_cols:
        agg[f"rk_{c}"] = agg[c].rank(method="min", ascending=True)
    agg["borda"] = agg[[f"rk_{c}" for c in metric_cols]].sum(axis=1)
    agg = agg.sort_values("borda", ascending=False).reset_index(drop=True)
    agg["overall_rank"] = np.arange(1, len(agg) + 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "per_emotion_ranking.csv", index=False)
    agg.to_csv(out_dir / "aggregate_ranking.csv", index=False)
    pd.DataFrame(winners).to_csv(out_dir / "per_emotion_winners.csv", index=False)
    winner = agg.iloc[0]
    print(
        f"Recommended single pipeline: {winner['GenAI']} + {winner['Strategy']} "
        f"(borda={winner['borda']:.1f}, emotions_covered={int(winner['emotions_covered'])})"
    )
    run.finalize(
        status="completed",
        elapsed_s=timed() - t0,
        recommended=f"{winner['GenAI']}+{winner['Strategy']}",
    )
    return out_dir
