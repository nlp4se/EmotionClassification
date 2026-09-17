#!/usr/bin/env python3
"""Inter-annotator agreement and correctness report for the synthetic-review
human-as-judge validation (human_eval/synthetic_review_validation/).

Reads the three annotator files (QM, CC, MO) and the answer key, then reports,
pooled and per-emotion, and with/without CC:

- Cohen's kappa between every pair of annotators
- Average Cohen's kappa
- Fleiss' kappa across all annotators
- Per-annotator and average correctness (precision/recall/F1) against the
  generator's intended (ground-truth) emotion
- Correctness of the majority-vote aggregated label against ground truth

Each emotion is treated as an independent binary present/absent judgment per
sentence (the annotation itself is multi-label: up to a few emotions may be
marked per review). "Pooled" collapses all (sentence, emotion) pairs into one
long binary series; "per emotion" restricts to one emotion's column across
all 100 sentences.

Usage:
    python scripts/report_human_eval_agreement.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

DATA_DIR = Path(__file__).resolve().parents[1] / "human_eval" / "synthetic_review_validation"
ANSWER_KEY = DATA_DIR / "iteration_synthetic_ANSWER_KEY_do_not_share.csv"
ANNOTATORS = ["QM", "CC", "MO"]
EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]


def load_annotations(path: Path) -> dict[str, set[str]]:
    """row_id -> set of emotions marked by this annotator."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["reviews"]
    headers = [c.value for c in ws[1]]
    emo_cols = [i for i, h in enumerate(headers, 1) if h and str(h).startswith("emotion-")]
    rid_col = headers.index("row_id") + 1
    out: dict[str, set[str]] = {}
    for r in range(2, ws.max_row + 1):
        rid = ws.cell(row=r, column=rid_col).value
        if rid is None:
            continue
        marked = {ws.cell(row=r, column=c).value for c in emo_cols}
        marked.discard(None)
        out[rid] = marked
    return out


def to_binary_matrix(row_ids: list[str], annotations: dict[str, set[str]]) -> pd.DataFrame:
    """row_id x emotion binary presence matrix, columns ordered as EMOTIONS."""
    data = {e: [1 if e in annotations[rid] else 0 for rid in row_ids] for e in EMOTIONS}
    return pd.DataFrame(data, index=row_ids)


def fleiss_kappa(rating_matrices: list[np.ndarray]) -> float:
    """Fleiss' kappa for >=2 raters on binary (0/1) subject-level ratings.

    ``rating_matrices``: one 1-D 0/1 array per rater, all the same length
    (subjects). Converted internally to the standard n_ij subjects x
    categories count table Fleiss' kappa is defined over.
    """
    n_raters = len(rating_matrices)
    n_subjects = len(rating_matrices[0])
    counts = np.zeros((n_subjects, 2), dtype=float)  # [:, 0]=absent, [:, 1]=present
    for ratings in rating_matrices:
        for i, v in enumerate(ratings):
            counts[i, int(v)] += 1

    p_j = counts.sum(axis=0) / (n_subjects * n_raters)
    P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_bar_e = float(np.sum(p_j**2))
    if P_bar_e == 1.0:
        return 1.0
    return (P_bar - P_bar_e) / (1 - P_bar_e)


def kappa_label(k: float) -> str:
    if k < 0:
        return "poor"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def pairwise_kappa_table(mats: dict[str, pd.DataFrame], emotion: str | None = None) -> pd.DataFrame:
    names = list(mats.keys())
    rows = []
    for a, b in itertools.combinations(names, 2):
        xa = mats[a][emotion].values if emotion else mats[a].values.ravel()
        xb = mats[b][emotion].values if emotion else mats[b].values.ravel()
        k = cohen_kappa_score(xa, xb)
        rows.append({"pair": f"{a}-{b}", "kappa": k, "interpretation": kappa_label(k)})
    return pd.DataFrame(rows)


def majority_vote_matrix(mats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (sentence, emotion), 1 if a strict majority of annotators marked it
    present, else 0. For an odd number of raters this is always decisive; for
    an even number (e.g. the 2-annotator excluding-CC case) a majority
    requires unanimous agreement, since 1 of 2 is a tie, not a majority.
    """
    stacked = np.stack([m.values for m in mats.values()], axis=0)  # raters x sentences x emotions
    threshold = len(mats) // 2 + 1
    votes = (stacked.sum(axis=0) >= threshold).astype(int)
    return pd.DataFrame(votes, index=next(iter(mats.values())).index, columns=EMOTIONS)


def correctness_table(mats: dict[str, pd.DataFrame], gt: pd.DataFrame, emotion: str | None = None) -> pd.DataFrame:
    """Precision/recall/F1 of each annotator's binary emotion judgments against
    the generator's intended emotion. Since each review has exactly one
    intended emotion, recall here already answers "did the annotator catch
    the true label" -- there is no separate hit-rate to report.
    """
    rows = []
    for name, mat in mats.items():
        y_true = (gt[emotion].values if emotion else gt.values.ravel())
        y_pred = (mat[emotion].values if emotion else mat.values.ravel())
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        rows.append({"annotator": name, "precision": p, "recall": r, "f1": f1})
    df = pd.DataFrame(rows)
    if len(df) > 1:
        avg = df[["precision", "recall", "f1"]].mean()
        df.loc[len(df)] = {"annotator": "AVERAGE", **avg.to_dict()}
    return df


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def build_report(mats: dict[str, pd.DataFrame], gt: pd.DataFrame, title_suffix: str) -> list[str]:
    lines = [f"## Agreement and correctness{title_suffix}", ""]

    lines += ["### Inter-annotator agreement (pooled across all 9 emotions x 100 sentences)", ""]
    lines += [fmt(pairwise_kappa_table(mats)), ""]
    pk = pairwise_kappa_table(mats)
    lines += [f"Average Cohen's kappa: **{pk['kappa'].mean():.3f}** ({kappa_label(pk['kappa'].mean())})", ""]
    fk = fleiss_kappa([mats[a].values.ravel() for a in mats])
    lines += [f"Fleiss' kappa (all {len(mats)} annotators): **{fk:.3f}** ({kappa_label(fk)})", ""]

    lines += ["### Inter-annotator agreement, by emotion", ""]
    rows = []
    for e in EMOTIONS:
        pk_e = pairwise_kappa_table(mats, emotion=e)
        fk_e = fleiss_kappa([mats[a][e].values for a in mats])
        row = {"emotion": e, "avg_cohen_kappa": pk_e["kappa"].mean(), "fleiss_kappa": fk_e}
        for _, r in pk_e.iterrows():
            row[f"kappa[{r['pair']}]"] = r["kappa"]
        rows.append(row)
    lines += [fmt(pd.DataFrame(rows)), ""]

    lines += ["### Correctness vs. ground truth (pooled across all 9 emotions x 100 sentences)", ""]
    lines += [fmt(correctness_table(mats, gt)), ""]

    lines += ["### Correctness vs. ground truth, by emotion", ""]
    for e in EMOTIONS:
        lines += [f"**{e}**", "", fmt(correctness_table(mats, gt, emotion=e)), ""]

    majority = majority_vote_matrix(mats)
    threshold = len(mats) // 2 + 1
    lines += [
        f"### Correctness of the majority-vote aggregated human label ({threshold} of {len(mats)})",
        "",
        "Per (sentence, emotion), the aggregated human label is 1 if at least",
        f"{threshold} of the {len(mats)} annotators marked that emotion present, else 0",
        "(for 2 annotators this means unanimous agreement, since 1 of 2 is a tie,",
        "not a majority). Scored against the generator's intended emotion, matching",
        "how a 2-of-3 majority-vote ground truth is normally derived from multiple",
        "annotators.",
        "",
        "Pooled (all 9 emotions x 100 sentences):",
        "",
        fmt(correctness_table({"majority_vote": majority}, gt)),
        "",
        "By emotion:",
        "",
        fmt(
            pd.DataFrame(
                [
                    {"emotion": e, **correctness_table({"majority_vote": majority}, gt, emotion=e).iloc[0].drop("annotator").to_dict()}
                    for e in EMOTIONS
                ]
            )
        ),
        "",
    ]

    return lines


def main() -> None:
    answer_key = pd.read_csv(ANSWER_KEY)
    answer_key = answer_key.set_index("row_id")
    row_ids = list(answer_key.index)
    gt = answer_key[EMOTIONS]

    annotations = {a: load_annotations(DATA_DIR / f"iteration_synthetic_{a}.xlsx") for a in ANNOTATORS}
    mats = {a: to_binary_matrix(row_ids, annotations[a]) for a in ANNOTATORS}

    report = [
        "# Human-as-judge agreement and correctness report",
        "",
        f"- Annotators: {', '.join(ANNOTATORS)}",
        f"- Sentences: {len(row_ids)}",
        f"- Emotions: {', '.join(EMOTIONS)}",
        "- Agreement unit: each (sentence, emotion) pair is an independent binary present/absent judgment.",
        "- Correctness: precision/recall/F1 of each annotator's binary judgments against the",
        "  generator's intended emotion (a single positive label per sentence). Recall here is",
        "  exactly the fraction of sentences where the intended emotion is among those marked.",
        "",
        "---",
        "",
    ]
    report += build_report(mats, gt, title_suffix=" (all annotators)")
    report += ["---", ""]
    mats_no_cc = {a: mats[a] for a in ANNOTATORS if a != "CC"}
    report += build_report(mats_no_cc, gt, title_suffix=" (excluding CC)")

    out_path = DATA_DIR / "agreement_report.md"
    out_path.write_text("\n".join(report) + "\n")
    print(f"Report written to {out_path}")
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
