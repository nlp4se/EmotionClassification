"""Generate the three Results-section figures (RQ1, RQ2, RQ3) from the
verified experiment outputs already cached under /tmp/rq{1,2,3}_full.json
and /tmp/rq2_cost.json. Run once from the repo root with the project venv
active; writes PDFs to figures/.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
})

# Colorblind-safe categorical pair (Wong 2011 palette), print-legible.
C_ML = "#0072B2"   # blue -- multi-label / proprietary / undersampling
C_BIN = "#D55E00"  # vermillion -- binary ensemble / open-source / genai_aug
GRAY = "#888888"

# ---------------------------------------------------------------- RQ1 ----
d1 = json.load(open("/tmp/rq1_full.json"))
encoders = ["bert-base-cased", "bert-large-cased", "roberta-base", "roberta-large",
            "distilbert-base-cased", "deberta-v3-base", "deberta-v3-large",
            "xlnet-base-cased", "xlnet-large-cased"]
short = {"bert-base-cased": "BERT-base", "bert-large-cased": "BERT-large",
         "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large",
         "distilbert-base-cased": "DistilBERT", "deberta-v3-base": "DeBERTa-v3-base",
         "deberta-v3-large": "DeBERTa-v3-large", "xlnet-base-cased": "XLNet-base",
         "xlnet-large-cased": "XLNet-large"}
order = encoders  # catalogue order (Table~\ref{tab:models}), matches tab:rq1-agg

fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = range(len(order))
w = 0.36
ml_vals = [d1["multilabel"][e]["macro_f1"][0] for e in order]
ml_err = [d1["multilabel"][e]["macro_f1"][1] for e in order]
bin_vals = [d1["binary"][e]["macro_f1"][0] for e in order]
bin_err = [d1["binary"][e]["macro_f1"][1] for e in order]
ax.bar([i - w / 2 for i in x], ml_vals, width=w, yerr=ml_err, capsize=2,
       color=C_ML, label="Multi-label", edgecolor="white", linewidth=0.4)
ax.bar([i + w / 2 for i in x], bin_vals, width=w, yerr=bin_err, capsize=2,
       color=C_BIN, label="Binary ensemble", edgecolor="white", linewidth=0.4)
ax.set_xticks(list(x))
ax.set_xticklabels([short[e] for e in order], rotation=30, ha="right")
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 0.55)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("figures/rq1-encoder-comparison.pdf")
plt.close(fig)

# ---------------------------------------------------------------- RQ2 ----
d2 = json.load(open("/tmp/rq2_full.json"))
API = ["gpt-5.3-chat", "gemini-3-flash", "claude-haiku-4-5", "mistral-large-2512"]
OSS = ["deepseek-r1-8b", "mistral", "gpt-oss-20b", "gemma3-4b", "llama3.1-8b", "qwen3-8b"]
STRATS = ["zero_shot", "few_shot_guidelines", "few_shot_guidelines_dataset"]
dec_short = {"gpt-5.3-chat": "GPT-5.2", "gemini-3-flash": "Gemini-3.5-Flash",
             "claude-haiku-4-5": "Claude-Haiku-4.5", "mistral-large-2512": "Mistral-Large",
             "deepseek-r1-8b": "DeepSeek-R1-8B", "mistral": "Mistral-7B",
             "gpt-oss-20b": "GPT-OSS-20B", "gemma3-4b": "Gemma3-4B",
             "llama3.1-8b": "Llama3.1-8B", "qwen3-8b": "Qwen3-8B"}


def best_macro(dec):
    return max(d2[dec][s]["t0"]["macro_f1"] for s in STRATS)


all_dec = sorted(API, key=lambda d: -best_macro(d)) + sorted(OSS, key=lambda d: -best_macro(d))
fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = range(len(all_dec))
vals = [best_macro(d) for d in all_dec]
colors = [C_ML if d in API else C_BIN for d in all_dec]
ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, width=0.6)
ax.axvline(len(API) - 0.5, color="#bbbbbb", linewidth=0.8, linestyle=":")
ax.set_xticks(list(x))
ax.set_xticklabels([dec_short[d] for d in all_dec], rotation=30, ha="right")
ax.set_ylabel("Macro-F1 (best strategy, $T{=}0$)")
ax.set_ylim(0, 0.72)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
from matplotlib.patches import Patch
handles = [Patch(facecolor=C_ML, label="Proprietary (API)"),
           Patch(facecolor=C_BIN, label="Open-source (local)")]
ax.legend(handles=handles, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("figures/rq2-decoder-comparison.pdf")
plt.close(fig)

# ---------------------------------------------------------------- RQ3 ----
d3 = json.load(open("/tmp/rq3_full.json"))
cutoffs = [50, 100, 150, 200, 250, 300, 350]
under_vals = [d3[f"undersample_c{c}"]["macro_f1"][0] for c in cutoffs]
genai_vals = [d3[f"genai_aug_n{c}"]["macro_f1"][0] for c in cutoffs]
baseline = d3["none"]["macro_f1"][0]

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.axhline(baseline, color=GRAY, linewidth=1.0, linestyle="--", label="Baseline (no mitigation)")
ax.plot(cutoffs, under_vals, marker="s", color=C_ML, linewidth=1.4, markersize=4,
        label="Undersampling (cut-off)")
ax.plot(cutoffs, genai_vals, marker="o", color=C_BIN, linewidth=1.4, markersize=4,
        label="Generative augmentation ($n$ injected)")
ax.set_xlabel("Positives per emotion (cut-off / injected $n$)")
ax.set_ylabel("Macro-F1")
ax.set_ylim(0.2, 0.65)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
fig.tight_layout()
fig.savefig("figures/rq3-imbalance-trajectory.pdf")
plt.close(fig)

# ------------------------------------------------ RQ3 per-emotion ----
emotions = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger",
            "Anticipation", "Neutral"]
# Fixed per-emotion hue assignment (muted, matched saturation/lightness so
# no single emotion visually dominates): Joy=yellow, Sadness=blue,
# Surprise=orange, Fear=purple, Anger=red, Anticipation=pink, Trust=cyan,
# Disgust=green, Neutral=grey. Reused everywhere an emotion needs its own
# color (currently only this trajectory figure colors by emotion).
EMO_COLOR = {
    "Joy": "#C5A239", "Sadness": "#3973C5", "Surprise": "#C57A39", "Fear": "#7A39C5",
    "Anger": "#C54039", "Anticipation": "#C5397F", "Trust": "#39AEC5", "Disgust": "#39C550",
    "Neutral": "#8C8C8C",
}
EMO_MARKER = {
    "Joy": "o", "Trust": "s", "Fear": "^", "Surprise": "v", "Sadness": "D",
    "Disgust": "P", "Anger": "X", "Anticipation": "*", "Neutral": "h",
}
xs = [0] + cutoffs
fig, ax = plt.subplots(figsize=(7.0, 3.4))
for e in emotions:
    ys = [d3["none"]["per_emotion"][e]] + [d3[f"genai_aug_n{c}"]["per_emotion"][e] for c in cutoffs]
    ax.plot(xs, ys, marker=EMO_MARKER[e], color=EMO_COLOR[e], linewidth=1.2,
            markersize=4.5, label=e)
ax.set_xlabel("Injected synthetic reviews per emotion ($n$); $n{=}0$ is the no-augmentation baseline")
ax.set_ylabel("F1")
ax.set_xticks(xs)
ax.set_ylim(0, 0.9)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5,
          handlelength=1.6, labelspacing=0.4)
fig.tight_layout()
fig.savefig("figures/rq3-per-emotion-trajectory.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------ per-emotion grouped bars ----
# Emotions sorted by support (positives in the N=1090 working corpus),
# descending -- matches tab:rq{1,2,3}-per-emotion row order.
EMO_ORDER = ["Joy", "Sadness", "Anticipation", "Trust", "Neutral",
             "Surprise", "Disgust", "Anger", "Fear"]


def grouped_bar_per_emotion(vals_a, vals_b, label_a, label_b, outfile):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = range(len(EMO_ORDER))
    w = 0.36
    ax.bar([i - w / 2 for i in x], vals_a, width=w, color=C_ML, label=label_a,
           edgecolor="white", linewidth=0.4)
    ax.bar([i + w / 2 for i in x], vals_b, width=w, color=C_BIN, label=label_b,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(EMO_ORDER)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


# RQ1: multi-label (RoBERTa-large) vs. binary ensemble (BERT-base)
ml_pe = [d1["multilabel"]["roberta-large"]["per_emotion"][e] for e in EMO_ORDER]
bin_pe = [d1["binary"]["bert-base-cased"]["per_emotion"][e] for e in EMO_ORDER]
grouped_bar_per_emotion(ml_pe, bin_pe, "Multi-label (RoBERTa$_{large}$)",
                         "Binary ensemble (BERT$_{base}$)",
                         "figures/rq1-per-emotion.pdf")

# RQ2: best proprietary (Gemini-3.5-Flash) vs. best open-source (GPT-OSS-20B),
# each at its own best strategy, T=0 (few_shot_guidelines_dataset for both).
gem_pe = [d2["gemini-3-flash"]["few_shot_guidelines_dataset"]["t0"]["per_emotion"][e] for e in EMO_ORDER]
oss_pe = [d2["gpt-oss-20b"]["few_shot_guidelines_dataset"]["t0"]["per_emotion"][e] for e in EMO_ORDER]
grouped_bar_per_emotion(gem_pe, oss_pe, "Gemini-3.5-Flash", "GPT-OSS-20B",
                         "figures/rq2-per-emotion.pdf")

# RQ3: baseline vs. best configuration (genai_aug n=100 + bce_weight)
base_pe = [d3["none"]["per_emotion"][e] for e in EMO_ORDER]
best_pe = [d3["genai_aug_n100+bce_pos_weight"]["per_emotion"][e] for e in EMO_ORDER]
grouped_bar_per_emotion(base_pe, best_pe, "Baseline (no mitigation)",
                         "Best config. (GenAI aug. $n{=}100$ + BCE pos. weight)",
                         "figures/rq3-per-emotion.pdf")

print("wrote figures/rq1-encoder-comparison.pdf")
print("wrote figures/rq2-decoder-comparison.pdf")
print("wrote figures/rq3-imbalance-trajectory.pdf")
print("wrote figures/rq3-per-emotion-trajectory.pdf")
print("wrote figures/rq1-per-emotion.pdf")
print("wrote figures/rq2-per-emotion.pdf")
print("wrote figures/rq3-per-emotion.pdf")
