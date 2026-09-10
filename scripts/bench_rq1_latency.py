"""One-off inference-latency benchmark for RQ1 encoders, matching RQ2's
per-sentence cost profile (Table tab:rq2-cost). Loads each backbone from its
pretrained HF checkpoint (fine-tuned weights don't affect forward-pass
latency; architecture + hardware do) and times pure inference over the real
1090-sentence corpus, batch_size=16, max_length=512 -- identical to
configs/default.yaml training config. Same RTX 4090 used for the original
runs. Binary-ensemble total = single binary-classifier latency x 9 (one
forward pass per emotion, num_labels=2), matching the actual ensemble
architecture (9 independent classifiers).
"""
import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_cls.data.dataset import load_ground_truth

EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]
ENCODERS = {
    "bert-base-cased": "bert-base-cased",
    "bert-large-cased": "bert-large-cased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "distilbert-base-cased": "distilbert-base-cased",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "xlnet-base-cased": "xlnet-base-cased",
    "xlnet-large-cased": "xlnet-large-cased",
}
BATCH_SIZE = 16
MAX_LENGTH = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

df = load_ground_truth("Datasets/GroundTruth.csv")
sentences = df["sentence"].tolist()
n = len(sentences)
print(f"N sentences = {n}, device = {device}")


def bench(hub_id, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hub_id, num_labels=num_labels, problem_type="multi_label_classification"
    ).to(device).eval()

    batches = [sentences[i:i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    # warm-up (excluded from timing): CUDA kernel compilation / cudnn autotune
    with torch.no_grad():
        enc = tokenizer(batches[0], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        model(**enc)
        if device == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for b in batches:
            enc = tokenizer(b, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            model(**enc)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return elapsed


results = {}
for name, hub_id in ENCODERS.items():
    ml_elapsed = bench(hub_id, num_labels=len(EMOTIONS))
    bin_single_elapsed = bench(hub_id, num_labels=2)
    results[name] = {
        "multilabel_total_s": ml_elapsed,
        "multilabel_ms_per_sentence": 1000 * ml_elapsed / n,
        "binary_single_total_s": bin_single_elapsed,
        "binary_single_ms_per_sentence": 1000 * bin_single_elapsed / n,
        "binary_ensemble_ms_per_sentence": 1000 * bin_single_elapsed / n * len(EMOTIONS),
    }
    print(name, results[name])

with open("/tmp/rq1_latency.json", "w") as f:
    json.dump(results, f, indent=2)
print("wrote /tmp/rq1_latency.json")
