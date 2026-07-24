# EmotionClassification

Replication package for **fine-grained multi-label emotion classification of mobile app reviews** with large language models (IST paper draft under `paper/IST_Emotions/`).

Ground truth and annotation guidelines come from [Motger et al., *What About Emotions?*](https://arxiv.org/abs/2505.23452).

## Why Hugging Face + PyTorch for encoders?

Yes — that is the recommended stack:

- `transformers.Trainer` + `AutoModelForSequenceClassification` with `problem_type="multi_label_classification"`
- custom losses (`BCE pos_weight`, class weights, Focal, Adaptive Focal) via a thin `Trainer.compute_loss` override
- checkpoints publishable to Hugging Face Hub

Decoder-only models use **Ollama** locally (open-source) and official HTTP APIs for proprietary models.

## Repository layout

```
configs/                  # default.yaml, models.yaml, imbalance.yaml
src/emotion_cls/          # installable package
  data/                   # GT + synthetic loaders, CV splits
  losses/                 # imbalance-aware losses
  imbalance/              # undersample, MLSMOTE helpers, injection
  training/               # RQ1/RQ3 encoder fine-tuning + metrics
  decoding/               # RQ2 clients (Ollama/APIs) + classify
  augmentation/           # RQ4 generate + MiniLM/Borda ranking
  cli.py                  # emotion-cls entrypoint
Datasets/                 # GroundTruth.csv + existing synthetic CSVs
legacy/                   # previous notebooks/scripts (reference only)
paper/IST_Emotions/       # LaTeX manuscript
outputs/                  # created at runtime (metrics, rankings)
models/                   # created at runtime (checkpoints; gitignored)
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip
pip install -e .

cp .env.example .env
# edit .env with API keys as needed
```

### Ollama (local open-source generators / classifiers)

1. Install from https://ollama.com  
2. Pull models used in `configs/models.yaml`, e.g.:

```bash
ollama pull gemma3:27b
ollama pull qwen3:30b-a3b
ollama pull mistral-small3.1:24b
ollama pull deepseek-r1:8b
ollama pull llama4:scout
ollama pull gpt-oss:20b
```

3. Ensure `OLLAMA_HOST=http://localhost:11434` in `.env` (default).

### API keys (proprietary)

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | OpenAI chat models |
| `ANTHROPIC_API_KEY` (or legacy `CLAUDE_API_KEY`) | Anthropic Claude |
| `GEMINI_API_KEY` | Google Gemini |
| `MISTRAL_API_KEY` | Mistral API |

Never commit `.env`.

## Quick checks

```bash
emotion-cls list-models
emotion-cls export-run-config
emotion-cls train-encoder --dry-run
emotion-cls classify-decoder --decoder gemma3-27b --dry-run
```

---

## Experiments (mapped to RQs)

### RQ1 — Encoder-only classification

**Multi-label (single head, top-3 inference, 10-fold multilabel stratified CV):**

```bash
emotion-cls train-encoder --encoder bert-base-cased --head multilabel
emotion-cls train-encoder --encoder roberta-base --head multilabel
emotion-cls train-encoder --encoder deberta-v3-large --head multilabel
```

**Binary ensemble** (one binary classifier per emotion, assembled with top-3):

```bash
emotion-cls train-encoder --encoder bert-base-cased --head binary
```

Encoder keys are listed in `configs/models.yaml` (`bert-base-cased` is the baseline). Outputs land in `outputs/encoder_multilabel/...` or `outputs/encoder_binary/...`.

### RQ2 — Decoder-only classification (zero / few-shot)

```bash
# Local (Ollama)
emotion-cls classify-decoder --decoder gemma3-27b --strategy zero_shot
emotion-cls classify-decoder --decoder llama4-scout --strategy few_shot_guidelines

# Proprietary API
emotion-cls classify-decoder --decoder gpt-5.3-chat --strategy few_shot_guidelines_dataset
emotion-cls classify-decoder --decoder claude-opus-4-6 --strategy few_shot_guidelines
```

Strategies:

- `zero_shot` — guideline definitions only  
- `few_shot_guidelines` — definitions + guideline examples  
- `few_shot_guidelines_dataset` — + up to 5 GT exemplars **from the training fold only**

### RQ3 — Data imbalance mitigation

Pass `--imbalance` (comma-separated combinations allowed):

| Value | Meaning |
|---|---|
| `none` | Baseline multilabel BCE |
| `bce_pos_weight` | `pos_weight[c] = neg/pos` |
| `bce_weight` | `w[c] = N / (|E|·pos)` |
| `focal` | Focal loss + `alpha[c] = neg/(pos+neg)` |
| `adaptive_focal` | Adaptive Focal Loss |
| `undersample` | Cap positives at `--undersample-cutoff` (fold-internal) |
| `genai_aug` | Inject synthetic rows (`--aug-inject-n`, `--synthetic-ml-path`) |

Examples:

```bash
emotion-cls train-encoder --encoder bert-base-cased --imbalance bce_pos_weight
emotion-cls train-encoder --encoder bert-base-cased --imbalance focal
emotion-cls train-encoder --encoder bert-base-cased --imbalance adaptive_focal
emotion-cls train-encoder --encoder bert-base-cased --imbalance undersample --undersample-cutoff 150
emotion-cls sweep-undersample --encoder bert-base-cased --cutoffs 50,100,150,200,250,300,350

# GenAI injection (bins of 50) after generating a multilabel synthetic CSV
emotion-cls train-encoder --encoder bert-base-cased \
  --imbalance genai_aug,bce_pos_weight \
  --aug-inject-n 100 \
  --synthetic-ml-path path/to/synthetic_multilabel.csv
```

**MLSMOTE** is implemented in embedding space (`emotion_cls.imbalance.mlsmote`) for optional head-only experiments; for full encoder fine-tuning prefer **GenAI text augmentation** (paper §4).

### RQ4 — Synthetic generation + utility ranking

**Rank existing pilot corpora** under `Datasets/{Claude,Gemini,OpenAi}/...` (MiniLM + Borda):

```bash
emotion-cls rank-augmentation
```

**Generate until majority-class parity** (recommended generator from the pilot: Claude + few-shot guidelines+dataset):

```bash
# single emotion
emotion-cls generate --decoder claude --emotion Fear --strategy few_shot_guidelines_dataset

# all Plutchik emotions up to majority count
emotion-cls generate-parity --decoder claude --strategy few_shot_guidelines_dataset
```

`claude` / `gemini` / `openai` keys under `augmentation_pilot` in `configs/models.yaml` are the cheaper pilot generators; full RQ2 decoder keys also work if you want the same model for classification and generation.

Generation stops when `human_count + synthetic_count >= target` (`target` defaults to the majority emotion count). Synthetic files are written under `Datasets/generated/...` and kept separate from `Datasets/GroundTruth.csv`.

### Inference with a saved encoder

```bash
emotion-cls predict --model-path models/your-checkpoint --input reviews.txt --output predicted_reviews.csv
```

---

## Configuration

- `configs/default.yaml` — folds (10), `max_labels` (3), emotions, training hyperparameters  
- `configs/models.yaml` — full encoder + decoder catalogue  
- `configs/imbalance.yaml` — method descriptions and suggested bins  

Override via CLI flags or edit the YAML. Export the resolved config:

```bash
emotion-cls export-run-config --out outputs/resolved_config.yaml
```

## Legacy code

Previous notebooks and scripts live under `legacy/` (`Finne-tunning_implementation/`, `generate_emotion_reviews.py`, `use_model.py`). Prefer the `emotion-cls` CLI for new runs; legacy artefacts remain for reference and historical results.

## Citation

Please cite the forthcoming IST article and the prior ground-truth paper:

- Motger et al. *What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews*. arXiv:2505.23452, 2025.
