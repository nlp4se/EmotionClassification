# Emotion Classification from Mobile App Reviews

Replication package accompanying the study on encoder-only and decoder-only large language models for multi-label emotion classification of mobile app reviews, including data-imbalance mitigation and synthetic data augmentation.

The human-labelled ground truth and annotation guidelines are those introduced in [Motger et al. (2025)](https://arxiv.org/abs/2505.23452).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) for local open-source generative models
- API keys for proprietary models (optional, depending on the experiment)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
cp .env.example .env               # then set keys as needed
```

Environment variables are documented in `.env.example`.

## Repository structure

| Path | Description |
|------|-------------|
| `configs/` | Experiment configuration (models, training, imbalance) |
| `src/emotion_cls/` | Source package and `emotion-cls` CLI |
| `Datasets/` | Ground-truth CSV, annotation guidelines, and synthetic review corpora |
| `scripts/` | Auxiliary data-preparation scripts |
| `paper/IST_Emotions/` | Manuscript sources |
| `legacy/` | Earlier exploratory notebooks and scripts |
| `outputs/` | Experiment outputs (created at runtime) |
| `models/` | Fine-tuned checkpoints (created at runtime; gitignored) |

## Models

Encoder-only and decoder-only models are declared in `configs/models.yaml`.

```bash
emotion-cls list-models
```

Encoder fine-tuning uses Hugging Face Transformers (PyTorch). Open-source decoder-only models are served with Ollama; proprietary models use their official APIs.

## Experiments

Default settings (10-fold multilabel stratified CV, top-3 labels, emotion set, hyperparameters) are in `configs/default.yaml`. Override via CLI flags or by editing the YAML files.

### Encoder-only classification

```bash
# Multi-label fine-tuning
emotion-cls train-encoder --encoder bert-base-cased --head multilabel
emotion-cls train-encoder --encoder roberta-large --head multilabel

# Binary ensemble (one classifier per emotion; top-3 assembly)
emotion-cls train-encoder --encoder bert-base-cased --head binary
```

Results are written under `outputs/encoder_multilabel/` and `outputs/encoder_binary/`.

### Decoder-only classification

Prompting strategies: `zero_shot`, `few_shot_guidelines`, `few_shot_guidelines_dataset`.
Prompts are built from `Datasets/guidelines/Annotation Guidelines.txt` (the official annotation guidelines).

```bash
emotion-cls classify-decoder --decoder gemma3-4b --strategy zero_shot
emotion-cls classify-decoder --decoder gemma3-4b --strategy zero_shot --temperature 0.3
emotion-cls classify-decoder --decoder claude-opus-4-6 --strategy few_shot_guidelines_dataset
```

Temperature is a tunable RQ2 factor (`--temperature`; default `0.0` from `configs/default.yaml`).
Grid candidates are listed under `decoding.temperature_grid` (`0.0`, `0.3`, `0.7`).
Each temperature writes to its own resumable directory
`outputs/decoder_classify/<decoder>/<strategy>/t<temp>/`.

### Data-imbalance mitigation

| `--imbalance` | Description |
|---------------|-------------|
| `none` | Baseline multilabel BCE |
| `bce_pos_weight` | `pos_weight[c] = neg_c / pos_c` |
| `bce_weight` | `w[c] = N / (\|E\| · pos_c)` |
| `focal` | Focal loss with class-wise α |
| `adaptive_focal` | Adaptive Focal Loss |
| `undersample` | Cap positives per emotion (`--undersample-cutoff`) |
| `genai_aug` | Inject synthetic reviews (`--aug-inject-n`, `--synthetic-ml-path`) |

Methods may be combined with commas (e.g. `undersample,bce_pos_weight`).

```bash
emotion-cls train-encoder --encoder bert-base-cased --imbalance bce_pos_weight
emotion-cls train-encoder --encoder bert-base-cased --imbalance undersample --undersample-cutoff 150
emotion-cls sweep-undersample --encoder bert-base-cased --cutoffs 50,100,150,200,250,300,350

emotion-cls train-encoder --encoder bert-base-cased \
  --imbalance genai_aug,bce_pos_weight \
  --aug-inject-n 100 \
  --synthetic-ml-path Datasets/synthetic_multilabel.csv
```

Build a multilabel synthetic CSV from provider folders under `Datasets/`:

```bash
python scripts/build_synthetic_multilabel.py \
  --strategy few_shot_guidelines_dataset \
  --genai Claude \
  --n-per-emotion 100 \
  --out Datasets/synthetic_multilabel.csv
```

### Synthetic generation and augmentation utility

```bash
# Rank existing synthetic corpora (diversity / novelty / on-emotion + Borda)
emotion-cls rank-augmentation

# Generate until majority-class parity
emotion-cls generate --decoder claude --emotion Fear --strategy few_shot_guidelines_dataset
emotion-cls generate-parity --decoder claude --strategy few_shot_guidelines_dataset
```

Synthetic outputs are stored under `Datasets/generated/` and are not mixed into `Datasets/GroundTruth.csv`.

### Inference

```bash
emotion-cls predict \
  --model-path models/<checkpoint> \
  --input reviews.txt \
  --output predicted_reviews.csv
```

## Configuration

- `configs/default.yaml` — data paths, CV, training hyperparameters  
- `configs/models.yaml` — encoder and decoder catalogues  
- `configs/imbalance.yaml` — imbalance method definitions  

```bash
emotion-cls export-run-config --out outputs/resolved_config.yaml
```

## Experiment logging and resume

Runs are **resumable by default** (`experiment.resume: true` in `configs/default.yaml`).
Re-executing the same command skips finished work and continues from the last checkpoint.

| Experiment | Resume unit | Periodic flush |
|------------|-------------|----------------|
| Encoder multilabel | completed `fold_N/metrics.json` | after each fold (+ HF epoch checkpoints) |
| Encoder binary | fold metrics; within fold, `probs.npy` / `emotions_done.json` per emotion | after each emotion |
| Decoder classify | fold metrics; within fold, per-sentence `predictions.csv` | after **every** sentence |
| Generation | existing synthetic CSV row count | after every batch |
| Undersample sweep | per-cutoff directory `undersample_c<N>/` | same as encoder |

Each run directory also stores:

- `run_meta.json` — config snapshot, start/end timestamps, run id  
- `events.jsonl` — append-only event log (fold start/done, LLM calls, errors)  
- `progress.json` — completed folds/units  
- `usage_totals.json` — aggregated prompt/completion tokens and latency (API / Ollama)

Force a clean re-run with `--no-resume`.

```bash
emotion-cls train-encoder --encoder bert-base-cased --head multilabel
emotion-cls classify-decoder --decoder gemma3-4b --strategy zero_shot
# after a crash, the same commands continue where they left off
```

## Licence and citation

Please cite the associated publication when using this package, and the ground-truth dataset paper:

> Motger, Q., Oriol, M., Tiessler, M., Franch, X., & Marco, J. (2025). *What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews*. arXiv:2505.23452.
