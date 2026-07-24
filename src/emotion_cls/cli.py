"""Command-line interface for the replication experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import click
import yaml

from emotion_cls.config import load_default_config, load_env, resolve_path


def _apply_overrides(cfg: dict, values: dict) -> dict:
    cfg = deepcopy(cfg)
    for key, val in values.items():
        if val is None:
            continue
        # dotted paths like training.encoder
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


@click.group()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
@click.option("--models-config", default="configs/models.yaml", show_default=True)
@click.pass_context
def main(ctx: click.Context, config_path: str, models_config: str) -> None:
    """Emotion classification replication package."""
    load_env()
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_default_config(config_path, models_config)


@main.command("list-models")
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """Print encoder and decoder catalogues."""
    models = ctx.obj["cfg"]["_models"]
    click.echo("Encoders:")
    for k, v in models["encoders"].items():
        mark = " (baseline)" if v.get("baseline") else ""
        click.echo(f"  - {k}: {v['hub_id']}{mark}")
    click.echo("Decoders:")
    for k, v in models["decoders"].items():
        click.echo(f"  - {k}: backend={v['backend']} id={v['model_id']} ({v['access']})")


@main.command("train-encoder")
@click.option("--encoder", default=None, help="Key from configs/models.yaml or raw HF id")
@click.option("--head", type=click.Choice(["multilabel", "binary"]), default=None)
@click.option(
    "--imbalance",
    default=None,
    help="none|bce_pos_weight|bce_weight|focal|adaptive_focal|undersample|genai_aug (comma-ok)",
)
@click.option("--undersample-cutoff", type=int, default=None)
@click.option("--aug-inject-n", type=int, default=None, help="Synthetic positives per emotion (bins of 50)")
@click.option("--synthetic-ml-path", default=None, help="CSV with synthetic multilabel rows for genai_aug")
@click.option("--folds", type=int, default=None)
@click.option("--epochs", type=float, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--no-resume", is_flag=True, help="Ignore existing fold checkpoints and re-run from scratch")
@click.pass_context
def train_encoder(
    ctx: click.Context,
    encoder: str | None,
    head: str | None,
    imbalance: str | None,
    undersample_cutoff: int | None,
    aug_inject_n: int | None,
    synthetic_ml_path: str | None,
    folds: int | None,
    epochs: float | None,
    dry_run: bool,
    no_resume: bool,
) -> None:
    """Fine-tune encoder-only models (Hugging Face Transformers)."""
    # Isolate GenAI inject sizes into distinct resumable output dirs
    imb_method = imbalance
    if imbalance and "genai_aug" in imbalance and aug_inject_n is not None:
        parts = []
        for p in imbalance.split(","):
            p = p.strip()
            if p == "genai_aug":
                parts.append(f"genai_aug_n{aug_inject_n}")
            else:
                parts.append(p)
        imb_method = ",".join(parts)

    cfg = _apply_overrides(
        ctx.obj["cfg"],
        {
            "training.encoder": encoder,
            "training.head": head,
            "training.epochs": epochs,
            "evaluation.n_folds": folds,
            "imbalance.method": imb_method,
            "imbalance.undersample_cutoff": undersample_cutoff,
            "imbalance.aug_inject_n": aug_inject_n,
            "imbalance.synthetic_ml_path": synthetic_ml_path,
            "experiment.resume": False if no_resume else None,
        },
    )
    from emotion_cls.training.encoder import run_binary_ensemble_cv, run_multilabel_cv

    head = cfg["training"]["head"]
    resume = bool(cfg.get("experiment", {}).get("resume", True)) and not no_resume
    if head == "multilabel":
        results = run_multilabel_cv(cfg, dry_run=dry_run, resume=resume)
        if not dry_run:
            click.echo(f"Folds done: {len(results)}")
    else:
        results = run_binary_ensemble_cv(cfg, dry_run=dry_run, resume=resume)
        if not dry_run:
            click.echo(f"Binary ensemble folds done: {len(results)}")


@main.command("classify-decoder")
@click.option("--decoder", required=True, help="Key from configs/models.yaml decoders")
@click.option(
    "--strategy",
    type=click.Choice(["zero_shot", "few_shot_guidelines", "few_shot_guidelines_dataset"]),
    default=None,
)
@click.option("--folds", type=int, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--no-resume", is_flag=True, help="Ignore saved predictions and re-run from scratch")
@click.pass_context
def classify_decoder(
    ctx: click.Context,
    decoder: str,
    strategy: str | None,
    folds: int | None,
    dry_run: bool,
    no_resume: bool,
) -> None:
    """Zero-/few-shot classification with Ollama or proprietary APIs."""
    cfg = _apply_overrides(
        ctx.obj["cfg"],
        {
            "decoding.strategy": strategy,
            "evaluation.n_folds": folds,
            "experiment.resume": False if no_resume else None,
        },
    )
    from emotion_cls.decoding.classify import run_decoder_classification

    resume = bool(cfg.get("experiment", {}).get("resume", True)) and not no_resume
    out = run_decoder_classification(cfg, decoder, dry_run=dry_run, resume=resume)
    click.echo(f"Output: {out}")


@main.command("generate")
@click.option("--decoder", required=True, help="Generator key from configs/models.yaml")
@click.option("--emotion", required=True)
@click.option("--target-count", type=int, default=None, help="Desired total positives (default: majority parity)")
@click.option("--batch-size", type=int, default=10)
@click.option(
    "--strategy",
    type=click.Choice(["zero_shot", "few_shot_guidelines", "few_shot_guidelines_dataset"]),
    default="few_shot_guidelines_dataset",
)
@click.option("--no-resume", is_flag=True)
@click.pass_context
def generate(
    ctx: click.Context,
    decoder: str,
    emotion: str,
    target_count: int | None,
    batch_size: int,
    strategy: str,
    no_resume: bool,
) -> None:
    """Generate synthetic reviews until a target count is reached."""
    from emotion_cls.augmentation.generate import generate_for_emotion

    if no_resume:
        ctx.obj["cfg"].setdefault("experiment", {})["resume"] = False
    path = generate_for_emotion(
        ctx.obj["cfg"],
        decoder_key=decoder,
        emotion=emotion,
        target_count=target_count,
        batch_size=batch_size,
        strategy=strategy,
        resume=not no_resume,
    )
    click.echo(f"Wrote {path}")


@main.command("generate-parity")
@click.option("--decoder", required=True)
@click.option(
    "--strategy",
    type=click.Choice(["zero_shot", "few_shot_guidelines", "few_shot_guidelines_dataset"]),
    default="few_shot_guidelines_dataset",
)
@click.option("--batch-size", type=int, default=10)
@click.option("--no-resume", is_flag=True)
@click.pass_context
def generate_parity(
    ctx: click.Context, decoder: str, strategy: str, batch_size: int, no_resume: bool
) -> None:
    """Generate synthetic reviews for all Plutchik emotions up to majority parity."""
    from emotion_cls.augmentation.generate import generate_for_emotion

    if no_resume:
        ctx.obj["cfg"].setdefault("experiment", {})["resume"] = False
    for emotion in ctx.obj["cfg"]["data"]["generation_emotions"]:
        path = generate_for_emotion(
            ctx.obj["cfg"],
            decoder_key=decoder,
            emotion=emotion,
            target_count=None,
            batch_size=batch_size,
            strategy=strategy,
            resume=not no_resume,
        )
        click.echo(f"{emotion}: {path}")


@main.command("rank-augmentation")
@click.pass_context
def rank_augmentation(ctx: click.Context) -> None:
    """Rank synthetic corpora with embedding-based utility metrics."""
    from emotion_cls.augmentation.utility import rank_augmentation_utility

    out = rank_augmentation_utility(ctx.obj["cfg"])
    click.echo(f"Wrote rankings under {out}")


@main.command("predict")
@click.option("--model-path", required=True, type=click.Path(exists=True))
@click.option("--input", "input_path", default="reviews.txt")
@click.option("--output", "output_path", default="predicted_reviews.csv")
@click.option("--max-labels", type=int, default=3)
@click.pass_context
def predict(ctx: click.Context, model_path: str, input_path: str, output_path: str, max_labels: int) -> None:
    """Run a fine-tuned multilabel encoder on a text file (one review per line)."""
    import csv

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from emotion_cls.training.metrics import top_k_from_logits

    emotions = ctx.obj["cfg"]["data"]["emotions"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    lines = [ln.strip() for ln in Path(input_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Review", "Emotions"])
        for line in lines:
            inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits.cpu().numpy()
            pred = top_k_from_logits(logits, k=max_labels)[0]
            labels = [emotions[i] for i, v in enumerate(pred) if v == 1]
            w.writerow([line, ", ".join(labels)])
    click.echo(f"Wrote {output_path}")


@main.command("sweep-undersample")
@click.option("--encoder", default="bert-base-cased")
@click.option("--cutoffs", default="50,100,150,200,250,300,350")
@click.option("--dry-run", is_flag=True)
@click.option("--no-resume", is_flag=True)
@click.pass_context
def sweep_undersample(
    ctx: click.Context, encoder: str, cutoffs: str, dry_run: bool, no_resume: bool
) -> None:
    """Undersampling cutoff sweep for multilabel fine-tuning.

    Each cutoff writes to its own resumable directory
    ``outputs/encoder_multilabel/<encoder>/undersample_c<cutoff>/``.
    """
    from emotion_cls.training.encoder import run_multilabel_cv

    for cutoff in [int(x) for x in cutoffs.split(",")]:
        cfg = _apply_overrides(
            ctx.obj["cfg"],
            {
                "training.encoder": encoder,
                "training.head": "multilabel",
                # Tag isolates output dirs per cutoff while still enabling undersampling
                "imbalance.method": f"undersample_c{cutoff}",
                "imbalance.undersample_cutoff": cutoff,
                "experiment.resume": False if no_resume else None,
            },
        )
        resume = bool(cfg.get("experiment", {}).get("resume", True)) and not no_resume
        click.echo(f"=== undersample cutoff={cutoff} ===")
        run_multilabel_cv(cfg, dry_run=dry_run, resume=resume)


@main.command("export-run-config")
@click.option("--out", default="outputs/resolved_config.yaml")
@click.pass_context
def export_run_config(ctx: click.Context, out: str) -> None:
    """Dump the resolved config (for reproducibility)."""
    path = resolve_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in ctx.obj["cfg"].items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    click.echo(f"Wrote {path}")


if __name__ == "__main__":
    main()
