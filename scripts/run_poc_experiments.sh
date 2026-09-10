#!/usr/bin/env bash
# Proof-of-concept smoke test — exercises every experiment type in the pipeline
# end to end with minimum data/epochs/folds/temperatures, so you can confirm
# everything works before committing to the full multi-day run.
#
# Launch detached so you can close the connection:
#   nohup bash scripts/run_poc_experiments.sh > logs/poc.log 2>&1 &
#
# Uses configs/smoke.yaml: a 35-row, label-diverse ground-truth subset
# (Datasets/smoke/ground_truth_smoke.csv, built below by
# scripts/make_smoke_subset.py), 2-fold CV, 1 epoch, max_length 64, and a
# single temperature. All outputs land under outputs_smoke/ + Datasets/smoke/
# so nothing collides with a real run's outputs/ + Datasets/generated/.
#
# Does NOT abort on a single step failure: every step is logged pass/fail and
# a summary is printed (and written to logs/poc_summary.txt) at the end, so
# one broken step doesn't hide the results of everything else.
#
# Calls proprietary APIs (OpenAI/Gemini/Anthropic/Mistral) a handful of times
# each (~35 short classification calls + a few generation calls per
# provider) — trivial cost, but real spend. Set SKIP_API=1 to skip those.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

mkdir -p logs Datasets/smoke outputs_smoke

CONFIG="configs/smoke.yaml"
CLI=(emotion-cls --config "$CONFIG")

SKIP_API="${SKIP_API:-0}"       # set 1 to skip proprietary decoder + generation calls
SKIP_OLLAMA="${SKIP_OLLAMA:-0}" # set 1 to skip local Ollama decoder classify

SUMMARY_FILE="logs/poc_summary.txt"
: > "$SUMMARY_FILE"

log() { echo "[$(date -Is)] $*"; }
phase() { echo; echo "======== $* ======== $(date -Is)"; echo; }

STEP_N=0
run_step() {
  # run_step "<label>" <command...>
  local label="$1"; shift
  STEP_N=$((STEP_N + 1))
  local logfile="logs/poc_step_$(printf '%02d' "$STEP_N")_$(echo "$label" | tr -c 'A-Za-z0-9' '_' | cut -c1-60).log"
  log "STEP $STEP_N: $label"
  log "  -> $* (log: $logfile)"
  if "$@" >"$logfile" 2>&1; then
    echo "PASS  $label" >>"$SUMMARY_FILE"
    log "  PASS"
  else
    echo "FAIL  $label  (see $logfile)" >>"$SUMMARY_FILE"
    log "  FAIL (see $logfile)"
  fi
}
skip_step() {
  echo "SKIP  $1" >>"$SUMMARY_FILE"
  log "SKIP: $1"
}

check_decoder_errors() {
  # classify-decoder swallows per-sentence errors (bad API key, rate limit, ...)
  # and still exits 0 so a long sweep survives a few flaky calls. That means a
  # 100%-broken key (e.g. an expired/invalid one) looks like a PASS from the
  # exit code alone. Inspect predictions.csv and fail the step if most/all
  # sentences errored.
  local decoder="$1" strategy="$2"
  python3 - "$decoder" "$strategy" <<'PY'
import sys, glob
import pandas as pd
decoder, strategy = sys.argv[1], sys.argv[2]
files = glob.glob(f"outputs_smoke/decoder_classify/{decoder}/{strategy}/*/fold_*/predictions.csv")
if not files:
    print("no predictions.csv found")
    sys.exit(1)
total, errors = 0, 0
for f in files:
    df = pd.read_csv(f)
    total += len(df)
    if "error" in df.columns:
        errors += df["error"].astype(str).replace("nan", "").str.len().gt(0).sum()
rate = errors / total if total else 1.0
print(f"{errors}/{total} sentences errored ({rate:.0%}) across {len(files)} fold file(s)")
sys.exit(1 if rate > 0.5 else 0)
PY
}

run_decoder_step() {
  # run_decoder_step "<label>" <decoder> <strategy> <command...>
  local label="$1" decoder="$2" strategy="$3"; shift 3
  STEP_N=$((STEP_N + 1))
  local logfile="logs/poc_step_$(printf '%02d' "$STEP_N")_$(echo "$label" | tr -c 'A-Za-z0-9' '_' | cut -c1-60).log"
  log "STEP $STEP_N: $label"
  log "  -> $* (log: $logfile)"
  if "$@" >"$logfile" 2>&1 && check_decoder_errors "$decoder" "$strategy" >>"$logfile" 2>&1; then
    echo "PASS  $label" >>"$SUMMARY_FILE"
    log "  PASS"
  else
    echo "FAIL  $label  (see $logfile)" >>"$SUMMARY_FILE"
    log "  FAIL (see $logfile)"
  fi
}

sync_generated_into_smoke_datasets() {
  # Mirror scripts/run_all_experiments.sh's layout so rank-augmentation /
  # build_synthetic_multilabel.py (which read Datasets/<Provider>/<Strategy>/)
  # can see what generate-parity wrote under Datasets/smoke/generated/.
  declare -A MAP=(
    [claude]="Claude/FewShootExamples"
    [gemini]="Gemini/FewShootExample"
    [openai]="OpenAi/FewShootExample"
    [mistral-large]="Mistral/FewShootExamples"
  )
  for g in "${!MAP[@]}"; do
    src="Datasets/smoke/generated/$g/few_shot_guidelines_dataset"
    dst="Datasets/smoke/${MAP[$g]}"
    mkdir -p "$dst"
    if [[ -d "$src" ]]; then
      find "$src" -maxdepth 1 -type f -name '*.csv' -exec cp -f {} "$dst/" \;
      log "Synced $src -> $dst"
    fi
  done
}

############################################
phase "0) Setup: smoke ground-truth subset + environment check"
############################################
if [[ ! -f Datasets/smoke/ground_truth_smoke.csv ]]; then
  run_step "build smoke ground-truth subset" \
    python scripts/make_smoke_subset.py --n-per-emotion 4 --out Datasets/smoke/ground_truth_smoke.csv
else
  log "Smoke ground truth already present, skipping build"
fi
run_step "list-models" "${CLI[@]}" list-models
run_step "export-run-config" "${CLI[@]}" export-run-config --out outputs_smoke/resolved_config.yaml

############################################
phase "1) Encoder multilabel — one representative per model family (base only)"
############################################
for enc in distilbert-base-cased bert-base-cased roberta-base deberta-v3-base xlnet-base-cased; do
  run_step "train-encoder multilabel $enc" \
    "${CLI[@]}" train-encoder --encoder "$enc" --head multilabel --imbalance none
done

############################################
phase "2) Encoder multilabel — imbalance-loss methods (fast encoder only)"
############################################
for imb in bce_pos_weight bce_weight focal adaptive_focal; do
  run_step "train-encoder imbalance $imb" \
    "${CLI[@]}" train-encoder --encoder distilbert-base-cased --head multilabel --imbalance "$imb"
done

############################################
phase "3) Undersample sweep"
############################################
run_step "sweep-undersample" \
  "${CLI[@]}" sweep-undersample --encoder distilbert-base-cased --cutoffs 2,4

############################################
phase "4) Encoder binary ensemble"
############################################
run_step "train-encoder binary" \
  "${CLI[@]}" train-encoder --encoder distilbert-base-cased --head binary

############################################
phase "5) Predict (inference) on an existing checkpoint"
############################################
if [[ -d models/test_distilbert ]]; then
  printf 'This app is wonderful, I love it!\nWhy does this keep crashing, so annoying.\n' \
    >outputs_smoke/poc_reviews.txt
  run_step "predict" \
    "${CLI[@]}" predict --model-path models/test_distilbert \
    --input outputs_smoke/poc_reviews.txt --output outputs_smoke/poc_predicted.csv
else
  skip_step "predict (no models/test_distilbert checkpoint present)"
fi

############################################
phase "6) Decoder classify — Ollama (open-source, local)"
############################################
if [[ "$SKIP_OLLAMA" != "1" ]]; then
  for strat in zero_shot few_shot_guidelines few_shot_guidelines_dataset; do
    run_decoder_step "classify-decoder gemma3-4b $strat" gemma3-4b "$strat" \
      "${CLI[@]}" classify-decoder --decoder gemma3-4b --strategy "$strat" --temperature 0.0
  done
  run_decoder_step "classify-decoder gemma3-4b temperature override (+ unload)" gemma3-4b zero_shot \
    "${CLI[@]}" classify-decoder --decoder gemma3-4b --strategy zero_shot --temperature 0.3 --unload-ollama
else
  skip_step "decoder classify Ollama (SKIP_OLLAMA=1)"
fi

############################################
phase "7) Decoder classify — proprietary APIs"
############################################
if [[ "$SKIP_API" != "1" ]]; then
  for dec in gpt-5.3-chat gemini-3-flash claude-haiku-4-5 mistral-large-2512; do
    run_decoder_step "classify-decoder $dec" "$dec" zero_shot \
      "${CLI[@]}" classify-decoder --decoder "$dec" --strategy zero_shot
  done
else
  skip_step "decoder classify APIs (SKIP_API=1)"
fi

############################################
phase "8) Synthetic generation (RQ4) — one call per generator, capped emotions"
############################################
if [[ "$SKIP_API" != "1" ]]; then
  for gen in claude gemini openai mistral-large; do
    run_step "generate-parity $gen" \
      "${CLI[@]}" generate-parity --decoder "$gen" --strategy few_shot_guidelines_dataset --batch-size 3
  done
  sync_generated_into_smoke_datasets
else
  skip_step "generate-parity (SKIP_API=1)"
fi

############################################
phase "9) Augmentation utility ranking + synthetic multilabel build"
############################################
if [[ "$SKIP_API" != "1" ]]; then
  run_step "rank-augmentation" "${CLI[@]}" rank-augmentation
  run_step "build_synthetic_multilabel" \
    python scripts/build_synthetic_multilabel.py --config "$CONFIG" \
    --strategy few_shot_guidelines_dataset --genai Claude --n-per-emotion 5 \
    --out Datasets/smoke/synthetic_multilabel_smoke.csv
else
  skip_step "rank-augmentation / build_synthetic_multilabel (SKIP_API=1, no synthetic data generated)"
fi

############################################
phase "10) GenAI-augmentation imbalance mitigation"
############################################
if [[ -s Datasets/smoke/synthetic_multilabel_smoke.csv ]]; then
  run_step "train-encoder genai_aug" \
    "${CLI[@]}" train-encoder --encoder distilbert-base-cased --head multilabel \
    --imbalance genai_aug,bce_pos_weight --aug-inject-n 5 \
    --synthetic-ml-path Datasets/smoke/synthetic_multilabel_smoke.csv
else
  skip_step "train-encoder genai_aug (no synthetic CSV built)"
fi

############################################
phase "SUMMARY"
############################################
cat "$SUMMARY_FILE"
n_pass=$(grep -c '^PASS' "$SUMMARY_FILE" || true)
n_fail=$(grep -c '^FAIL' "$SUMMARY_FILE" || true)
n_skip=$(grep -c '^SKIP' "$SUMMARY_FILE" || true)
log "PoC finished: $n_pass passed, $n_fail failed, $n_skip skipped."
log "Full summary: $SUMMARY_FILE — per-step logs: logs/poc_step_*.log"
if [[ "$n_fail" != "0" ]]; then
  exit 1
fi
