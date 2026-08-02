#!/usr/bin/env bash
# Master experiment runner — start once and detach:
#   nohup bash scripts/run_all_experiments.sh > logs/master.log 2>&1 &
#
# Resume-safe: re-running skips completed folds / sentences / generations.
# GPU-heavy phases run sequentially; API classify + generation start in parallel.
#
# Deliberately does NOT use `set -e`: this runs unattended for hours/days
# across 9 encoders x several imbalance methods, 6 local + 4 API decoders x 3
# strategies x 3 temperatures, and 4 generators. A single failure (CUDA OOM on
# one encoder, a transient API error, one bad ollama pull) must not abort
# every independent phase that comes after it. Every command is run through
# run_logged, which logs a WARNING and continues on failure; failures are
# collected in logs/master_failures.log and summarized at the end. Re-running
# this script picks up unfinished work via the resume mechanism.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

mkdir -p logs "$ROOT/Datasets/generated"
FAILURES_LOG="logs/master_failures.log"
: > "$FAILURES_LOG"

STRATEGY="${STRATEGY:-few_shot_guidelines_dataset}"
BEST_ENC="${BEST_ENC:-}"          # empty => auto-pick after RQ1 multilabel
SYNTH_CSV="${SYNTH_CSV:-Datasets/synthetic_multilabel_best.csv}"
N_PER_EMOTION="${N_PER_EMOTION:-350}"
SKIP_RQ2_OSS="${SKIP_RQ2_OSS:-0}" # set 1 to skip local Ollama classify
SKIP_RQ2_API="${SKIP_RQ2_API:-0}"
SKIP_RQ4_GEN="${SKIP_RQ4_GEN:-0}"
SKIP_RQ1_BINARY="${SKIP_RQ1_BINARY:-0}"
SKIP_RQ3="${SKIP_RQ3:-0}"

ENCODERS=(
  bert-base-cased bert-large-cased
  roberta-base roberta-large
  distilbert-base-cased
  deberta-v3-base deberta-v3-large
  xlnet-base-cased xlnet-large-cased
)
DECODERS_OSS=(deepseek-r1-8b mistral gpt-oss-20b gemma3-4b llama3.1-8b qwen3-8b)
DECODERS_API=(gpt-5.3-chat gemini-3-flash claude-opus-4-6 mistral-large-2512)
GENERATORS=(claude gemini openai mistral-large)
STRATEGIES=(zero_shot few_shot_guidelines few_shot_guidelines_dataset)
TEMPS=(0.0 0.3 0.7)
IMBALANCE_LOSSES=(none bce_pos_weight bce_weight focal adaptive_focal)
CUTOFFS=50,100,150,200,250,300,350
AUG_NS=(50 100 150 200 250 300 350)

log() { echo "[$(date -Is)] $*"; }
phase() { echo; echo "======== $* ======== $(date -Is)"; echo; }

run_logged() {
  local logfile="$1"; shift
  log "RUN -> $logfile :: $*"
  # Append to phase log and master stdout. Never let a single failed command
  # abort the whole unattended run: log it and keep going (see header note).
  if ! "$@" >>"$logfile" 2>&1; then
    local msg="FAILED [$(date -Is)] (see $logfile) :: $*"
    log "WARN: $msg"
    echo "$msg" >>"$FAILURES_LOG"
    return 1
  fi
}

classify_error_rate_ok() {
  # `classify-decoder` swallows per-sentence errors (bad API key, rate limit,
  # transient 5xx, ...) and still exits 0 so one bad key doesn't nuke the CV
  # loop. That means a decoder whose key is entirely broken looks identical
  # to a clean run from the exit code alone. Flag it explicitly instead.
  local decoder="$1" strategy="$2" temp="$3"
  python3 - "$decoder" "$strategy" "$temp" <<'PY'
import sys, glob
import pandas as pd
decoder, strategy, temp = sys.argv[1], sys.argv[2], sys.argv[3]
tag = f"t{float(temp):g}"
files = glob.glob(f"outputs/decoder_classify/{decoder}/{strategy}/{tag}/fold_*/predictions.csv")
if not files:
    sys.exit(1)
total, errors = 0, 0
for f in files:
    df = pd.read_csv(f)
    total += len(df)
    if "error" in df.columns:
        errors += df["error"].astype(str).replace("nan", "").str.len().gt(0).sum()
rate = errors / total if total else 1.0
print(f"{decoder}/{strategy}/{tag}: {errors}/{total} sentences errored ({rate:.0%})")
sys.exit(1 if rate > 0.5 else 0)
PY
}

pick_best_encoder() {
  python - <<'PY'
from pathlib import Path
import json
import sys
root = Path("outputs/encoder_multilabel")
best, best_f1 = None, -1.0
for summary in sorted(root.glob("*/none/cv_summary.json")):
    data = json.loads(summary.read_text())
    f1 = float(data.get("aggregate", {}).get("macro_f1", {}).get("mean", -1))
    enc = summary.parent.parent.name
    print(f"  candidate {enc}: macro_f1={f1:.4f}", file=sys.stderr)
    if f1 > best_f1:
        best, best_f1 = enc, f1
if not best:
    raise SystemExit("No multilabel cv_summary.json found under outputs/encoder_multilabel/*/none/")
print(best)
PY
}

sync_generated_into_datasets() {
  # Map generator key -> Datasets/<Provider>/<StrategyFolder>
  declare -A MAP=(
    [claude]="Claude/FewShootExamples"
    [gemini]="Gemini/FewShootExample"
    [openai]="OpenAi/FewShootExample"
    [mistral-large]="Mistral/FewShootExamples"
  )
  for g in "${GENERATORS[@]}"; do
    src="Datasets/generated/$g/$STRATEGY"
    dst="Datasets/${MAP[$g]}"
    mkdir -p "$dst"
    if [[ -d "$src" ]]; then
      log "Sync $src -> $dst"
      # Prefer Neutral and all emotion CSVs; overwrite with newer gens
      find "$src" -maxdepth 1 -type f -name '*.csv' -print0 | while IFS= read -r -d '' f; do
        base="$(basename "$f")"
        # Normalize few_shot_guidelines_dataset suffix for loader-friendly names
        out="$dst/$base"
        cp -f "$f" "$out"
      done
    else
      log "WARN: missing $src (generation may still be running or failed)"
    fi
  done
}

choose_genai_for_build() {
  # Prefer Claude if present; else first available provider folder with CSVs
  for pair in "Claude:Claude" "Gemini:Gemini" "OpenAi:GPT" "Mistral:Mistral"; do
    folder="${pair%%:*}"
    genai="${pair##*:}"
    if compgen -G "Datasets/$folder/*/*.csv" > /dev/null || compgen -G "Datasets/$folder/*/*/*.csv" > /dev/null; then
      if find "Datasets/$folder" -type f -name '*.csv' | head -1 | grep -q .; then
        echo "$genai"
        return 0
      fi
    fi
  done
  echo "Claude"
}

############################################
phase "0) Environment check"
############################################
log "ROOT=$ROOT"
emotion-cls list-models >>logs/master_env.log 2>&1 || true
df -h . | tail -1 | tee -a logs/master_env.log
du -sh outputs .cache/huggingface 2>/dev/null | tee -a logs/master_env.log || true

############################################
phase "Background: RQ2 API + RQ4 generation (no GPU contention with local Ollama)"
############################################
API_PID=""
GEN_PID=""

if [[ "$SKIP_RQ2_API" != "1" ]]; then
  (
    set -uo pipefail
    source "$ROOT/.venv/bin/activate"
    cd "$ROOT"
    for d in "${DECODERS_API[@]}"; do
      for s in "${STRATEGIES[@]}"; do
        for t in "${TEMPS[@]}"; do
          echo "===== API $d | $s | T=$t $(date -Is) ====="
          if emotion-cls classify-decoder --decoder "$d" --strategy "$s" --temperature "$t"; then
            classify_error_rate_ok "$d" "$s" "$t" \
              || echo "WARN: API $d | $s | T=$t has a >50% sentence error rate (check the key?), continuing"
          else
            echo "WARN: API $d | $s | T=$t failed, continuing"
          fi
        done
      done
    done
    echo "===== RQ2 API DONE $(date -Is) ====="
  ) >logs/rq2_decoder_api.log 2>&1 &
  API_PID=$!
  log "RQ2 API background PID=$API_PID"
fi

if [[ "$SKIP_RQ4_GEN" != "1" ]]; then
  (
    set -uo pipefail
    source "$ROOT/.venv/bin/activate"
    cd "$ROOT"
    for g in "${GENERATORS[@]}"; do
      echo "===== generate-parity $g $(date -Is) ====="
      emotion-cls generate-parity --decoder "$g" --strategy "$STRATEGY" \
        || echo "WARN: generate-parity $g failed, continuing"
    done
    echo "===== RQ4 GENERATE DONE $(date -Is) ====="
  ) >logs/rq4_generate_parity.log 2>&1 &
  GEN_PID=$!
  log "RQ4 generate background PID=$GEN_PID"
fi

############################################
phase "1) RQ1 encoder multilabel (GPU)"
############################################
for enc in "${ENCODERS[@]}"; do
  log "===== MULTILABEL $enc ====="
  run_logged logs/rq1_multilabel_all.log \
    emotion-cls train-encoder --encoder "$enc" --head multilabel --imbalance none
done
log "RQ1 multilabel DONE"

############################################
phase "2) RQ1 encoder binary (GPU)"
############################################
if [[ "$SKIP_RQ1_BINARY" != "1" ]]; then
  for enc in "${ENCODERS[@]}"; do
    log "===== BINARY $enc ====="
    run_logged logs/rq1_binary_all.log \
      emotion-cls train-encoder --encoder "$enc" --head binary
  done
  log "RQ1 binary DONE"
else
  log "SKIP RQ1 binary"
fi

############################################
phase "3) RQ2 open-source Ollama classify (GPU) — pull / eval / remove per model"
############################################
if [[ "$SKIP_RQ2_OSS" != "1" ]]; then
  for d in "${DECODERS_OSS[@]}"; do
    log "===== OSS model lifecycle start: $d ====="
    # Resolve Ollama id and ensure pulled once for this decoder
    model_id="$(
      python - <<PY
from emotion_cls.config import load_default_config, decoder_spec
cfg = load_default_config()
print(decoder_spec(cfg, "$d")["model_id"])
PY
    )"
    python - <<PY
from emotion_cls.decoding.ollama_lifecycle import ensure_ollama_model
ensure_ollama_model("$model_id", pull_if_missing=True)
PY

    last_s="${STRATEGIES[-1]}"
    last_t="${TEMPS[-1]}"
    for s in "${STRATEGIES[@]}"; do
      for t in "${TEMPS[@]}"; do
        unload_flag=()
        if [[ "$s" == "$last_s" && "$t" == "$last_t" ]]; then
          unload_flag=(--unload-ollama)
          log "===== OSS $d | $s | T=$t (final cell -> unload) ====="
        else
          log "===== OSS $d | $s | T=$t ====="
        fi
        if run_logged logs/rq2_decoder_oss.log \
          emotion-cls classify-decoder --decoder "$d" --strategy "$s" --temperature "$t" \
          "${unload_flag[@]}"
        then
          classify_error_rate_ok "$d" "$s" "$t" || {
            msg="OSS $d | $s | T=$t has a >50% sentence error rate, continuing"
            log "WARN: $msg"; echo "$msg" >>"$FAILURES_LOG"
          }
        fi
      done
    done
    log "===== OSS model lifecycle done: $d ====="
  done
  log "RQ2 OSS DONE"
else
  log "SKIP RQ2 OSS"
fi

############################################
phase "Wait for background API / generation"
############################################
if [[ -n "$API_PID" ]]; then
  log "Waiting for RQ2 API PID=$API_PID"
  wait "$API_PID" || log "WARN: RQ2 API exited non-zero (see logs/rq2_decoder_api.log)"
fi
if [[ -n "$GEN_PID" ]]; then
  log "Waiting for RQ4 generate PID=$GEN_PID"
  wait "$GEN_PID" || log "WARN: RQ4 generate exited non-zero (see logs/rq4_generate_parity.log)"
fi

############################################
phase "4) RQ4 rank + sync synthetics + build CSV"
############################################
sync_generated_into_datasets
run_logged logs/rq4_rank.log emotion-cls rank-augmentation || log "WARN: rank-augmentation failed"

GENAI="$(choose_genai_for_build)"
log "Building synthetic multilabel CSV with --genai $GENAI -> $SYNTH_CSV"
run_logged logs/rq4_build_synth_csv.log \
  python scripts/build_synthetic_multilabel.py \
    --strategy "$STRATEGY" \
    --genai "$GENAI" \
    --n-per-emotion "$N_PER_EMOTION" \
    --out "$SYNTH_CSV"

############################################
phase "5) RQ3 imbalance on best RQ1 encoder (GPU)"
############################################
if [[ "$SKIP_RQ3" != "1" ]]; then
  if [[ -z "$BEST_ENC" ]]; then
    log "Auto-picking BEST_ENC from RQ1 multilabel summaries..."
    BEST_ENC="$(pick_best_encoder)" || BEST_ENC=""
  fi
  if [[ -z "$BEST_ENC" ]]; then
    msg="RQ3 skipped: no BEST_ENC (no encoder finished RQ1 multilabel — check logs/rq1_multilabel_all.log)"
    log "WARN: $msg"; echo "$msg" >>"$FAILURES_LOG"
  else
  log "BEST_ENC=$BEST_ENC"

  for imb in "${IMBALANCE_LOSSES[@]}"; do
    log "===== LOSS $imb ====="
    run_logged logs/rq3_imbalance_all.log \
      emotion-cls train-encoder --encoder "$BEST_ENC" --head multilabel --imbalance "$imb"
  done

  log "===== UNDERSAMPLE SWEEP ====="
  run_logged logs/rq3_imbalance_all.log \
    emotion-cls sweep-undersample --encoder "$BEST_ENC" --cutoffs "$CUTOFFS"

  for n in "${AUG_NS[@]}"; do
    log "===== GENAI_AUG n=$n ====="
    run_logged logs/rq3_imbalance_all.log \
      emotion-cls train-encoder --encoder "$BEST_ENC" --head multilabel \
        --imbalance genai_aug --aug-inject-n "$n" --synthetic-ml-path "$SYNTH_CSV"
  done

  run_logged logs/rq3_imbalance_all.log \
    emotion-cls train-encoder --encoder "$BEST_ENC" --head multilabel \
      --imbalance undersample,bce_pos_weight --undersample-cutoff 150

  run_logged logs/rq3_imbalance_all.log \
    emotion-cls train-encoder --encoder "$BEST_ENC" --head multilabel \
      --imbalance genai_aug,bce_pos_weight --aug-inject-n 100 --synthetic-ml-path "$SYNTH_CSV"

  log "RQ3 DONE"
  fi
else
  log "SKIP RQ3"
fi

phase "ALL EXPERIMENTS FINISHED"
log "Disk:"; df -h . | tail -1
log "Outputs:"; du -sh outputs .cache/huggingface 2>/dev/null || true
n_fail=$(wc -l <"$FAILURES_LOG" | tr -d ' ')
if [[ "$n_fail" != "0" ]]; then
  log "WARNING: $n_fail command(s) failed or had a high error rate during this run:"
  cat "$FAILURES_LOG"
  log "Re-running this script will resume/retry unfinished work; see $FAILURES_LOG for the full list."
else
  log "No failures recorded."
fi
log "Master log complete."
