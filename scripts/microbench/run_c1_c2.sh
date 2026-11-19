#!/usr/bin/env bash
# C1 (flow dense vs packed) and C2 (HiFT fold / layout / half) microbenchmarks on held A6000s, then the ASR quality
# pass and both reports. Runs on the compute node inside the job (ssh or srun --jobid); nothing here starts a job.
#
#   bash scripts/microbench/run_c1_c2.sh --gpu-c1 0 --gpu-c2 1 --out <PROJECT_DATA_ROOT>/experiments/<id> [--optional-arms]
#
# C1 arms run one after another on --gpu-c1 (one process each); C2 runs on --gpu-c2 at the same time; the ASR server
# (Qwen3-ASR-1.7B, cached) starts on --gpu-c2 after C2 ends. The stock sglang-omni 520b49ee checkout is used.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LR="${LEARNING_ROOT:-$HOME/workspace/sglang-omni-learning}"
GPU_C1=0 GPU_C2=1 OUT="" OPTIONAL=0 ASR_PORT=18090
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-c1) GPU_C1="$2"; shift 2 ;;
    --gpu-c2) GPU_C2="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --optional-arms) OPTIONAL=1; shift ;;
    --asr-port) ASR_PORT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 2; }
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$LR/scripts/env-unicorn.sh"
PY="$LR/.venv/bin/python"
MODEL="$LR/models/MiniCPM-o-4_5"
ASSETS="${ASSETS:-$HOME/workspace/omni-serving-research/results/assets/synthetic-v1}"
REPLAY="$REPO/research/next_steps/results/c1_replay_batches.json"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
log "repo $(git -C "$REPO" rev-parse HEAD) sglang-omni $(git -C "$LR/repos/sglang-omni" rev-parse HEAD) gpus c1=$GPU_C1 c2=$GPU_C2"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv >> "$OUT/run.log"

c1_arms=(REF D P0 P1 D_ng)
[[ "$OPTIONAL" == 1 ]] && c1_arms+=(P2 P1h)
(
  for arm in "${c1_arms[@]}"; do
    extra=()
    case "$arm" in D_ng|P2|P1h) extra=(--primary-only) ;; esac
    log "C1 arm $arm start"
    CUDA_VISIBLE_DEVICES="$GPU_C1" "$PY" "$REPO/scripts/microbench/c1_flow_packed.py" run --arm "$arm" --model-path "$MODEL" \
      --replay "$REPLAY" --assets "$ASSETS" --out "$OUT/c1/$arm" --fresh-inductor-cache "${extra[@]}" \
      > "$OUT/c1_$arm.log" 2>&1
    log "C1 arm $arm exit $?"
  done
) &
C1_PID=$!
log "C2 start"
CUDA_VISIBLE_DEVICES="$GPU_C2" "$PY" "$REPO/scripts/microbench/c2_hift.py" run --arms A,B,C,D --model-path "$MODEL" \
  --replay "$REPLAY" --assets "$ASSETS" --out "$OUT/c2" > "$OUT/c2.log" 2>&1
log "C2 exit $?"
wait "$C1_PID"
log "C1 all arms finished"

# ASR server for the quality gates (its own session, so the whole process group can be stopped by its id)
# the configuration verified on an A6000 (sglang-omni-learning configs/unicorn/asr-learning.yaml), with the 1.7B model
cat > "$OUT/asr_config.yaml" <<EOF
config_cls: Qwen3ASRPipelineConfig
name: ppv-quality-asr
model_path: $LR/models/Qwen3-ASR-1.7B
stages:
  asr:
    factory:
      dtype: bfloat16
      enable_async_decode: false
      enable_encoder_cuda_graph: false
      pre_lm_cache_size_bytes: 67108864
      pre_lm_cache_max_entries: 8
      pre_lm_max_batch_size: 1
      request_build_max_workers: 1
      request_build_max_pending: 2
      prefill_coalesce_requests: 0
      prefill_coalesce_wait_ms: 1
    engine:
      max_running_requests: 1
      mem_fraction_static: 0.15
      max_total_tokens: 8192
      context_length: 8192
      enable_torch_compile: false
      disable_cuda_graph: true
      attention_backend: triton
      disable_custom_all_reduce: true
EOF
log "ASR server start"
( cd "$LR/repos/sglang-omni" && CUDA_VISIBLE_DEVICES="$GPU_C2" exec setsid "$PY" -m sglang_omni.cli serve --config "$OUT/asr_config.yaml" \
    --model-name Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port "$ASR_PORT" > "$OUT/asr_server.log" 2>&1 ) &
ASR_PID=$!
ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$ASR_PORT/health" | grep -q healthy; then ready=1; break; fi
  sleep 5
done
asr_args=()
if [[ "$ready" == 1 ]]; then
  log "ASR server healthy"
  asr_args=(--asr-url "http://127.0.0.1:$ASR_PORT" --asr-model Qwen/Qwen3-ASR-1.7B)
else
  log "ASR server not healthy after 15 min; quality gates will be reported without ASR (not passed)"
fi
"$PY" "$REPO/scripts/microbench/quality_eval.py" c1 --root "$OUT/c1" "${asr_args[@]}" --out "$OUT/c1_quality.json" >> "$OUT/run.log" 2>&1
"$PY" "$REPO/scripts/microbench/quality_eval.py" c2 --root "$OUT/c2" "${asr_args[@]}" --out "$OUT/c2_quality.json" >> "$OUT/run.log" 2>&1
# the server is its own process group (setsid); stop exactly that group, never by name pattern
kill -TERM -- "-$ASR_PID" 2>/dev/null || kill -TERM "$ASR_PID" 2>/dev/null
wait "$ASR_PID" 2>/dev/null
sleep 5
left="$(ps -o pid= -g "$ASR_PID" 2>/dev/null | tr -s ' \n' ' ')"
log "ASR server stopped${left:+; still running in its group: $left}"
"$PY" "$REPO/scripts/microbench/c1_flow_packed.py" report --in "$OUT/c1" --quality "$OUT/c1_quality.json" --out "$OUT/c1_microbench.json" >> "$OUT/run.log" 2>&1
"$PY" "$REPO/scripts/microbench/c2_hift.py" report --in "$OUT/c2" --quality "$OUT/c2_quality.json" --c1-report "$OUT/c1_microbench.json" \
  --out "$OUT/c2_microbench.json" >> "$OUT/run.log" 2>&1
log "done; copy c1_microbench.json and c2_microbench.json (no raw audio) to research/next_steps/results/"
