#!/usr/bin/env bash
# Multi-GPU watch for AriaPool TSC workers (mirrors run.sh layout).
#
# Layouts (same as run.sh):
#   exactly 2x 16 GB          -> dual: one stack in wtp2 (aria-tsc-wtp2)
#   otherwise (16 GB+ cards)  -> one stack per GPU: w0, w1, … (aria-tsc-wN)
#
# Usage:
#   ./watch-vllm-multi.sh              # one check (for cron)
#   ./watch-vllm-multi.sh --loop       # keep checking every INTERVAL seconds
#   INTERVAL=60 ./watch-vllm-multi.sh --loop
#   RESTART_ALL=1 ./watch-vllm-multi.sh   # on any failure, re-run ./run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCK="$ROOT/.watch-vllm-multi.lock"
LOG="$ROOT/watch-vllm-multi.log"
INTERVAL="${INTERVAL:-120}"
COOLDOWN="${COOLDOWN:-300}"   # seconds after a restart before another restart is allowed
STATE_DIR="$ROOT/.watch-vllm-multi.state"
MIN_16=14500
MIN_24=21500
RESTART_ALL="${RESTART_ALL:-0}"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"
}

# ── GPU / layout detection (aligned with run.sh) ───────────────────────────
detect_layout() {
  mapfile -t VRAM < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  NGPU=${#VRAM[@]}
  [ "$NGPU" -gt 0 ] || { log "ERROR: no NVIDIA GPU detected"; return 1; }

  n16=0; n24=0
  GPU_IDS=()
  GPU_ENGINES=()
  for g in "${!VRAM[@]}"; do
    mb="${VRAM[$g]// /}"
    if   [ "$mb" -ge "$MIN_24" ]; then
      n24=$((n24+1))
      GPU_IDS+=("$g")
      GPU_ENGINES+=("vllm-bf16")
    elif [ "$mb" -ge "$MIN_16" ]; then
      n16=$((n16+1))
      GPU_IDS+=("$g")
      GPU_ENGINES+=("llama")
    fi
  done

  MODE="per-gpu"
  if [ "$NGPU" -eq 2 ] && [ "$n16" -eq 2 ] && [ "$n24" -eq 0 ]; then
    MODE="dual"
  fi
}

# True if GPU $1 has a VLLM::EngineCore compute process.
vllm_on_gpu() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
    | grep -q 'VLLM::EngineCore'
}

# Count how many of the given GPU indices currently host EngineCore.
vllm_gpu_count() {
  local n=0 g
  for g in "$@"; do
    vllm_on_gpu "$g" && n=$((n+1))
  done
  echo "$n"
}

# Llama / any non-vLLM worker: treat "backend + proxy containers up" as healthy.
# (llama.cpp process names vary; container presence is the reliable signal.)
containers_match() {
  local pattern="$1"
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qE "$pattern"
}

in_cooldown() {
  local key="$1"
  local f="$STATE_DIR/$key"
  [[ -f "$f" ]] || return 1
  local last now
  last="$(cat "$f" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  (( now - last < COOLDOWN ))
}

mark_restart() {
  mkdir -p "$STATE_DIR"
  date +%s >"$STATE_DIR/$1"
}

ensure_supervisor_file() {
  local inst="$1"
  if [[ -d "$inst/vllm_supervisor.py" ]]; then
    log "WARN: $inst/vllm_supervisor.py is a directory — removing"
    rm -rf "$inst/vllm_supervisor.py"
  fi
  if [[ -f "$ROOT/vllm_supervisor.py" && ! -f "$inst/vllm_supervisor.py" ]]; then
    cp -f "$ROOT/vllm_supervisor.py" "$inst/vllm_supervisor.py"
    log "copied vllm_supervisor.py into $(basename "$inst")"
  fi
}

restart_instance() {
  local inst_dir="$1"   # e.g. w0 or wtp2
  local project="$2"    # e.g. aria-tsc-w0
  local profile="$3"    # e.g. vllm-bf16 | llama | vllm-tp2
  local key="$4"

  log "restarting $project (profile=$profile)"
  if [[ -d "$ROOT/$inst_dir" ]]; then
    ( cd "$ROOT/$inst_dir" && docker compose -p "$project" --profile '*' down ) || true
  else
    log "WARN: $ROOT/$inst_dir missing — falling back to ./run.sh"
    ( cd "$ROOT" && ./run.sh )
    mark_restart "$key"
    return
  fi
  ensure_supervisor_file "$ROOT/$inst_dir"
  ( cd "$ROOT/$inst_dir" && docker compose -p "$project" --profile "$profile" up -d )
  mark_restart "$key"
  log "restart finished: $project"
}

restart_via_runsh() {
  local key="$1"
  log "RESTART_ALL: bringing workers back via ./run.sh"
  # Stop known stacks so run.sh can recreate cleanly.
  if [[ -d "$ROOT/wtp2" ]]; then
    ( cd "$ROOT/wtp2" && docker compose -p aria-tsc-wtp2 --profile '*' down ) || true
  fi
  local d
  for d in "$ROOT"/w[0-9]*; do
    [[ -d "$d" ]] || continue
    local name base
    base="$(basename "$d")"
    ( cd "$d" && docker compose -p "aria-tsc-$base" --profile '*' down ) || true
  done
  ( cd "$ROOT" && ./run.sh )
  mark_restart "$key"
  log "restart finished: run.sh"
}

maybe_restart() {
  local key="$1"
  shift
  if in_cooldown "$key"; then
    log "WARN: [$key] unhealthy but still in cooldown (${COOLDOWN}s) — skip"
    return 0
  fi
  if [[ "$RESTART_ALL" == "1" ]]; then
    restart_via_runsh "$key"
  else
    "$@"
  fi
}

# ── Health checks ──────────────────────────────────────────────────────────
check_dual() {
  local ok=1
  local eng_count
  eng_count="$(vllm_gpu_count 0 1)"
  if [[ "$eng_count" -lt 1 ]]; then
    log "WARN: [dual] no VLLM::EngineCore on GPU0/1"
    ok=0
  elif [[ "$eng_count" -lt 2 ]]; then
    log "WARN: [dual] EngineCore on $eng_count/2 GPUs (expected both in tp2)"
    ok=0
  fi

  if ! containers_match 'aria-tsc-wtp2-vllm-backend'; then
    log "WARN: [dual] vllm-backend container not up (aria-tsc-wtp2)"
    ok=0
  fi
  if ! containers_match 'aria-tsc-wtp2-miner-proxy'; then
    log "WARN: [dual] miner-proxy container not up (aria-tsc-wtp2)"
    ok=0
  fi

  if [[ "$ok" -eq 1 ]]; then
    log "OK: [dual] EngineCore on both GPUs, wtp2 stack up"
    return 0
  fi

  maybe_restart dual \
    restart_instance wtp2 aria-tsc-wtp2 vllm-tp2 dual
}

check_per_gpu() {
  local i g engine inst project backend_pat proxy_pat ok any_bad=0
  if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    log "ERROR: no GPU ≥16 GB to monitor"
    return 1
  fi

  for i in "${!GPU_IDS[@]}"; do
    g="${GPU_IDS[$i]}"
    engine="${GPU_ENGINES[$i]}"
    inst="w$g"
    project="aria-tsc-w$g"
    ok=1

    if [[ "$engine" == "vllm-bf16" ]]; then
      backend_pat="${project}-vllm-backend"
      proxy_pat="${project}-miner-proxy"
      if ! vllm_on_gpu "$g"; then
        log "WARN: [GPU$g/$engine] no VLLM::EngineCore"
        ok=0
      fi
    else
      backend_pat="${project}-llama-backend"
      proxy_pat="${project}-miner-proxy"
      # llama: no EngineCore — require backend container + some compute on that GPU
      if ! containers_match "$backend_pat"; then
        log "WARN: [GPU$g/$engine] llama-backend container not up"
        ok=0
      fi
      local apps
      apps="$(nvidia-smi -i "$g" --query-compute-apps=process_name --format=csv,noheader 2>/dev/null || true)"
      if [[ -z "${apps// /}" ]]; then
        log "WARN: [GPU$g/$engine] no compute apps on GPU (idle / crashed?)"
        ok=0
      fi
    fi

    if ! containers_match "$backend_pat"; then
      log "WARN: [GPU$g/$engine] backend container not up ($backend_pat)"
      ok=0
    fi
    if ! containers_match "$proxy_pat"; then
      log "WARN: [GPU$g/$engine] miner-proxy container not up ($proxy_pat)"
      ok=0
    fi

    if [[ "$ok" -eq 1 ]]; then
      log "OK: [GPU$g/$engine] healthy"
    else
      any_bad=1
      maybe_restart "w$g" \
        restart_instance "$inst" "$project" "$engine" "w$g"
    fi
  done

  return 0
}

check_once() {
  exec 9>"$LOCK"
  if ! flock -n 9; then
    log "another watch-vllm-multi instance holds the lock — skip"
    return 0
  fi

  detect_layout || return 1
  log "layout=$MODE gpus=${NGPU} watching=${#GPU_IDS[@]} card(s)"

  if [[ "$MODE" == "dual" ]]; then
    check_dual
  else
    check_per_gpu
  fi
}

main() {
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
  command -v docker >/dev/null || { echo "docker not found"; exit 1; }
  command -v flock >/dev/null || { echo "flock not found (util-linux)"; exit 1; }
  [[ -x "$ROOT/run.sh" ]] || chmod +x "$ROOT/run.sh"
  mkdir -p "$STATE_DIR"

  if [[ "${1:-}" == "--loop" ]]; then
    log "loop mode: interval=${INTERVAL}s cooldown=${COOLDOWN}s restart_all=${RESTART_ALL}"
    while true; do
      check_once || true
      sleep "$INTERVAL"
    done
  else
    check_once
  fi
}

main "$@"
