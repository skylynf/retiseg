#!/usr/bin/env bash
# Stages of the first server batch, in order. Run from anywhere:
#
#   bash bench/scripts/server_run.sh env        conda envs (cluster_env.sh)
#   bash bench/scripts/server_run.sh check      data, weights, GPUs, test suite
#   bash bench/scripts/server_run.sh cache      1440 canvas caches
#   bash bench/scripts/server_run.sh smoke      B1 and B-std smoke, 20 steps each
#   bash bench/scripts/server_run.sh resmoke M2MRF,H2Former   smoke of the named models only
#   bash bench/scripts/server_run.sh probe      B-std budget probe, 4 jobs
#   bash bench/scripts/server_run.sh e1r        M2MRF author retrain, effective batch 4
#   bash bench/scripts/server_run.sh b1-seed0   B1 seed 0, train and validation
#   bash bench/scripts/server_run.sh bs-seed0   B-std seed 0, after the budget is fixed
#   bash bench/scripts/server_run.sh bs-score0 | b1-score0   test maps and metrics for seed 0
#   bash bench/scripts/server_run.sh bs-rest    B-std seeds 1 and 2
#   bash bench/scripts/server_run.sh status | curves [GLOB]
#
# Long stages start in the background with nohup, so the SSH session can
# close. Each stage writes runs/logs/<time>_<stage>.log and one line in
# runs/logs/stages.tsv: time, stage, git commit, tree state, log, pid.
# Each job also writes its own runs/<run>/console.log.
# Launcher stages take only free cards; RETISEG_GPUS="4 5 6 7" restricts them
# further, e.g. while the probe holds 0-3 and E1r holds 4.
# Formal stages refuse a working tree with uncommitted changes under bench/,
# because environment.json records only the commit. RETISEG_ALLOW_DIRTY=1
# overrides that and the override is written in stages.tsv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/runs/logs"
mkdir -p "$LOGS"
BS_JOBS=bench/configs/bs_jobs.yaml

pick_tool() {
  local tool
  for tool in micromamba mamba conda; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf '%s\n' "$tool"
      return 0
    fi
  done
  echo "No micromamba, mamba, or conda on PATH." >&2
  return 1
}

py() {
  local env_name="$1"
  shift
  "$(pick_tool)" run -n "$env_name" --no-capture-output python "$@"
}

tree_state() {
  if [ -n "$(git status --porcelain -- bench)" ]; then
    echo dirty
  else
    echo clean
  fi
}

require_clean() {
  if [ "$(tree_state)" = dirty ] && [ "${RETISEG_ALLOW_DIRTY:-0}" != 1 ]; then
    git status --short -- bench >&2
    echo "bench/ has uncommitted changes. Commit first, or set RETISEG_ALLOW_DIRTY=1." >&2
    exit 1
  fi
}

record() {
  local stage="$1" log="$2" pid="$3"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$stage" "$(git rev-parse --short HEAD)" \
    "$(tree_state)$([ "${RETISEG_ALLOW_DIRTY:-0}" = 1 ] && echo '+allowed')" "$log" "$pid" >> "$LOGS/stages.tsv"
}

background() {
  local stage="$1"
  shift
  local log="$LOGS/$(date +%Y%m%d_%H%M%S)_${stage}.log"
  nohup setsid "$@" > "$log" 2>&1 < /dev/null &
  local pid=$!
  record "$stage" "$log" "$pid"
  echo "$stage started, pid $pid"
  echo "  log: tail -f $log"
}

launch() {
  local stage="$1"
  shift
  background "$stage" "$(pick_tool)" run -n retiseg --no-capture-output python bench/scripts/launch_b1.py "$@"
}

stage_env() {
  local log="$LOGS/$(date +%Y%m%d_%H%M%S)_env.log"
  bash bench/scripts/cluster_env.sh 2>&1 | tee "$log"
  record env "$log" -
}

stage_check() {
  local missing=0 path
  for path in dataset/prepared/IDRiD dataset/prepared/DDR "dataset/iDRID/A. Segmentation" \
    pretrained/swin_tiny_patch4_window7_224.pth runs/E1_m2mrf/hrnetv2_w48-d2186c55.pth; do
    if [ -e "$path" ]; then
      echo "ok       $path"
    else
      echo "MISSING  $path"
      missing=1
    fi
  done
  if [ ! -e pretrained/hrnetv2_w48-d2186c55.pth ] && [ -e runs/E1_m2mrf/hrnetv2_w48-d2186c55.pth ]; then
    cp runs/E1_m2mrf/hrnetv2_w48-d2186c55.pth pretrained/
    echo "copied   pretrained/hrnetv2_w48-d2186c55.pth"
  fi
  [ "$missing" = 0 ] || { echo "copy the missing paths from the workstation first" >&2; exit 1; }
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  # One download of the torchvision ImageNet weights, so eight jobs do not race for it.
  py retiseg -c 'from torchvision.models import resnet101, ResNet101_Weights, resnet34, ResNet34_Weights
resnet101(weights=ResNet101_Weights.IMAGENET1K_V1); resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
print("torchvision ImageNet weights cached")'
  py retiseg -m pytest bench/tests -q -p no:cacheprovider
  py retiseg-hacdr -m pytest bench/tests/test_hacdr_recipe.py -q -p no:cacheprovider
}

stage_cache() {
  local dataset
  for dataset in IDRiD DDR; do
    py retiseg -m bench.data.canvas_cache --data "dataset/prepared/$dataset" --diameter 1440
  done
}

stage_probe() {
  require_clean
  local gpus=(${PROBE_GPUS:-0 1 2 3})
  local cells=(bs_probe_unet_idrid_seed0 bs_probe_unet_ddr_seed0 bs_probe_m2mrf_idrid_seed0 bs_probe_m2mrf_ddr_seed0)
  local i cell script run
  for i in "${!cells[@]}"; do
    cell="${cells[$i]}"
    script=bench/train.py
    [[ "$cell" == *m2mrf* ]] && script=bench/scripts/train_m2mrf_b1.py
    run="runs/BSprobe_${cell#bs_probe_}"
    mkdir -p "$run"
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" background "probe_${cell#bs_probe_}" bash -c \
      "$(pick_tool) run -n retiseg --no-capture-output python $script --config bench/configs/cells/$cell.yaml --run-dir $run --diagnostic --resume >> $run/console.log 2>&1"
  done
  echo "when they finish: bash bench/scripts/server_run.sh curves 'BSprobe_*'"
}

stage_e1r() {
  require_clean
  local run=runs/E1r_m2mrf_idrid_seed0_bs4_a100
  mkdir -p "$run"
  local resume=()
  if [ -f "$run/last.pt" ]; then
    resume=(--resume "$run/last.pt")
  fi
  CUDA_VISIBLE_DEVICES="${E1R_GPU:-4}" background e1r "$(pick_tool)" run -n retiseg --no-capture-output \
    python bench/train_m2mrf.py --run-dir "$run" ${resume[@]+"${resume[@]}"}
}

stage="${1:-}"
case "$stage" in
  env) stage_env ;;
  check) stage_check ;;
  cache) stage_cache ;;
  smoke)
    tool="$(pick_tool)"
    background smoke bash -c "$tool run -n retiseg --no-capture-output python bench/scripts/launch_b1.py --submit --smoke; \
$tool run -n retiseg --no-capture-output python bench/scripts/launch_b1.py --jobs $BS_JOBS --submit --smoke" ;;
  resmoke)
    [ -n "${2:-}" ] || { echo "usage: server_run.sh resmoke MODEL[,MODEL]" >&2; exit 1; }
    tool="$(pick_tool)"
    background "resmoke_${2//,/_}" bash -c "$tool run -n retiseg --no-capture-output python bench/scripts/launch_b1.py --submit --smoke --only $2; \
$tool run -n retiseg --no-capture-output python bench/scripts/launch_b1.py --jobs $BS_JOBS --submit --smoke --only $2" ;;
  probe) stage_probe ;;
  e1r) stage_e1r ;;
  b1-seed0) require_clean; launch b1_seed0 --submit --seed0 ;;
  b1-score0) require_clean; launch b1_score0 --submit --score-seed0 ;;
  bs-seed0) require_clean; launch bs_seed0 --jobs "$BS_JOBS" --submit --seed0 ;;
  bs-score0) require_clean; launch bs_score0 --jobs "$BS_JOBS" --submit --score-seed0 ;;
  bs-rest) require_clean; launch bs_rest --jobs "$BS_JOBS" --submit --rest ;;
  status) py retiseg bench/scripts/run_status.py status --glob "${2:-*}" ;;
  curves) py retiseg bench/scripts/run_status.py curves --glob "${2:-*}" ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1 ;;
esac
