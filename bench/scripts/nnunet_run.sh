#!/usr/bin/env bash
# One nnU-Net v2 job: train fold 0 (this project's train/val), predict val and
# test with softmax, export to write_prob maps, evaluate with bench.eval.
#
#   CUDA_VISIBLE_DEVICES=4 bash bench/scripts/nnunet_run.sh IDRiD default
#   CUDA_VISIBLE_DEVICES=5 bash bench/scripts/nnunet_run.sh DDR resencm
#   RETISEG_NN_REP=rep1 CUDA_VISIBLE_DEVICES=4 bash bench/scripts/nnunet_run.sh IDRiD default   # repeat run
#
# Data come from bench/scripts/nnunet_bridge.py convert and splits; planning
# and preprocessing are nnUNetv2_plan_and_preprocess (default) and
# nnUNetPlannerResEncM. Run again after a crash: training continues from
# nnU-Net's checkpoint_latest.pth with --c, finished steps are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
dataset="$1"
plan="$2"
case "$dataset" in
  IDRiD) id=601 ;;
  DDR) id=602 ;;
  *) echo "dataset must be IDRiD or DDR" >&2; exit 1 ;;
esac
case "$plan" in
  default) plans=nnUNetPlans ;;
  resencm) plans=nnUNetResEncUNetMPlans ;;
  *) echo "plan must be default or resencm" >&2; exit 1 ;;
esac
lower="$(echo "$dataset" | tr '[:upper:]' '[:lower:]')"
bin=/home/qurr/miniconda3/envs/retiseg-nnunet/bin
export nnUNet_raw="$ROOT/dataset/nnunet/raw"
export nnUNet_preprocessed="$ROOT/dataset/nnunet/preprocessed"
# RETISEG_NN_REP=rep1 trains a repeat of the same plan (nnU-Net fixes no seed; docs/paper/16: three
# default runs). The repeat gets its own nnU-Net results root and run directory.
export nnUNet_results="$ROOT/runs/nnunet${RETISEG_NN_REP:+_$RETISEG_NN_REP}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-16}"
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK GROUP_RANK MASTER_ADDR MASTER_PORT
data="dataset/prepared/$dataset"
folder="Dataset${id}_${dataset}"
run="runs/NN_${plan}_${lower}_fold0${RETISEG_NN_REP:+_$RETISEG_NN_REP}"
fold_dir="$nnUNet_results/$folder/nnUNetTrainer__${plans}__2d/fold_0"
mkdir -p "$run"
trap 'echo "== exit $? $(date -Is)" >> "$run/console.log"' EXIT
{
  echo "== start nnunet $(date -Is) gpu ${CUDA_VISIBLE_DEVICES:-} host $(hostname) pid $$ commit $(git rev-parse --short HEAD)"
  if [ ! -f "$fold_dir/checkpoint_final.pth" ]; then
    resume=()
    [ -f "$fold_dir/checkpoint_latest.pth" ] && resume=(--c)
    "$bin/nnUNetv2_train" "$id" 2d 0 -p "$plans" ${resume[@]+"${resume[@]}"}
  fi
  for split in val test; do
    if [ ! -f "$run/.prob_${split}_done" ]; then
      out="$run/nnunet_$split"
      "$bin/nnUNetv2_predict" -i "$nnUNet_raw/$folder/images_$split" -o "$out" -d "$id" -c 2d -f 0 -p "$plans" \
        --save_probabilities -npp 4 -nps 4
      "$bin/python" bench/scripts/nnunet_bridge.py export --data "$data" --npz-dir "$out" --pred-dir "$run/prob_$split" --split "$split"
      touch "$run/.prob_${split}_done"
      rm -f "$out"/*.npz
    fi
  done
  "$bin/python" -m bench.eval.cli evaluate --data "$data" --split test --pred "$run/prob_test" \
    --val-pred "$run/prob_val" --out "$run/metrics_test"
} >> "$run/console.log" 2>&1
