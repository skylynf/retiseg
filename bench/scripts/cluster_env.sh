#!/usr/bin/env bash
# Repeatable B1 environments. Safe to run again: existing envs are kept and
# packages are installed only when the version check fails.
# The repo root is the directory that contains this script. No home directory
# is written into the file. Pretrained weights are printed, not downloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# CUDA 12.4 wheels satisfy the PyTorch 2.x + CUDA 12.x requirement.
# Override the index with RETISEG_TORCH_INDEX if the cluster mirror differs.
TORCH_INDEX="${RETISEG_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

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

TOOL="$(pick_tool)"

env_exists() {
  "$TOOL" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"
}

ensure_env() {
  local name="$1"
  local python_version="$2"
  if env_exists "$name"; then
    echo "env ${name} already exists"
  else
    "$TOOL" create -y -n "$name" "python=${python_version}"
  fi
}

write_repo_pth() {
  local name="$1"
  local site
  site="$("$TOOL" run -n "$name" python -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "$ROOT" > "${site}/retiseg.pth"
}

run_python() {
  local name="$1"
  shift
  "$TOOL" run -n "$name" python "$@"
}

retiseg_ok() {
  run_python retiseg -c 'import torch, torchvision, bench
version = torch.__version__.split("+")[0]
if int(version.split(".")[0]) < 2:
    raise SystemExit("torch %s is not 2.x" % torch.__version__)
cuda = torch.version.cuda or ""
if cuda.split(".")[0] != "12":
    raise SystemExit("torch cuda %s is not 12.x" % cuda)
print("retiseg", torch.__version__, "cuda", cuda)'
}

install_retiseg() {
  ensure_env retiseg 3.11
  write_repo_pth retiseg
  if retiseg_ok; then
    echo "retiseg already imports torch 2.x, torchvision, and bench"
    return 0
  fi
  run_python retiseg -m pip install --upgrade pip
  run_python retiseg -m pip install torch torchvision --index-url "$TORCH_INDEX"
  run_python retiseg -m pip install -r "$ROOT/bench/requirements-eval.txt"
  write_repo_pth retiseg
  retiseg_ok
}

# M2MRF README: pytorch 1.6.0, mmcv-full 1.2.0, local mmsegmentation 0.8.0, CUDA 10.2, Python 3.7.
# HACDR-Net README: PyTorch 1.10.0+, mmcv-full 1.6.2, mmsegmentation 0.30.0, Python 3.8, CUDA 10.1+.
# Those pins cannot share one env, so retiseg-mmcv is not created.
m2mrf_ok() {
  run_python retiseg-m2mrf -c 'import mmcv, mmseg, torch
assert torch.__version__.startswith("1.6.0"), torch.__version__
assert mmcv.__version__ == "1.2.0", mmcv.__version__
assert mmseg.__version__ == "0.8.0", mmseg.__version__
print("retiseg-m2mrf", torch.__version__, "mmcv", mmcv.__version__, "mmseg", mmseg.__version__)'
}

install_m2mrf() {
  ensure_env retiseg-m2mrf 3.7
  write_repo_pth retiseg-m2mrf
  if m2mrf_ok; then
    echo "retiseg-m2mrf already matches the M2MRF README pins"
    return 0
  fi
  "$TOOL" install -y -n retiseg-m2mrf pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch
  # README also passed -i https://pypi.douban.com/simple/. That mirror is not required.
  run_python retiseg-m2mrf -m pip install mmcv-full==1.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
  run_python retiseg-m2mrf -m pip install opencv-python scipy tensorboard tensorboardX terminaltables matplotlib scikit-learn
  run_python retiseg-m2mrf -m pip install -e "$ROOT/official_code/M2MRF"
  write_repo_pth retiseg-m2mrf
  m2mrf_ok
}

hacdr_ok() {
  run_python retiseg-hacdr -c 'import mmcv, mmseg, torch
version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
if version < (1, 10):
    raise SystemExit("torch %s is below 1.10.0" % torch.__version__)
assert mmcv.__version__ == "1.6.2", mmcv.__version__
assert mmseg.__version__ == "0.30.0", mmseg.__version__
print("retiseg-hacdr", torch.__version__, "mmcv", mmcv.__version__, "mmseg", mmseg.__version__)'
}

install_hacdr() {
  ensure_env retiseg-hacdr 3.8
  write_repo_pth retiseg-hacdr
  if hacdr_ok; then
    echo "retiseg-hacdr already matches the HACDR-Net README pins"
    return 0
  fi
  # README says PyTorch 1.10.0+ and CUDA 10.1+. mmcv-full 1.6.2 publishes a
  # torch 1.10.0 / CUDA 11.3 wheel, which meets that minimum.
  "$TOOL" install -y -n retiseg-hacdr pytorch=1.10.0 torchvision=0.11.0 cudatoolkit=11.3 -c pytorch
  run_python retiseg-hacdr -m pip install mmcv-full==1.6.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
  # The checkout is mmsegmentation 0.30.0 plus the HACDR modules. Editable
  # install uses that tree. A second copy from PyPI is not installed over it.
  run_python retiseg-hacdr -m pip install -e "$ROOT/official_code/HACDR-Net"
  run_python retiseg-hacdr -m pip install mmcv-full==1.6.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
  write_repo_pth retiseg-hacdr
  hacdr_ok
}

print_weights() {
  cat <<'EOF'
weights are not downloaded by this script.

U-Net: pretrained none.
FCT: pretrained none. The PyTorch entry draws Kaiming normal weights.

DeepLabv3+: card name ResNet101_Weights.IMAGENET1K_V1.
  torchvision fetches that enum on first use. This script does not.

HRNet: filename hrnetv2_w48_imagenet_pretrained_top1_21.pth
  path in the Cityscapes yaml: dataset/pretrained_models/hrnetv2_w48_imagenet_pretrained_top1_21.pth
  yaml: official_code/HRNet-Semantic-Segmentation/experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml

Swin-Unet: filename swin_tiny_patch4_window7_224.pth
  README section 1 says to put the pretrained Swin-T in pretrained_ckpt/
  config: official_code/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml
  PRETRAIN_CKPT: ./pretrained_ckpt/swin_tiny_patch4_window7_224.pth
  folder link in official_code/Swin-Unet/README.md (Google Drive). Not fetched here.

H2Former: filename resnet34.pth
  official_code/H2Former/README.md says to put the pretrained ImageNet model in the repo directory.
  official_code/H2Former/idrid_train.py loads ./resnet34.pth

M2MRF backbone key, not a local file: open-mmlab://msra/hrnetv2_w48
  official_code/M2MRF/configs/_base_/models/fcn_hr48.py
  README test filename: fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth
  Google Drive links for the released checkpoints are in official_code/M2MRF/README.md. Not fetched here.

HACDR-Net: pretrained=None and load_from=None. No weight filename is declared.
EOF
}

echo "M2MRF pins mmcv-full==1.2.0 and mmsegmentation 0.8.0."
echo "HACDR-Net pins mmcv-full==1.6.2 and mmsegmentation 0.30.0."
echo "The pins conflict, so the envs are retiseg-m2mrf and retiseg-hacdr. retiseg-mmcv is not created."

install_retiseg
install_m2mrf
install_hacdr
print_weights
