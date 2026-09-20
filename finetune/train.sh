#!/usr/bin/env bash
# Single-node AuK-Flash fine-tune (flash_grid recipe, patch 0001 required).
# Everything is env-overridable; works on any box with a GPU + AuK clone.
#
#   AUK_REPO=/path/AuK bash finetune/train.sh
set -euo pipefail

# --- paths (edit or export) ---
AUK_REPO=${AUK_REPO:-$PWD/AuK}
DATA_DIR=${DATA_DIR:-$PWD/data}
CKPT_DIR=${CKPT_DIR:-$PWD/ckpts}
PACK_DIR=${PACK_DIR:-$(cd "$(dirname "$0")" && pwd)}

train_jsonl=${train_jsonl:-$DATA_DIR/train.jsonl}
val_jsonl=${val_jsonl:-$DATA_DIR/val.jsonl}
config=${config:-$PACK_DIR/config_flash_ru.yaml}
init_ckpt=${init_ckpt:-$CKPT_DIR/AuK-Flash/auk_flash.safetensors}
output_dir=${output_dir:-$PWD/runs/auk_flash_ru}
cond_cache=${cond_cache:-}   # root dir from precompute_cond.py (needs train/ + val/)

# --- resources / hyperparams (L40S/48GB defaults) ---
num_gpus=${num_gpus:-1}
learning_rate=${learning_rate:-2e-5}
max_grad_norm=${max_grad_norm:-1.0}
num_train_epochs=${num_train_epochs:-20}
gradient_accumulation_steps=${gradient_accumulation_steps:-4}
warmup_steps=${warmup_steps:-100}
frames_threshold=${frames_threshold:-2700}   # ~54s audio/batch; drop to ~1200 on 24GB
max_samples=${max_samples:-8}
ema_beta=${ema_beta:-0.9999}
save_per_updates=${save_per_updates:-500}
last_per_updates=${last_per_updates:-200}
val_per_updates=${val_per_updates:-200}
logging_steps=${logging_steps:-10}
dataloader_num_workers=${dataloader_num_workers:-4}
seed=${seed:-666}

# --- sanity ---
for f in "$train_jsonl" "$config" "$init_ckpt"; do
  [[ -f $f ]] || { echo "missing: $f" >&2; exit 1; }
done
mkdir -p "$output_dir"

cd "$AUK_REPO"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

args=(
  --train_jsonl "$train_jsonl"
  --config "$config"
  --init_ckpt "$init_ckpt"
  --output_dir "$output_dir"
  --learning_rate "$learning_rate"
  --max_grad_norm "$max_grad_norm"
  --num_train_epochs "$num_train_epochs"
  --gradient_accumulation_steps "$gradient_accumulation_steps"
  --warmup_steps "$warmup_steps"
  --frames_threshold "$frames_threshold"
  --max_samples "$max_samples"
  --ema_beta "$ema_beta"
  --save_per_updates "$save_per_updates"
  --last_per_updates "$last_per_updates"
  --val_per_updates "$val_per_updates"
  --logging_steps "$logging_steps"
  --dataloader_num_workers "$dataloader_num_workers"
  --seed "$seed"
)
[[ -f $val_jsonl ]] && args+=(--val_jsonl "$val_jsonl")
[[ -n $cond_cache ]] && args+=(--cond_cache "$cond_cache")

exec accelerate launch --num_processes "$num_gpus" --num_machines 1 \
  --mixed_precision bf16 -m auk.train.train "${args[@]}"
