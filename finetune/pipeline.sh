#!/usr/bin/env bash
# End-to-end driver: dataset -> jsonl -> smoke -> precompute -> train.
# Every stage is skippable; everything is env-overridable.
#
#   PROFILE=trial  bash finetune/pipeline.sh            # quick check run
#   PROFILE=sota   bash finetune/pipeline.sh            # full corpus
#   SKIP_FETCH=1 SKIP_PRECOMPUTE=1 ...                  # re-enter mid-pipeline
set -euo pipefail

PACK_DIR=${PACK_DIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_DIR=${DATA_DIR:-$PWD/data}
CKPT_DIR=${CKPT_DIR:-$PWD/ckpts}
AUK_REPO=${AUK_REPO:-$PWD/AuK}

PROFILE=${PROFILE:-trial}
corpus=${corpus:-$DATA_DIR/${PROFILE}_corpus}
train_jsonl=${train_jsonl:-$DATA_DIR/train.jsonl}
val_jsonl=${val_jsonl:-$DATA_DIR/val.jsonl}
cache_root=${cache_root:-$DATA_DIR/cache}
config=${config:-$PACK_DIR/config_flash_ru.yaml}
init_ckpt=${init_ckpt:-$CKPT_DIR/AuK-Flash/auk_flash.safetensors}
extra_fetch=${extra_fetch:-}

SKIP_FETCH=${SKIP_FETCH:-0}
SKIP_PREPARE=${SKIP_PREPARE:-0}
SKIP_PRECOMPUTE=${SKIP_PRECOMPUTE:-0}
SKIP_SMOKE=${SKIP_SMOKE:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}

echo "== AuK-Flash ru pipeline | profile=$PROFILE =="

if [[ $SKIP_FETCH != 1 ]]; then
  echo "-- fetch ($PROFILE) -> $corpus"
  python3 "$PACK_DIR/fetch_dataset.py" --profile "$PROFILE" --out "$corpus" $extra_fetch
fi

if [[ $SKIP_PREPARE != 1 ]]; then
  echo "-- prepare_jsonl -> $train_jsonl / $val_jsonl"
  python3 "$PACK_DIR/prepare_jsonl.py" --meta "$corpus/meta.tsv" --clips "$corpus" \
      --out "$train_jsonl" --val-out "$val_jsonl" --val-by-speaker
fi

if [[ $SKIP_SMOKE != 1 ]]; then
  echo "-- smoke: jsonl + layout"
  python3 "$PACK_DIR/smoke_test.py" --jsonl "$train_jsonl" "$val_jsonl" \
      --layout --ckpt-dir "$CKPT_DIR/AuK-Flash" --qwen-dir "$CKPT_DIR/Qwen2.5-Omni-3B" \
      --auk-repo "$AUK_REPO"
fi

if [[ $SKIP_PRECOMPUTE != 1 ]]; then
  echo "-- precompute -> $cache_root/{train,val} (runs thinker+VAE once)"
  for split in train val; do
    j=$train_jsonl; [[ $split == val ]] && j=$val_jsonl
    [[ -f $j ]] || continue
    PYTHONPATH="$AUK_REPO/src" python3 "$PACK_DIR/precompute_cond.py" \
        --jsonl "$j" --out "$cache_root/$split" \
        --config "$config" --ckpt "$init_ckpt" --auk-repo "$AUK_REPO"
  done
  python3 "$PACK_DIR/smoke_test.py" --cache "$cache_root/train" --jsonl "$train_jsonl" \
      --auk-repo "$AUK_REPO"
fi

if [[ $SKIP_TRAIN != 1 ]]; then
  echo "-- train"
  cond_cache=$cache_root train_jsonl=$train_jsonl val_jsonl=$val_jsonl \
    config=$config init_ckpt=$init_ckpt AUK_REPO=$AUK_REPO \
    bash "$PACK_DIR/train.sh"
fi

echo "== done =="
