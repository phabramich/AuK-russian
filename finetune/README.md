# AuK-Flash RU fine-tune pack

Environment-agnostic pipeline: rented GPU box (L40S/4090/A100), Modal,
notebook, or local. Trains **only the DiT** on the 4 distilled t-points —
Qwen thinker + VAE stay frozen, the 4-step Flash property is preserved.

## Files

| File | What |
|---|---|
| `patches/0001-flash-grid-training.patch` | `t_sampling: flash_grid` — t sampled only at the 4 distilled points |
| `patches/0002-cond-cache-and-speed.patch` | cached-conditioning mode (`--cond_cache`), val synthesis on Flash recipe, auto 8-bit AdamW if bitsandbytes installed |
| `config_flash_ru.yaml` | flash config: `t_sampling: flash_grid`, `audio/cond_drop_prob: 0` |
| `fetch_dataset.py` | HF corpora → clips + meta.tsv (`--profile trial` / `--profile sota`) |
| `prepare_jsonl.py` | meta.tsv → train/val JSONL, same-speaker ref pairs, `--val-by-speaker` |
| `precompute_cond.py` | frozen thinker+VAE → per-sample .pt cache (the big speedup) |
| `smoke_test.py` | JSONL + cache validation, tiny CPU model check |
| `export_ema.py` | `model_last.pt` → safetensors for inference/MLX |
| `pipeline.sh` | end-to-end driver: fetch → jsonl → smoke → precompute → train |
| `train.sh` | single-node launcher, all env-overridable |

## Quickstart (one command)

```bash
AUK_REPO=$PWD/AuK PROFILE=trial  bash finetune/pipeline.sh   # ~20h corpus
AUK_REPO=$PWD/AuK PROFILE=sota   bash finetune/pipeline.sh   # ~400+h corpus
# re-enter mid-pipeline: SKIP_FETCH=1 SKIP_PREPARE=1 ... env flags per stage
```

## Setup (once, on the train box)

```bash
git clone https://github.com/Tencent-Hunyuan/AuK.git && cd AuK
git checkout e3828bdac1712bbf7bcb3a2a01eec6fa5168fe32
git apply /path/to/finetune/patches/0001-flash-grid-training.patch
git apply /path/to/finetune/patches/0002-cond-cache-and-speed.patch
pip install -e ".[train]" datasets soundfile
# optional: pip install bitsandbytes flash-attn   (8-bit opt / faster attn)

# weights (~17GB)
mkdir -p ckpts/AuK-Flash ckpts/Qwen2.5-Omni-3B
cd ckpts/AuK-Flash
curl -LO https://huggingface.co/tencent/AuK-Flash/resolve/main/auk_flash.safetensors
curl -LO https://huggingface.co/tencent/AuK-Flash/resolve/main/vae.safetensors
cd ../Qwen2.5-Omni-3B
for i in 1 2; do   # shard 3 = talker/token2wav, not needed
  curl -LO https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/main/model-0000$i-of-00003.safetensors
done
for f in config.json generation_config.json preprocessor_config.json \
         tokenizer.json tokenizer_config.json vocab.json merges.txt \
         added_tokens.json special_tokens_map.json chat_template.json \
         model.safetensors.index.json; do
  curl -LO https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/main/$f
done
```

## Data — two tracks

**Trial (~few hours):** Common Voice ru validation+test, capped at 20 h.

```bash
python finetune/fetch_dataset.py --profile trial --out data/trial_corpus
```

**SOTA-style (~400+ h public data):** CV-ru all splits + Golos-100 + Golos-10 + Sova.

```bash
python finetune/fetch_dataset.py --profile sota --out data/ru_corpus
# or pick sources: --sources cv-ru golos10
# single voice trial: --single-speaker auto
```

Both write `clips/` + `meta.tsv` (path / speaker / normalized sentence —
digits and symbols expanded to Russian words). Then:

```bash
python finetune/prepare_jsonl.py --meta data/ru_corpus/meta.tsv \
    --clips data/ru_corpus --out data/train.jsonl --val-out data/val.jsonl \
    --val-by-speaker        # speaker-disjoint val — no leakage
```

Check dataset licenses yourself — the pack only downloads public HF mirrors;
nothing is redistributed.

## Precompute (the speedup)

One pass over the manifest runs the frozen thinker + VAE and writes per-sample
`.pt` files (target latent, ref latent, fused 2048-d text embeds). Training then
runs **without the 3B encoder** — roughly 2-3× cheaper steps and ~6 GB less VRAM.

```bash
PYTHONPATH=$AUK_REPO/src python finetune/precompute_cond.py \
    --jsonl data/train.jsonl --out data/cache/train \
    --config finetune/config_flash_ru.yaml \
    --ckpt  ckpts/AuK-Flash/auk_flash.safetensors
# same for val:
PYTHONPATH=$AUK_REPO/src python finetune/precompute_cond.py \
    --jsonl data/val.jsonl --out data/cache/val \
    --config finetune/config_flash_ru.yaml --ckpt ckpts/AuK-Flash/auk_flash.safetensors
```

Resumable (skips existing files), writes `ok.txt` index map so partially-failed
samples are simply excluded, plus `meta.json` fingerprint — `smoke_test --cache`
warns if the jsonl changed since the cache was built (re-run precompute then;
the cache is raw tensors, not self-invalidating).

## Smoke (before burning GPU-hours)

```bash
python finetune/smoke_test.py --jsonl data/train.jsonl data/val.jsonl \
    --layout --ckpt-dir ckpts/AuK-Flash --qwen-dir ckpts/Qwen2.5-Omni-3B
python finetune/smoke_test.py --cache data/cache/train   # cache integrity
python finetune/smoke_test.py --model-smoke --auk-repo $PWD/AuK   # fwd+bwd both paths
```

## Train

```bash
AUK_REPO=$PWD/AuK bash finetune/train.sh                      # classic path
cond_cache=data/cache AUK_REPO=$PWD/AuK bash finetune/train.sh # cached path (fast)
# L40S/48GB: defaults fine. 24GB: frames_threshold=1200 max_samples=4
# short runs (<10k updates): ema_beta=0.999 — default 0.9999 EMA won't track
# watch: output_dir/samples/update_N_gen.wav — synthesized with the 4-step
#        Flash recipe when flash_grid is active
```

Resume is automatic if `output_dir/model_last.pt` exists (fresh run → new output_dir).

## Deploy

```bash
python finetune/export_ema.py --ckpt runs/auk_flash_ru/model_last.pt \
    --out auk_flash_ru.safetensors --ref ckpts/AuK-Flash/auk_flash.safetensors
```

EMA → safetensors in the original key layout (text_encoder excluded — it's the
frozen upstream thinker). Then `convert_to_mlx.py` + `quantize_mlx.py` →
MLX-8bit artifact identical to vanch007-style builds, or drop the safetensors
into the ComfyUI pack layout.

## Speed/memory cheatsheet

| Lever | Effect |
|---|---|
| `--cond_cache` | thinker out of the loop: −6 GB VRAM, ~2-3× step |
| `bitsandbytes` installed | AdamW states 12 GB → ~3 GB (auto-detected) |
| `attn_backend: flash_attn` in yaml + `pip install flash-attn` | faster long-context attn |
| `ema_beta=0.999` for short runs | EMA actually converges to trained weights |
| `frames_threshold` | batch size control; 2700 ok @48GB, ~1200 @24GB |

## Risks (honest)

- `flash_grid` is off-label: 4-point t-training may underfit complex phonemes.
  Check `samples/` early (~500 updates). Fallback: `init_ckpt=auk_base` +
  `t_sampling=logistic_normal` — reliable, loses 4-step speed.
- `drop_prob=0` matches Flash inference (CFG permanently off).
- Text normalization is the #1 quality lever for ru — inspect `meta.tsv`
  rows before training; garbage transcripts degrade more than small data.
