#!/usr/bin/env python3
"""Precompute frozen conditioning + VAE latents for cached training (patch 0002).

Runs the frozen Qwen thinker and VAE once over a JSONL manifest and writes one
.pt file per sample. With --cond_cache, the trainer skips the thinker entirely
(~6 GB VRAM) and skips VAE encoding in the train loop — the dominant per-step
cost for a frozen 3B encoder becomes zero.

    PYTHONPATH=$AUK_REPO/src python precompute_cond.py \
        --jsonl data/train.jsonl --out data/cache/train \
        --config finetune/config_flash_ru.yaml \
        --ckpt  ckpts/AuK-Flash/auk_flash.safetensors

For val split, run again with --out <same_root>/val and --jsonl val.jsonl.

Output file {out}/{index:06d}.pt contains:
    latent       fp16 [T, 64]    target VAE latent
    latent_len   int64 scalar    valid frames of target latent
    ref_latent   fp16 [T', 64]   reference (prompt) VAE latent, empty if none
    ref_len      int64 scalar
    text_embeds  fp16 [S, 2048]  fused thinker conditioning (layer_weights/scale
                                 taken from --ckpt, matching inference-time fusion)
    text_len     int64 scalar    valid tokens of text_embeds

Resumable: existing files are skipped. Requires the full train env
(transformers + qwen_omni_utils + torchaudio + auk deps on PYTHONPATH).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

TARGET_SR = 24000


def load_wav(path, resamplers):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
        audio = resamplers[sr](audio)
    return audio


def prep_cond_messages(messages):
    """Mirror AukJsonlDataset.__getitem__: mark text-only user turns, drop assistant."""
    msgs = [dict(m, content=[dict(c) for c in m["content"]]) for m in messages]
    for m in msgs:
        if m["role"] == "user" and not any(c["type"] == "audio" for c in m["content"]):
            for c in m["content"]:
                if c["type"] == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                    c["text"] += "|<no_prompt_audio>|"
    return [m for m in msgs if m["role"] != "assistant"]


def read_fusion_params(ckpt_path):
    """layer_weights / layer_scale live at CFMEdit top level in the ckpt."""
    from safetensors import safe_open
    w, s = None, None
    with safe_open(ckpt_path, "pt") as f:
        for k in f.keys():
            if k.endswith("layer_weights"):
                w = f.get_tensor(k)
            elif k.endswith("layer_scale"):
                s = f.get_tensor(k)
    if w is None:
        print(f"warn: layer_weights not found in {ckpt_path}; using zeros(36)", file=sys.stderr)
        w = torch.zeros(36)
    if s is None:
        print(f"warn: layer_scale not found in {ckpt_path}; using ones(1)", file=sys.stderr)
        s = torch.ones(1)
    return w.float(), s.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True, help="cache dir, e.g. data/cache/train")
    ap.add_argument("--config", default="ckpts/AuK/config.yaml",
                    help="model yaml (reads vae + text_encoder paths)")
    ap.add_argument("--ckpt", required=True, help="auk_flash.safetensors (fusion params)")
    ap.add_argument("--auk-repo", default=os.environ.get("AUK_REPO", "AuK"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.auk_repo, "src"))
    from omegaconf import OmegaConf
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from auk.model.cfm_edit import CFMEdit
    from auk.model.vae import load_vae_model
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig

    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out, exist_ok=True)

    mc = OmegaConf.load(a.config).model
    print(f"loading thinker from {mc.text_encoder.text_encoder_path} ...")
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        mc.text_encoder.text_encoder_path, torch_dtype=torch.bfloat16)
    if thinker.visual is not None:
        del thinker.visual
        thinker.visual = None
    thinker.lm_head = torch.nn.Identity()  # hidden states only — no logits needed
    thinker.config.use_cache = False
    thinker.eval().requires_grad_(False).to(a.device)
    processor = Qwen2_5OmniProcessor.from_pretrained(mc.text_encoder.text_encoder_path)

    layer_w, layer_s = read_fusion_params(a.ckpt)

    print(f"loading VAE {mc.vae.vae_name} ...")
    vae = load_vae_model(
        vae_name=mc.vae.vae_name,
        vae_cfg=BigVGANFlowVAEConfig.from_dict(
            OmegaConf.to_container(mc.vae.get("model_init_kwargs", OmegaConf.create({})), resolve=True)),
        vae_ckpt=mc.vae.vae_model_path,
        map_location="cpu",
    ).eval().requires_grad_(False).to(a.device)

    resamplers = {}
    done_ids = []
    skipped = 0
    for i, row in enumerate(rows):
        dst = os.path.join(a.out, f"{i:06d}.pt")
        if os.path.exists(dst):
            done_ids.append(i)
            continue
        try:
            messages = row["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            cond_msgs = prep_cond_messages(messages)

            cond = CFMEdit.build_cond_inputs([cond_msgs], processor).to(a.device)
            with torch.no_grad():
                out = thinker(**cond, output_hidden_states=True)
            hs = out.hidden_states
            d = hs[0].shape[-1]
            stacked = torch.stack([F.layer_norm(h, [d]) for h in hs[1:]])  # (L,1,S,d)
            w = F.softmax(layer_w.to(stacked.device), dim=0)
            text_embeds = (stacked * w[:, None, None, None]).sum(0) * layer_s.to(stacked.device)
            text_embeds = text_embeds[0]  # [S, d]
            text_len = int(cond["attention_mask"][0].sum().item())

            urls = {}
            for m in messages:
                for c in m["content"]:
                    if c["type"] == "audio":
                        urls[m["role"]] = c.get("audio_url") or c.get("audio")

            tgt = load_wav(urls["assistant"], resamplers).to(a.device)
            with torch.no_grad():
                latent, latent_lens = vae.encoding_and_normalization(
                    tgt.unsqueeze(0), sample_lengths=torch.tensor([tgt.shape[-1]], device=a.device))

            ref_url = urls.get("user")
            if ref_url:
                ref = load_wav(ref_url, resamplers).to(a.device)
                with torch.no_grad():
                    ref_latent, ref_lens = vae.encoding_and_normalization(
                        ref.unsqueeze(0), sample_lengths=torch.tensor([ref.shape[-1]], device=a.device))
            else:
                ref_latent = torch.zeros(1, 0, latent.shape[-1], device=a.device)
                ref_lens = torch.zeros(1, dtype=torch.long, device=a.device)

            torch.save({
                "latent": latent[0].half().cpu(),
                "latent_len": latent_lens[0].cpu(),
                "ref_latent": ref_latent[0].half().cpu(),
                "ref_len": ref_lens[0].cpu(),
                "text_embeds": text_embeds.half().cpu(),
                "text_len": torch.tensor(text_len),
            }, dst)
            done_ids.append(i)
            if len(done_ids) % 50 == 0:
                print(f"{len(done_ids)}/{len(rows)} cached")
        except Exception as e:
            skipped += 1
            print(f"  sample {i} failed: {e}", file=sys.stderr)

    # trainer-side index map: only rows with a cache file are iterated
    with open(os.path.join(a.out, "ok.txt"), "w") as f:
        for i in sorted(done_ids):
            f.write(f"{i}\n")
    # fingerprint for stale-cache detection (checked by smoke_test --cache)
    import hashlib
    jhash = hashlib.sha1(open(a.jsonl, "rb").read()).hexdigest()[:12]
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"jsonl": os.path.abspath(a.jsonl), "jsonl_sha1": jhash,
                   "ckpt": os.path.abspath(a.ckpt), "config": os.path.abspath(a.config),
                   "thinker": str(mc.text_encoder.text_encoder_path),
                   "n_files": len(done_ids)}, f, indent=2)
    print(f"done: {len(done_ids)} cached / {skipped} failed -> {a.out} (ok.txt + meta.json written)")


if __name__ == "__main__":
    main()
