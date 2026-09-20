#!/usr/bin/env python3
"""Smoke checks for the AuK-Flash RU fine-tune pack. No model weights needed.

  --jsonl data/train.jsonl   validate manifest (schema, files, durations)
  --cache data/cache/train   validate precompute_cond.py output (.pt files)
  --model-smoke              tiny CFMEdit forward+backward w/ flash_grid +
                             cached-cond path (needs upstream repo: AUK_REPO env)
  --layout                   check ckpt/qwen file layout before a run
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

GRID = {0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835}
MIN_SEC, MAX_SEC = 0.3, 30.0


def check_jsonl(path):
    ok, bad = 0, 0
    problems = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        i += 1
        try:
            s = json.loads(line)
            msgs = s["messages"]
            dur = float(s["duration"])
            assert 0 < dur <= MAX_SEC, f"duration {dur}"
            user = next(m for m in msgs if m["role"] == "user")
            tgt = next(m for m in msgs if m["role"] == "assistant")
            texts = [c for c in user["content"] if c.get("type") == "text"]
            auds_u = [c for c in user["content"] if c.get("type") == "audio"]
            auds_t = [c for c in tgt["content"] if c.get("type") == "audio"]
            assert texts, "user has no text item"
            assert auds_t, "assistant has no audio item"
            for c in auds_u + auds_t:
                p = c.get("audio_url") or c.get("audio")
                assert p, "audio item without path"
                assert os.path.isfile(p), f"not found: {p}"
            ok += 1
        except Exception as e:
            bad += 1
            problems.append(f"  line {i}: {e}")
    print(f"{path}: {ok} ok, {bad} bad")
    for p in problems[:15]:
        print(p)
    return bad == 0


def check_cache(cache_dir, jsonl_path=None):
    """Validate precompute_cond.py output without touching the model."""
    import torch
    need = {"latent", "latent_len", "ref_latent", "ref_len", "text_embeds", "text_len"}
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    if not files:
        print(f"{cache_dir}: no .pt files")
        return False
    ok = bad = 0
    for fn in files:
        try:
            d = torch.load(os.path.join(cache_dir, fn), map_location="cpu",
                           weights_only=True)
            assert need <= set(d), f"missing keys {need - set(d)}"
            assert d["latent"].ndim == 2 and d["latent"].shape[1] == 64, \
                f"latent {tuple(d['latent'].shape)}"
            assert d["ref_latent"].ndim == 2 and d["ref_latent"].shape[1] == 64, \
                f"ref_latent {tuple(d['ref_latent'].shape)}"
            assert d["text_embeds"].ndim == 2, f"text_embeds {tuple(d['text_embeds'].shape)}"
            assert int(d["latent_len"]) <= d["latent"].shape[0]
            assert int(d["ref_len"]) <= d["ref_latent"].shape[0]
            assert int(d["text_len"]) <= d["text_embeds"].shape[0]
            ok += 1
        except Exception as e:
            bad += 1
            print(f"  {fn}: {e}")
    line_n = None
    if jsonl_path and os.path.isfile(jsonl_path):
        line_n = sum(1 for l in open(jsonl_path, encoding="utf-8") if l.strip())
    has_oktxt = os.path.isfile(os.path.join(cache_dir, "ok.txt"))
    extra = f" | ok.txt={'yes' if has_oktxt else 'NO (partial cache will crash training)'}"
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.isfile(meta_path) and jsonl_path and os.path.isfile(jsonl_path):
        import hashlib
        meta = json.load(open(meta_path))
        jhash = hashlib.sha1(open(jsonl_path, "rb").read()).hexdigest()[:12]
        if meta.get("jsonl_sha1") and meta["jsonl_sha1"] != jhash:
            extra += f" | STALE (built from different jsonl: {meta.get('jsonl')})"
            bad += 1
        elif meta.get("jsonl_sha1"):
            extra += " | fingerprint OK"
    if line_n is not None:
        extra += f" | jsonl lines={line_n} cache files={len(files)}"
        if not has_oktxt and line_n != len(files):
            extra += " MISMATCH"
    print(f"{cache_dir}: {ok} ok, {bad} bad{extra}")
    return bad == 0 and (line_n is None or has_oktxt or line_n == len(files))


def check_layout(ckpt_dir, qwen_dir):
    need = [
        (f"{ckpt_dir}/auk_flash.safetensors", "DiT checkpoint"),
        (f"{ckpt_dir}/vae.safetensors", "VAE"),
        (f"{qwen_dir}/model-00001-of-00003.safetensors", "thinker shard 1"),
        (f"{qwen_dir}/model-00002-of-00003.safetensors", "thinker shard 2"),
        (f"{qwen_dir}/model.safetensors.index.json", "weight index"),
        (f"{qwen_dir}/config.json", "qwen config"),
        (f"{qwen_dir}/preprocessor_config.json", "processor cfg"),
        (f"{qwen_dir}/tokenizer.json", "tokenizer"),
    ]
    bad = 0
    for p, name in need:
        exists = os.path.isfile(p)
        print(f"  {'OK ' if exists else 'MISS'} {name}: {p}")
        bad += not exists
    return bad == 0


def model_smoke(auk_repo):
    sys.path.insert(0, os.path.join(auk_repo, "src"))
    import torch
    import torch.nn as nn
    from auk.model.cfm_edit import CFMEdit
    from auk.model.flux2_edit import Flux2Edit

    class StubEnc(nn.Module):
        """36-layer hidden-state stub; skips real text encoding."""
        def __init__(self, d=32, n_layers=36):
            super().__init__()
            self.d, self.n = d, n_layers
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=n_layers))

        def forward(self, **kw):
            b, n = 1, 4
            hs = tuple(torch.randn(b, n, self.d) for _ in range(self.n + 1))
            return SimpleNamespace(hidden_states=hs)

    dit = Flux2Edit(dim=64, heads=2, dim_head=32, num_layers=1,
                    num_single_layers=1, latent_dim=8, text_hidden_dim=32,
                    attn_backend="torch")
    model = CFMEdit(transformer=dit, text_encoder=StubEnc(),
                    text_processor=None, num_channels=8,
                    t_sampling="flash_grid",
                    audio_drop_prob=0.0, cond_drop_prob=0.0)

    # 1) flash_grid sampling only returns grid points
    t = model.sample_time(512, torch.float32, torch.device("cpu"))
    got = set(t.tolist())
    assert got <= GRID, f"t outside grid: {got - GRID}"
    print(f"flash_grid sampling OK — values seen: {sorted(got)}")

    # 2) one forward+backward on synthetic latents
    cond = {"input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    latent = torch.randn(1, 16, 8)
    x0 = torch.randn(1, 16, 8)
    ref = torch.randn(1, 8, 8)
    lens = torch.tensor([16])
    loss, _, _ = model(latent, text=cond, ref_latent=ref, lens=lens, x0=x0)
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    loss.backward()
    grads = sum(p.grad is not None for p in model.parameters())
    print(f"forward+backward OK — loss={loss.item():.4f}, grads on {grads} params")

    # 3) cached-conditioning path (patch 0002): text_embeds dict bypasses encoder
    model.zero_grad()
    cached = {"text_embeds": torch.randn(1, 4, 32),
              "context_mask": torch.ones(1, 4, dtype=torch.bool)}
    emb, mask = model.encode_text(cached, torch.device("cpu"))
    assert emb.shape == (1, 4, 32) and emb.dtype == torch.float32 and mask.dtype == torch.bool, \
        f"encode_text cache path: {emb.shape} {emb.dtype} {mask.dtype}"
    loss2, _, _ = model(latent, text=cached, ref_latent=ref, lens=lens, x0=x0)
    assert torch.isfinite(loss2), f"non-finite cached-path loss {loss2}"
    loss2.backward()
    print(f"cached-cond forward+backward OK — loss={loss2.item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="*", default=[])
    ap.add_argument("--cache", nargs="*", default=[])
    ap.add_argument("--model-smoke", action="store_true")
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--auk-repo", default=os.environ.get("AUK_REPO", "AuK"))
    ap.add_argument("--ckpt-dir", default="ckpts/AuK-Flash")
    ap.add_argument("--qwen-dir", default="ckpts/Qwen2.5-Omni-3B")
    a = ap.parse_args()

    ok = True
    for j in a.jsonl:
        ok &= check_jsonl(j)
    for c in a.cache:
        # compare against same-named jsonl if given (train/ <-> train.jsonl)
        j = a.jsonl[0] if len(a.jsonl) == 1 else None
        ok &= check_cache(c, j)
    if a.layout:
        ok &= check_layout(a.ckpt_dir, a.qwen_dir)
    if a.model_smoke:
        model_smoke(a.auk_repo)
    if not (a.jsonl or a.layout or a.model_smoke or a.cache):
        ap.print_help()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
