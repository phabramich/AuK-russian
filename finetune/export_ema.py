#!/usr/bin/env python3
"""Export trained EMA weights -> safetensors (same key layout as auk_flash).

    python export_ema.py --ckpt runs/auk_flash_ru/model_last.pt \
        --out runs/auk_flash_ru/auk_flash_ru.safetensors \
        [--ref ckpts/AuK-Flash/auk_flash.safetensors]

--ref compares exported keys against the original Flash ckpt (missing/extra
keys are reported; text_encoder.* is expected to be absent — it is frozen and
never saved).
"""
import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model_last.pt or model_N.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="original auk_flash.safetensors for key check")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict")
    if ema is None:
        sys.exit(f"no model/ema state in {a.ckpt}; keys: {sorted(ckpt)}")
    src = "ema_model_state_dict" if "ema_model_state_dict" in ckpt else "model_state_dict"
    state = {k.replace("ema_model.", ""): v.float() for k, v in ema.items()
             if k not in ("initted", "step") and "text_encoder." not in k}

    save_file(state, a.out, metadata={"source": src, "update": str(ckpt.get("update", -1))})
    print(f"wrote {a.out} | {len(state)} tensors from {src} (update={ckpt.get('update')})")

    if a.ref:
        from safetensors import safe_open
        with safe_open(a.ref, "pt") as f:
            ref_keys = {k for k in f.keys() if "text_encoder." not in k}
        mine = set(state)
        missing, extra = sorted(ref_keys - mine), sorted(mine - ref_keys)
        print(f"key check vs {a.ref}: missing={len(missing)} extra={len(extra)}")
        for k in (missing + extra)[:10]:
            print("  ", k)


if __name__ == "__main__":
    main()
