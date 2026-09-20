"""Re-save Qwen2.5-Omni-3B as a thinker-only checkpoint.

Drops talker + token2wav (~5GB AuK never loads), the vision tower (~1.1GB)
and lm_head (~0.6GB). Result: ~4.4GB on disk instead of ~11.5GB.

Usage:
    python scripts/strip_thinker.py <src_dir_or_repo_id> <dst_dir>
    python scripts/strip_thinker.py Qwen/Qwen2.5-Omni-3B ckpts/Qwen2.5-Omni-3B-thinker
"""
import sys
import torch
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

src, dst = sys.argv[1], sys.argv[2]

print(f"Loading thinker from {src}...")
thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    src, dtype=torch.bfloat16, low_cpu_mem_usage=True
)
thinker.visual = None
thinker.lm_head = torch.nn.Identity()

thinker.save_pretrained(dst)
Qwen2_5OmniProcessor.from_pretrained(src).save_pretrained(dst)

import os
size_gb = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, files in os.walk(dst)
    for f in files
) / 1024**3
print(f"Saved thinker-only checkpoint to {dst} ({size_gb:.2f} GB)")
