"""Strip optimizer state from a training checkpoint for release.

Keeps model weights, config, and metadata; drops optimizer/scheduler state
(~3x size reduction). The stripped file loads identically in evaluate.py.

Usage: python strip_checkpoint.py <checkpoint.pt> [output.pt]
"""
import sys

import torch

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".pt", "_stripped.pt")

ckpt = torch.load(src, map_location="cpu", weights_only=False)
kept = {k: v for k, v in ckpt.items()
        if k not in ("optimizer_state_dict", "scheduler_state_dict")}
print(f"keys kept: {sorted(kept.keys())}")
dropped = sorted(set(ckpt.keys()) - set(kept.keys()))
print(f"keys dropped: {dropped}")
torch.save(kept, dst)

import os
print(f"{src}: {os.path.getsize(src)/1e6:.0f} MB -> {dst}: {os.path.getsize(dst)/1e6:.0f} MB")
