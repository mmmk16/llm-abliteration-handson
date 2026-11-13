"""0章 手順 3 の確認: TransformerLens 4.x (TransformerBridge) が MPS で動くか"""
import sys

import torch
from transformer_lens import TransformerBridge

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"

model = TransformerBridge.boot_transformers(MODEL, device="mps", dtype=torch.bfloat16)
logits, cache = model.run_with_cache("日本で一番高い山は")

print("logits:", tuple(logits.shape))
keys = list(cache.keys())
print("cached activations:", len(keys))
for k in keys[:8]:
    print(" ", k, tuple(cache[k].shape))
