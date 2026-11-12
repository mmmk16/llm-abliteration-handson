"""0章 手順 5 の確認: transformers でモデルを読み込み、生成速度とメモリを測る"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "日本で一番高い山は?"
MAX_NEW = 50

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="mps")
model.eval()

messages = [{"role": "user", "content": PROMPT}]
# Qwen3 は既定で thinking モードが入るので、短く答えさせるため無効にする
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tok(text, return_tensors="pt").to("mps")

t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
torch.mps.synchronize()
dt = time.perf_counter() - t0

new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
print("--- output")
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print("--- stats")
print(f"model: {MODEL}")
print(f"new tokens: {new_tokens}  time: {dt:.2f}s  tokens/sec: {new_tokens / dt:.1f}")
print(f"current_allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
print(f"driver_allocated:  {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
