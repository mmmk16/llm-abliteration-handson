"""対話 REPL: 加工前後のモデルとチャット形式で続けてやりとりする。

会話履歴を保持して多ターンで話せる。出力はトークンごとに流して表示する。

使い方:
    uv run python scripts/chat_repl.py                                   # models/Qwen3-1.7B-abliterated
    uv run python scripts/chat_repl.py models/Qwen3-4B-abliterated
    uv run python scripts/chat_repl.py Qwen/Qwen3-1.7B                   # 加工前と比べたいとき
    uv run python scripts/chat_repl.py --think models/Qwen3-4B-abliterated   # thinking を有効にする
REPL 内のコマンド:
    /reset  会話履歴を消す     /system <文>  system プロンプトを設定     /exit  終了(Ctrl-D でも可)
"""
import argparse
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="models/Qwen3-1.7B-abliterated")
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0, help="0 なら貪欲(乱数を使わず、ほぼ同じ答え)")
    p.add_argument("--think", action="store_true", help="Qwen3 の thinking モードを有効にする")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="mps").eval()
    print(f"model: {args.model}  (/reset で履歴を消す, /exit で終了)")
    messages: list[dict] = []
    system: str | None = None

    while True:
        try:
            user = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = []
            print("(履歴を消しました)")
            continue
        if user.startswith("/system "):
            system = user[len("/system "):].strip()
            messages = []
            print(f"(system を設定し、履歴を消しました: {system})")
            continue

        messages.append({"role": "user", "content": user})
        convo = ([{"role": "system", "content": system}] if system else []) + messages
        text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, enable_thinking=args.think)
        inputs = tok(text, return_tensors="pt").to("mps")
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=args.max_new, streamer=streamer,
                          do_sample=args.temperature > 0, temperature=args.temperature or None)
        Thread(target=lambda: model.generate(**gen_kwargs), daemon=True).start()

        print("モデル> ", end="", flush=True)
        reply = []
        try:
            for piece in streamer:
                print(piece, end="", flush=True)
                reply.append(piece)
        except KeyboardInterrupt:
            print("\n(中断)")
        print()
        messages.append({"role": "assistant", "content": "".join(reply).strip()})


if __name__ == "__main__":
    main()
