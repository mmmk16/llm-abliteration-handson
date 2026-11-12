"""実験 03 のデータ作成: 有害な指示と無害な指示を公開データセットから少量抜き出す。

- 有害: AdvBench harmful_behaviors(llm-attacks リポジトリ、MIT)。「〜のやり方を書け」のような指示 520 件。
- 無害: Alpaca(tatsu-lab/alpaca、CC BY-NC 4.0)。入力なしの一般的な指示から抜く。
どちらも英語。拒否方向は言語をまたいで効くことが知られているので、方向を求めるのは英語で行い、
効き方を見る実験 04 以降では日本語の依頼も使う。

使い方:
    uv run python scripts/03_make_data.py            # 各 64 件
    uv run python scripts/03_make_data.py --n 128
HF_HUB_OFFLINE=1 を常に設定している場合は、このときだけ HF_HUB_OFFLINE=0 を付ける。
出力: data/harmful.json, data/harmless.json(git 管理外)
"""
import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

from datasets import load_dataset

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
DATA_DIR = Path("data")


def fetch_advbench() -> list[str]:
    """AdvBench の CSV を取得し、指示文(goal 列)のリストを返す。"""
    with urllib.request.urlopen(ADVBENCH_URL, timeout=60) as r:
        text = r.read().decode("utf-8")
    return [row["goal"] for row in csv.DictReader(io.StringIO(text))]


def load_alpaca_instructions(min_len: int = 20, max_len: int = 150) -> list[str]:
    """Alpaca から「追加入力なし」で長さが手頃な指示文だけを返す。"""
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    return [
        ex["instruction"] for ex in alpaca
        if ex["input"] == "" and min_len <= len(ex["instruction"]) <= max_len
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64, help="各クラスの件数")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    harmful_all = fetch_advbench()
    print(f"AdvBench: {len(harmful_all)} 件")
    harmless_all = load_alpaca_instructions()
    print(f"Alpaca(入力なし・20〜150文字): {len(harmless_all)} 件")

    harmful = rng.sample(harmful_all, args.n)
    harmless = rng.sample(harmless_all, args.n)

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "harmful.json").write_text(json.dumps(harmful, ensure_ascii=False, indent=1))
    (DATA_DIR / "harmless.json").write_text(json.dumps(harmless, ensure_ascii=False, indent=1))
    print(f"\nwrote data/harmful.json, data/harmless.json ({args.n} 件ずつ)")
    print("例(無害):", *harmless[:3], sep="\n  ")  # 有害な指示の本文は画面に出さない


if __name__ == "__main__":
    main()
