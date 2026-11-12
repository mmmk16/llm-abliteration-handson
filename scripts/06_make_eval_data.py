"""実験 06 のデータ作成: 実験 03 で使わなかった指示を評価用に抜き出す。

実験 03〜05 は data/harmful.json, data/harmless.json(各 64 件)で方向を求め、その末尾 16 件で効きを見た。
ここでは同じ公開データから、その 64 件と重ならない指示を各 20 件抜く。
「方向を求めたときの指示」ではなく「初めて見る言い回し」でも拒否が消えているかを確かめるため。

使い方:
    uv run python scripts/06_make_eval_data.py
HF_HUB_OFFLINE=1 を常に設定している場合は、このときだけ HF_HUB_OFFLINE=0 を付ける。
出力: data/eval_harmful.json, data/eval_harmless.json(git 管理外)
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


def load_used_prompts() -> set[str]:
    """実験 03 で使った指示(有害・無害とも)の集合。これらは評価から除く。"""
    return (set(json.loads((DATA_DIR / "harmful.json").read_text()))
            | set(json.loads((DATA_DIR / "harmless.json").read_text())))


def fetch_advbench(exclude: set[str]) -> list[str]:
    """AdvBench の指示文のうち、exclude に含まれないものを返す。"""
    with urllib.request.urlopen(ADVBENCH_URL, timeout=60) as r:
        text = r.read().decode("utf-8")
    return [row["goal"] for row in csv.DictReader(io.StringIO(text)) if row["goal"] not in exclude]


def load_alpaca_instructions(exclude: set[str], min_len: int = 20, max_len: int = 150) -> list[str]:
    """Alpaca の「追加入力なし」で長さが手頃な指示文のうち、exclude に含まれないものを返す。"""
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    return [
        ex["instruction"] for ex in alpaca
        if ex["input"] == "" and min_len <= len(ex["instruction"]) <= max_len and ex["instruction"] not in exclude
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="各クラスの件数")
    p.add_argument("--seed", type=int, default=1, help="実験 03 とは別の seed")
    args = p.parse_args()
    rng = random.Random(args.seed)

    used = load_used_prompts()
    harmful_all = fetch_advbench(used)
    harmless_all = load_alpaca_instructions(used)
    print(f"候補: 有害 {len(harmful_all)} 件 / 無害 {len(harmless_all)} 件(実験 03 で使った {len(used)} 件は除外済み)")

    harmful = rng.sample(harmful_all, args.n)
    harmless = rng.sample(harmless_all, args.n)
    (DATA_DIR / "eval_harmful.json").write_text(json.dumps(harmful, ensure_ascii=False, indent=1))
    (DATA_DIR / "eval_harmless.json").write_text(json.dumps(harmless, ensure_ascii=False, indent=1))
    print(f"wrote data/eval_harmful.json, data/eval_harmless.json ({args.n} 件ずつ)")
    print("例(無害):", *harmless[:3], sep="\n  ")


if __name__ == "__main__":
    main()
