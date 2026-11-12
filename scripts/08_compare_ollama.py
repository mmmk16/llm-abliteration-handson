"""実験 08: Ollama に登録した加工後のモデルを、同じ手順で変換・登録した加工前のモデルと同じ指標で比べる。

比較相手に Ollama 公式タグ(qwen3:4b など)は使わない。量子化済みで、thinking 固定の版を指すこともあり、同条件にならないため。

Ollama の HTTP API(/api/chat)に同じ指示を投げ、拒否率(有害はラベルのみ、本文は出さない)、
生成速度(Ollama が返す eval_count / eval_duration)、無害な依頼への本文を並べる。
評価データは実験 06 と同じ data/eval_*.json。

使い方(Ollama が起動していること):
    uv run python scripts/08_compare_ollama.py --base qwen3-1.7b-original --abl qwen3-1.7b-abliterated
    uv run python scripts/08_compare_ollama.py --base qwen3-4b-original --abl qwen3-4b-abliterated
出力: outputs/08_ollama_<abl>.md
"""
import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import md_cell
from refusal_labels import LABELS, classify, count_labels, refusal_rate

GENERAL = ["日本で一番高い山は?一文で答えて。", "17 × 23 を計算して。答えだけ。"]  # 本文を並べる無害な質問


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="qwen3-1.7b-original", help="比較対象(同じ手順で登録した加工前のモデル)")
    p.add_argument("--abl", default="qwen3-1.7b-abliterated", help="ollama create で登録した加工済みモデル")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--max-new", type=int, default=40)
    return p.parse_args()


def chat(host: str, model: str, prompt: str, max_new: int) -> tuple[str, float]:
    """/api/chat に 1 回問い合わせ、(返答, tok/s) を返す。think=False で Qwen3 の thinking を切り、温度 0 で貪欲にする。"""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "think": False, "options": {"temperature": 0, "num_predict": max_new}}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    tps = d["eval_count"] / d["eval_duration"] * 1e9 if d.get("eval_duration") else 0.0  # eval_duration はナノ秒
    return d["message"]["content"], tps


@dataclass
class Result:
    """モデル 1 つぶんの結果。有害指示の本文は持たない。"""
    name: str
    harmful: dict[str, int]     # 有害のラベル内訳
    harmless: dict[str, int]    # 無害のラベル内訳
    tps: float                  # 無害依頼での平均 tok/s
    samples: list[str]          # GENERAL + 無害 2 件への返答(先頭 120 文字)


def evaluate(host: str, name: str, harmful: list[str], harmless: list[str], max_new: int) -> Result:
    """1 モデルについて拒否率、速度、返答例を集める。"""
    c = count_labels(chat(host, name, p, max_new)[0] for p in harmful)
    g, tps = {label: 0 for label in LABELS}, []  # 無害側は tok/s も集めるので手で回す
    for p in harmless:
        text, t = chat(host, name, p, max_new)
        g[classify(text)] += 1
        tps.append(t)
    samples = [chat(host, name, p, 80)[0].replace("\n", " ")[:120] for p in GENERAL + harmless[:2]]
    r = Result(name, c, g, sum(tps) / len(tps), samples)
    print(f"[{name}] 有害拒否率 {refusal_rate(c):.0%} {c}  無害拒否率 {refusal_rate(g):.0%} {g}  {r.tps:.1f} tok/s")
    return r


def write_report(args, n_harmful: int, n_harmless: int, results: list[Result], harmless: list[str]) -> None:
    """比較表と無害な依頼への本文を Markdown で保存する。"""
    tag = args.abl.replace(":", "_")
    lines = [f"# 実験 08: Ollama で比較 ({args.base} vs {args.abl})", "",
             f"評価データ: 有害 {n_harmful} 件 / 無害 {n_harmless} 件(実験 06 と同じ)。有害本文は載せない。", "",
             "| モデル | 有害拒否率 | 無害拒否率 | 無害崩壊 | tok/s |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {refusal_rate(r.harmful):.0%} | {refusal_rate(r.harmless):.0%} | {r.harmless['崩壊']} | {r.tps:.1f} |")
    lines += ["", "## 無害な依頼への本文", "",
              "評価データの依頼の本文は載せない(Alpaca、CC BY-NC 4.0)。番号は data/eval_harmless.json の並び順。", ""]
    # GENERAL は自分で書いた質問なので本文を載せる。評価データ由来の依頼は番号だけにする
    titles = [f"「{p}」" for p in GENERAL] + [f"無害な依頼 {k}" for k in range(1, len(harmless[:2]) + 1)]
    for i, title in enumerate(titles):
        lines.append(f"- {title}")
        for r in results:
            lines.append(f"  - {r.name}: {md_cell(r.samples[i])}")
    Path("outputs").mkdir(exist_ok=True)
    (Path("outputs") / f"08_ollama_{tag}.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote outputs/08_ollama_{tag}.md")


def main() -> None:
    args = parse_args()
    harmful = json.loads(Path("data/eval_harmful.json").read_text())
    harmless = json.loads(Path("data/eval_harmless.json").read_text())
    results = [evaluate(args.host, name, harmful, harmless, args.max_new) for name in (args.base, args.abl)]
    write_report(args, len(harmful), len(harmless), results, harmless)


if __name__ == "__main__":
    main()
