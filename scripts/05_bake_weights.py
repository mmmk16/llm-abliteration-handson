"""実験 05: 拒否方向を重みに焼き込む(abliteration)。

実験 04 は生成のたびに hook で残差から方向を引いていた。ここでは hook をやめ、
「残差ストリームに書き込む重み」から拒否方向 u の成分をあらかじめ取り除く。
    W <- W - u uᵀ W      (W の出力が u 方向を持てなくなる)
対象は Qwen3 の
  - 埋め込み  embed_tokens.weight      (語彙 × d。行ベクトルが残差に置かれるので W <- W - (W u) uᵀ)
  - 各層の attention 出力  self_attn.o_proj.weight  (d × 内部次元)
  - 各層の MLP 出力        mlp.down_proj.weight     (d × 中間次元)
これで全層・全位置の残差から u 成分が消え、実験 04 の ablate と同じ効果を重みだけで得る。
書き換えたモデルは safetensors で保存し、TransformerLens なしの素の transformers で読み直して確かめる。

安全のための方針(実験 04 と同じ):
  有害な評価指示は 拒否/応答/崩壊 のラベルと割合だけを数え、本文は表示も保存もしない。
  本文を見るのは無害な依頼だけ。

使い方:
    uv run python scripts/05_bake_weights.py --layer 19                           # 1.7B、実験 04 の採用層
    uv run python scripts/05_bake_weights.py --model Qwen/Qwen3-4B --layer 20
出力:
    models/<model>-abliterated/     書き換えたモデル(git 管理外)
    outputs/05_bake_<model>.md      加工前後の拒否率、u 成分の残り、保存サイズ
"""
import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ 直下の共用モジュールを読む
from common import chat_prompt, model_tag
from refusal_labels import count_labels, refusal_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=None, help="使う方向の層(実験 04 の採用層。既定は実験 03 の候補層)")
    p.add_argument("--out", default=None, help="保存先(既定 models/<model>-abliterated)")
    p.add_argument("--n-eval", type=int, default=16, help="有害/無害 各クラスの評価件数(末尾から。16 が上限)")
    p.add_argument("--max-new", type=int, default=40)
    return p.parse_args()


def bake(model, u: torch.Tensor) -> int:
    """残差に書き込む重みから u 成分を除く。書き換えたテンソルの数を返す。

    計算は fp32 で行い、結果を元の dtype(bf16)に戻す。
    """
    n = 0
    with torch.no_grad():
        emb = model.model.embed_tokens.weight                      # (vocab, d)
        w = emb.float()
        emb.copy_((w - torch.outer(w @ u, u)).to(emb.dtype)); n += 1
        for layer in model.model.layers:
            for lin in (layer.self_attn.o_proj, layer.mlp.down_proj):  # weight: (d, in)
                w = lin.weight.float()
                lin.weight.copy_((w - torch.outer(u, u @ w)).to(lin.weight.dtype)); n += 1
    return n


def load_model(name_or_dir: str):
    """素の transformers でモデルを bf16 / MPS に読み込む(hook なし)。"""
    return AutoModelForCausalLM.from_pretrained(name_or_dir, dtype=torch.bfloat16, device_map="mps").eval()


def generate(model, tok, prompt: str, max_new: int) -> str:
    """チャット形式に包んで貪欲法で生成し、返答の文字列を返す。"""
    inputs = tok(chat_prompt(tok, prompt), return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def u_component(model, tok, u: torch.Tensor, prompt: str) -> float:
    """無害な依頼を 1 件通し、各層の入口の残差(末尾位置)が u 方向をどれだけ持つか |h·u|/|h| の層平均。

    加工後はほぼ 0 になるはず(bf16 の丸めで 0.00x 程度は残る)。
    """
    inputs = tok(chat_prompt(tok, prompt), return_tensors="pt").to("mps")
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states[: len(model.model.layers)]
    vals = [abs(float(h[0, -1].float() @ u)) / float(h[0, -1].float().norm()) for h in hs]
    return sum(vals) / len(vals)


@dataclass
class Measurement:
    """加工前(または加工後)のモデル 1 つぶんの測定結果。"""
    harmful: dict[str, int]     # 有害指示のラベル内訳(本文は持たない)
    harmless: dict[str, int]    # 無害依頼のラベル内訳
    u_comp: float               # 残差に残る u 成分
    samples: list[str]          # 無害依頼 3 件の返答(先頭 80 文字)

    @property
    def harmful_rate(self) -> float:
        return refusal_rate(self.harmful)

    @property
    def harmless_rate(self) -> float:
        return refusal_rate(self.harmless)


def measure(model, tok, u: torch.Tensor, harmful: list[str], harmless: list[str], max_new: int) -> Measurement:
    """拒否率、u 成分、無害依頼の返答例をまとめて測る。"""
    return Measurement(
        harmful=count_labels(generate(model, tok, p, max_new) for p in harmful),
        harmless=count_labels(generate(model, tok, p, max_new) for p in harmless),
        u_comp=u_component(model, tok, u, harmless[0]),
        samples=[generate(model, tok, p, max_new)[:80] for p in harmless[:3]],
    )


def write_report(args, layer: int, tied: bool, n_baked: int, out_dir: Path, size_gb: float,
                 before: Measurement, after: Measurement, harmless: list[str]) -> None:
    """加工前後の比較を Markdown で保存する。"""
    lines = [f"# 実験 05: 重みに焼き込む ({args.model}, 層 {layer} の方向)", "",
             f"書き換えたテンソル: {n_baked} 個(埋め込み 1 + 各層の o_proj と down_proj)。保存先 `{out_dir}`、{size_gb:.2f} GB。",
             f"tie_word_embeddings={tied}", "",
             "| 対象 | 加工前 拒否率 | 加工後 拒否率 | 加工後の崩壊数 |", "|---|---|---|---|",
             f"| 有害な評価指示 | {before.harmful_rate:.0%} | {after.harmful_rate:.0%} | {after.harmful['崩壊']} |",
             f"| 無害な依頼 | {before.harmless_rate:.0%} | {after.harmless_rate:.0%} | {after.harmless['崩壊']} |", "",
             f"残差に残る u 成分(無害依頼 1 件、|h·u|/|h| の層平均): {before.u_comp:.3f} → {after.u_comp:.4f}", "",
             "## 無害な依頼の例(加工前 → 加工後)", "",
             "依頼の本文は載せない(Alpaca、CC BY-NC 4.0)。番号は評価に使った無害な依頼の並び順。", ""]
    for i, (a, b) in enumerate(zip(before.samples, after.samples), 1):
        lines += [f"- 無害な依頼 {i}", f"  - 前: {a}", f"  - 後: {b}"]
    Path("outputs").mkdir(exist_ok=True)
    tag = model_tag(args.model)
    (Path("outputs") / f"05_bake_{tag}.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote outputs/05_bake_{tag}.md")


def main() -> None:
    args = parse_args()
    tag = model_tag(args.model)
    out_dir = Path(args.out or f"models/{tag}-abliterated")
    d = torch.load(f"outputs/03_refusal_dirs_{tag}.pt")
    layer = args.layer if args.layer is not None else int(d["best_layer"])
    args.n_eval = min(args.n_eval, 16)
    harmful = json.loads(Path("data/harmful.json").read_text())[-args.n_eval:]
    harmless = json.loads(Path("data/harmless.json").read_text())[-args.n_eval:]

    tok = AutoTokenizer.from_pretrained(args.model)
    u = d["unit"][layer].to("mps", torch.float32)

    # 1) 加工前(素の transformers で読む。実験 04 の baseline と同じ値になるはず)
    print(f"model: {args.model}  層 {layer} の方向を使う")
    model = load_model(args.model)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    print(f"tie_word_embeddings={tied}(True なら埋め込みの書き換えが出力側 lm_head にも及ぶ)")
    before = measure(model, tok, u, harmful, harmless, args.max_new)
    print(f"[加工前] 有害拒否率 {before.harmful_rate:.0%} {before.harmful}  無害拒否率 {before.harmless_rate:.0%} {before.harmless}  u成分 {before.u_comp:.3f}")

    # 2) 焼き込んで保存
    n_baked = bake(model, u)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 1e9
    print(f"[焼き込み] {n_baked} 個のテンソルを書き換え → {out_dir}  ({size_gb:.2f} GB)")
    del model; gc.collect(); torch.mps.empty_cache()

    # 3) 保存したものを読み直して確認(hook なし)
    model = load_model(str(out_dir))
    after = measure(model, tok, u, harmful, harmless, args.max_new)
    print(f"[加工後] 有害拒否率 {after.harmful_rate:.0%} {after.harmful}  無害拒否率 {after.harmless_rate:.0%} {after.harmless}  u成分 {after.u_comp:.4f}")
    print("\n[無害な依頼の例] 加工前 → 加工後")
    for p, a, b in zip(harmless[:3], before.samples, after.samples):
        print(f"  依頼: {p[:60]}\n   前: {a!r}\n   後: {b!r}")

    write_report(args, layer, tied, n_baked, out_dir, size_gb, before, after, harmless)


if __name__ == "__main__":
    main()
