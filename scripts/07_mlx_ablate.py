"""実験 07: 実験 04(推論時に消す)と 05(重みに焼き込む)を MLX で手書きし直す。

TransformerLens の hook が裏でやっていたことを自分で書く。MLX には hook がないので、
mlx-lm の Model.__call__(埋め込み → 各層 → 正規化 → unembed)を展開し、
残差ストリーム h を手で受け渡しながら、好きな場所で拒否方向 u の成分を引く。
    1. 手書きループ(素)            → 実験 04 の baseline と同じ拒否率になるはず
    2. 手書きループ + ablate        → 実験 04 の ablate と同じ拒否率(0 付近)になるはず
    3. MLX で重みに焼き込み、保存、読み直して標準の forward で実行 → 実験 05 と同じになるはず
    4. 実験 05 で PyTorch が焼いたモデルを MLX で読み込む → 同じ重みなので同じになるはず
あわせて生成速度(tok/s)とピークメモリを記録する。

安全のための方針は 04・05 と同じ。有害指示はラベルと割合のみ、本文は表示も保存もしない。

使い方:
    uv run --group mlx python scripts/07_mlx_ablate.py --layer 19                          # 1.7B
    uv run --group mlx python scripts/07_mlx_ablate.py --model Qwen/Qwen3-4B --layer 20
出力: models/<model>-abliterated-mlx/(MLX で焼いたモデル)、outputs/07_mlx_<model>.md
"""
import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import torch  # 実験 03 の .pt を読むためだけに使う
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import save_config, save_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chat_prompt, model_tag
from refusal_labels import LABELS, classify, count_labels, refusal_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=None, help="使う方向の層(実験 04 の採用層)")
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--max-new", type=int, default=40)
    return p.parse_args()


def forward(model, tokens, cache, ablate=None):
    """mlx-lm の Model.__call__ を展開したもの。ablate を渡すと残差の各点(pre/mid/post)で適用する。

    h が残差ストリーム。各層は h に attention と MLP の出力を足すだけで、
    TransformerLens の hook_resid_{pre,mid,post} は f(...) を当てている場所に相当する。
    """
    f = ablate or (lambda h: h)
    m = model.model
    h = f(m.embed_tokens(tokens))                                         # 層 0 の resid_pre
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(m.layers, cache):
        h = f(h + layer.self_attn(layer.input_layernorm(h), mask, c))     # resid_mid
        h = f(h + layer.mlp(layer.post_attention_layernorm(h)))           # resid_post(= 次の層の resid_pre)
    h = m.norm(h)
    return m.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)


def greedy(model, tok, prompt: str, max_new: int, step) -> tuple[str, int]:
    """KV キャッシュ付きの貪欲生成を手で書く。step(tokens, cache) -> logits。戻り値は (返答, 生成トークン数)。"""
    text = chat_prompt(tok, prompt)
    y = mx.array(tok.encode(text))[None]
    cache = make_prompt_cache(model)
    eos = getattr(tok, "eos_token_ids", None) or {tok.eos_token_id}
    out = []
    for _ in range(max_new):
        logits = step(y, cache)
        nxt = mx.argmax(logits[0, -1]).item()      # .item() で遅延評価を確定させる
        if nxt in eos:
            break
        out.append(nxt)
        y = mx.array([[nxt]])
    return tok.decode(out), len(out)


def bake(model, u: mx.array) -> int:
    """残差に書き込む重みから u 成分を除く(実験 05 と同じ式)。書き換えたテンソルの数を返す。"""
    m = model.model
    dt = m.embed_tokens.weight.dtype
    w = m.embed_tokens.weight.astype(mx.float32)
    m.embed_tokens.weight = (w - mx.outer(w @ u, u)).astype(dt)
    mx.eval(m.embed_tokens.weight)        # 遅延評価: テンソルごとに確定させ、fp32 の中間を溜めない
    n = 1
    for layer in m.layers:
        for lin in (layer.self_attn.o_proj, layer.mlp.down_proj):
            w = lin.weight.astype(mx.float32)
            lin.weight = (w - mx.outer(u, u @ w)).astype(dt)
            mx.eval(lin.weight)
            n += 1
    return n


@dataclass
class Row:
    """結果の表の 1 行 = 1 方式。"""
    name: str
    harmful_rate: float
    harmless_rate: float
    harmless_collapse: int
    tps: float
    peak_gb: float


def measure(name: str, model, tok, step, harmful: list[str], harmless: list[str], max_new: int) -> Row:
    """1 方式について、拒否率(有害はラベルのみ)、無害依頼での tok/s、ピークメモリを測る。"""
    mx.reset_peak_memory()
    c = count_labels(greedy(model, tok, p, max_new, step)[0] for p in harmful)
    g = {label: 0 for label in LABELS}  # 無害側は tok/s も数えるので手で回す
    n_tok, t0 = 0, time.perf_counter()
    for p in harmless:
        text, n = greedy(model, tok, p, max_new, step)
        g[classify(text)] += 1
        n_tok += n
    tps = n_tok / (time.perf_counter() - t0)
    peak = mx.get_peak_memory() / 1e9
    print(f"[{name}] 有害拒否率 {refusal_rate(c):.0%} {c}  無害拒否率 {refusal_rate(g):.0%} {g}  {tps:.1f} tok/s  peak {peak:.2f} GB")
    return Row(name, refusal_rate(c), refusal_rate(g), g["崩壊"], tps, peak)


def check_forward_matches_standard(model, tok, prompt: str) -> float:
    """手書きループと mlx-lm 標準 forward の logits の最大差を返す(0 なら展開が正しい)。"""
    y = mx.array(tok.encode(chat_prompt(tok, prompt)))[None]
    return mx.abs(forward(model, y, make_prompt_cache(model)) - model(y, cache=make_prompt_cache(model))).max().item()


def write_report(args, layer: int, out_dir: Path, size: float, n_baked: int, rows: list[Row]) -> None:
    """4 方式の結果を Markdown の表にして保存する。"""
    tag = model_tag(args.model)
    lines = [f"# 実験 07: MLX で再実装 ({args.model}, 層 {layer})", "",
             f"MLX で焼いたモデル: `{out_dir}` ({size:.2f} GB、{n_baked} テンソル)", "",
             "| 方式 | 有害拒否率 | 無害拒否率 | 無害崩壊 | tok/s | ピークメモリ GB |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.name} | {r.harmful_rate:.0%} | {r.harmless_rate:.0%} | {r.harmless_collapse} | {r.tps:.1f} | {r.peak_gb:.2f} |" for r in rows]
    Path("outputs").mkdir(exist_ok=True)
    (Path("outputs") / f"07_mlx_{tag}.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote outputs/07_mlx_{tag}.md")


def main() -> None:
    args = parse_args()
    tag = model_tag(args.model)
    d = torch.load(f"outputs/03_refusal_dirs_{tag}.pt")
    layer = args.layer if args.layer is not None else int(d["best_layer"])
    u32 = mx.array(d["unit"][layer].numpy())
    harmful = json.loads(Path("data/harmful.json").read_text())[-args.n_eval:]
    harmless = json.loads(Path("data/harmless.json").read_text())[-args.n_eval:]
    rows = []

    # 1, 2) 元のモデルを手書きループで。u は bf16 に揃えて残差の dtype を変えない
    model, tok, config = load(args.model, return_config=True)
    u = u32.astype(model.model.embed_tokens.weight.dtype)
    ablate = lambda h: h - (h @ u)[..., None] * u
    print(f"model: {args.model}  層 {layer}  dtype {u.dtype}")
    diff = check_forward_matches_standard(model, tok, harmless[0])
    print(f"手書きループ vs 標準 forward の logits 最大差: {diff:.2e}(0 なら展開が正しい)")
    rows.append(measure("MLX 手書き(素)", model, tok, lambda y, c: forward(model, y, c), harmful, harmless, args.max_new))
    rows.append(measure("MLX 手書き + ablate", model, tok, lambda y, c: forward(model, y, c, ablate), harmful, harmless, args.max_new))

    # 3) MLX で焼き込んで保存 → 読み直して標準 forward(hook も手書きループも不要)
    n_baked = bake(model, u32)
    out_dir = Path(f"models/{tag}-abliterated-mlx")
    save_model(out_dir, model)
    save_config(config, out_dir / "config.json")   # save_model は重みだけなので設定とトークナイザは別に保存する
    tok.save_pretrained(out_dir)
    size = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 1e9
    print(f"[焼き込み] {n_baked} 個のテンソルを書き換え → {out_dir} ({size:.2f} GB)")
    del model; gc.collect(); mx.clear_cache()
    model, tok = load(str(out_dir))
    rows.append(measure("MLX 焼き込み(標準 forward)", model, tok, lambda y, c: model(y, cache=c), harmful, harmless, args.max_new))
    print("\n[標準 API] mlx_lm.generate で無害な依頼:")
    print(" ", mlx_generate(model, tok, prompt=chat_prompt(tok, harmless[0]), max_tokens=60).replace("\n", " ")[:160])
    del model; gc.collect(); mx.clear_cache()

    # 4) 実験 05 で PyTorch が焼いたモデルを MLX で読む
    torch_dir = Path(f"models/{tag}-abliterated")
    if torch_dir.exists():
        model, tok = load(str(torch_dir))
        rows.append(measure("PyTorch 焼き込み(05)を MLX で読込", model, tok, lambda y, c: model(y, cache=c), harmful, harmless, args.max_new))

    write_report(args, layer, out_dir, size, n_baked, rows)


if __name__ == "__main__":
    main()
