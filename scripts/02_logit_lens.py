"""実験 02: logit lens。各層の残差ストリームに最終層の読み出し(正規化 + unembed)を当て、
「その層の時点でモデルが次に出そうとしている単語」を層ごとに並べる。

使い方:
    uv run python scripts/02_logit_lens.py                                   # 1.7B、answer モード、答え「富士山」
    uv run python scripts/02_logit_lens.py --model Qwen/Qwen3-4B
    uv run python scripts/02_logit_lens.py --prompt "日本の首都の名前だけ答えて。" --answer 東京
    uv run python scripts/02_logit_lens.py --mode raw --prompt "The highest mountain in Japan is Mount"
    uv run python scripts/02_logit_lens.py --mode chat                       # 返答の最初の1語を見る

モード(lens を当てる位置の決め方):
    answer: (既定) チャット形式で返答を生成し、返答の中で答えの最初のトークンが出る直前の位置に lens を当てる。
            モデルが「日本で一番高い山は…」と復唱から始めても、答えを言おうとする瞬間を見られる。
    chat:   チャット形式で、返答の最初の1トークンを見る。
    raw:    文の続きをそのまま予測させる(続きの1語が答えになる文が向く)。
結果(<tag> は Qwen3-1.7B のようなモデル名): 画面の表、outputs/02_logit_lens_<tag>.md、outputs/02_logit_lens_<tag>.png
--prompt / --answer / --mode を変えたときは outputs/02_logit_lens_<tag>_custom.md / .png に書き、既定の質問の結果を上書きしない。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import TransformerBridge

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ 直下の共用モジュールを読む
from common import chat_prompt, md_cell, model_tag

TOP_K = 3  # 各層で表示する上位トークン数
OUT_DIR = Path("outputs")
DEFAULT_PROMPT = "日本で一番高い山の名前だけ答えて。"
DEFAULT_ANSWER = "富士山"
DEFAULT_MODE = "answer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--mode", choices=["answer", "chat", "raw"], default=DEFAULT_MODE)
    p.add_argument("--answer", default=DEFAULT_ANSWER, help="answer モードで探す答え(最初のトークンだけ使う)")
    p.add_argument("--max-new", type=int, default=40, help="answer モードで生成する最大トークン数")
    return p.parse_args()


def out_stem(args: argparse.Namespace) -> str:
    """出力ファイル名(拡張子なし)。質問・答え・モードを変えた実行は、既定の質問の結果を上書きしないよう _custom を付ける。"""
    custom = (args.prompt, args.answer, args.mode) != (DEFAULT_PROMPT, DEFAULT_ANSWER, DEFAULT_MODE)
    return f"02_logit_lens_{model_tag(args.model)}" + ("_custom" if custom else "")


def build_input(model: TransformerBridge, args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    """lens を当てる位置が末尾になるようにトークン列を作る。戻り値は (tokens, 位置の説明)。"""
    if args.mode == "raw":
        return model.to_tokens(args.prompt), "文の続き"

    prompt_tokens = model.to_tokens(chat_prompt(model.tokenizer, args.prompt))
    if args.mode == "chat":
        return prompt_tokens, "返答の最初の1語"

    # answer モード: 返答を生成し、答えの最初のトークンが出る直前までを入力にする
    with torch.no_grad():
        generated = model.generate(prompt_tokens, max_new_tokens=args.max_new, do_sample=False, verbose=False)
    reply_ids = generated[0, prompt_tokens.shape[1]:].tolist()
    reply = model.tokenizer.decode(reply_ids, skip_special_tokens=True)
    print(f"generated reply: {reply!r}")

    # 「富士山」は「富」「士」「山」に割れるので、探すのは最初のトークン「富」
    target_id = model.tokenizer(args.answer, add_special_tokens=False)["input_ids"][0]
    target_str = model.tokenizer.decode([target_id])
    if target_id not in reply_ids:
        print(f"答え {args.answer!r} の最初のトークン {target_str!r} が返答に出てこない。返答の最初の1語で代用する。")
        return prompt_tokens, "返答の最初の1語(答えが見つからず)"
    pos = reply_ids.index(target_id)
    tokens = generated[:, : prompt_tokens.shape[1] + pos]
    print(f"答えのトークン {target_str!r} は返答の {pos} 番目。その直前までの {tokens.shape[1]} トークンを入力にする。")
    return tokens, f"{target_str!r} の直前"


def lens_at_layer(model: TransformerBridge, cache, layer: int, answer_id: int) -> tuple[str, float]:
    """層 layer の出口の残差(末尾位置)に最終層の読み出しを当て、(top-k の文字列, 答えの確率) を返す。"""
    resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]                          # (d_model,)
    probs = model.unembed(model.ln_final(resid[None, None]))[0, 0].float().softmax(-1)  # 語彙ぶんの確率
    top = probs.topk(TOP_K)
    top_str = ", ".join(f"{model.tokenizer.decode([int(i)])!r} {p:.2f}" for p, i in zip(top.values, top.indices))
    return top_str, float(probs[answer_id])


def write_table(args, where: str, str_tokens: list[str], answer: str, p_final: float, rows: list) -> None:
    """層ごとの表を Markdown で保存する。"""
    lines = [f"# 実験 02: logit lens ({args.model})", "", f"prompt: `{args.prompt}`  mode: {args.mode}  lens 位置: {where}",
             f"入力の末尾: {str_tokens[-8:]}", f"最終層の予測: `{answer}` (p={p_final:.3f})", "",
             f"| 層 | p(予測) | top{TOP_K} |", "|---|---|---|"]
    lines += [f"| {layer} | {p:.3f} | {md_cell(top)} |" for layer, top, p in rows]
    (OUT_DIR / f"{out_stem(args)}.md").write_text("\n".join(lines) + "\n")


def plot(args, n_layers: int, answer: str, answer_p: list[float]) -> None:
    """層ごとの「答えの確率」を折れ線で描く。"""
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]  # macOS の日本語フォント
    plt.figure(figsize=(8, 4))
    plt.plot(range(n_layers), answer_p, marker="o")
    plt.xlabel("層"); plt.ylabel(f"p({answer!r})"); plt.title(f"logit lens: {args.prompt} [{args.mode}] ({args.model})")
    plt.ylim(0, 1); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_stem(args)}.png", dpi=120)


def main() -> None:
    args = parse_args()
    model = TransformerBridge.boot_transformers(args.model, device="mps", dtype=torch.bfloat16)
    tokens, where = build_input(model, args)
    logits, cache = model.run_with_cache(tokens)  # 全層の残差をまとめて取る
    str_tokens = model.to_str_tokens(tokens[0])

    # 最終層の予測(= 普通に生成したときの次のトークン)を「答え」とし、各層でその確率を追う
    final_probs = logits[0, -1].float().softmax(-1)
    answer_id = int(final_probs.argmax())
    answer = model.tokenizer.decode([answer_id])
    print(f"\nmodel: {args.model}  layers: {model.cfg.n_layers}  mode: {args.mode}  lens 位置: {where}")
    print(f"入力の末尾: {str_tokens[-8:]}")
    print(f"最終層の予測: {answer!r} (p={final_probs[answer_id]:.3f})\n")

    rows, answer_p = [], []
    for layer in range(model.cfg.n_layers):
        top_str, p_ans = lens_at_layer(model, cache, layer, answer_id)
        answer_p.append(p_ans)
        rows.append((layer, top_str, p_ans))
        print(f"layer {layer:2d}  p({answer!r})={p_ans:.3f}  top{TOP_K}: {top_str}")

    OUT_DIR.mkdir(exist_ok=True)
    write_table(args, where, str_tokens, answer, float(final_probs[answer_id]), rows)
    plot(args, model.cfg.n_layers, answer, answer_p)
    print(f"\nwrote outputs/{out_stem(args)}.md and {out_stem(args)}.png")


if __name__ == "__main__":
    main()
