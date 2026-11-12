"""実験 03: 拒否方向を求める。

有害な指示と無害な指示をそれぞれチャット形式に包んでモデルに通し、
「返答を始める直前の位置」の残差ストリーム(各層の resid_pre)を集める。
層ごとに「有害の平均 − 無害の平均」を取ったベクトルが拒否方向の候補。

各クラスの末尾 N_TEST 件は方向の計算に使わず、「初めて見る指示も分けられるか」の確認に使う。

使い方:
    uv run python scripts/03_refusal_direction.py                       # 1.7B、data/ の全件
    uv run python scripts/03_refusal_direction.py --model Qwen/Qwen3-4B
    uv run python scripts/03_refusal_direction.py --n 32                # 各クラス 32 件だけ使う
出力(<tag> は Qwen3-1.7B のようなモデル名):
    outputs/03_refusal_dirs_<tag>.pt        全層の方向ベクトル(あとの実験で使う)。sweep_range も入る
    outputs/03_refusal_direction_<tag>.md   層ごとの表(候補層とスイープ範囲つき。候補層の方向との余弦類似度も入る)
    outputs/03_refusal_direction_<tag>.png  層ごとのグラフ
    outputs/03_projection_hist_<tag>.png    候補層の方向への射影のヒストグラム(有害/無害)
    outputs/03_direction_lens_<tag>.md      各層の方向に logit lens を当てた表(上位 10 語と拒否語の z スコア)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from transformer_lens import TransformerBridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chat_prompt, md_cell, model_tag
from refusal_labels import REFUSAL_MARKERS

N_TEST = 16  # 各クラスの末尾 16 件は方向の計算に使わず、分離の確認に使う
OUT_DIR = Path("outputs")
LENS_TOP_K = 10  # 方向の logit lens で表に出す上位トークン数
# 方向の logit lens で z スコアを見る「拒否語」。判定用の定型句に、中国語・英語の言い回しを足したもの(重複は除く)
LENS_REFUSAL_WORDS = list(dict.fromkeys(
    REFUSAL_MARKERS + ["Sorry", "抱歉", "cannot", "CANNOT", "违法", "不允许", "安全", "合法", "申し訳"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--n", type=int, default=None, help="各クラスで使う件数(既定は全件)")
    return p.parse_args()


def collect(model: TransformerBridge, prompts: list[str]) -> torch.Tensor:
    """各プロンプトについて、末尾位置の resid_pre を全層ぶん集める → (n_prompts, n_layers, d_model)"""
    hook_names = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers)]
    rows = []
    for k, prompt in enumerate(prompts):
        text = chat_prompt(model.tokenizer, prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(text, names_filter=lambda name: name.endswith("hook_resid_pre"))
        rows.append(torch.stack([cache[h][0, -1].float().cpu() for h in hook_names]))
        if (k + 1) % 16 == 0:
            print(f"  {k + 1}/{len(prompts)}")
    return torch.stack(rows)


def layer_metrics(a_tr, b_tr, a_te, b_te, dirs, unit) -> dict[str, torch.Tensor]:
    """層ごとに、方向の良さを見る指標を計算する。各値の形は (n_layers,)。

    - norm: 差ベクトルの長さ。深い層ほど残差が大きくなるので単体では比べにくい
    - rel : norm ÷ その層の残差の典型的な長さ。「残差のうちどれだけが有害/無害の違いに使われているか」
    - sep : 訓練データを方向に射影したときの、クラス間の距離 ÷ クラス内のばらつき(分離度)
    - acc : 確認用データを射影し、訓練データの両クラス平均の中点で分けたときの正答率
    """
    norm = dirs.norm(dim=-1)
    resid_scale = torch.cat([a_tr, b_tr]).norm(dim=-1).mean(0)
    rel = norm / resid_scale
    # 射影 = 方向 u との内積
    proj_a_tr = (a_tr * unit).sum(-1); proj_b_tr = (b_tr * unit).sum(-1)
    mid = (proj_a_tr.mean(0) + proj_b_tr.mean(0)) / 2
    proj_a_te = (a_te * unit).sum(-1); proj_b_te = (b_te * unit).sum(-1)
    acc = ((proj_a_te > mid).float().mean(0) + (proj_b_te < mid).float().mean(0)) / 2
    sep = (proj_a_tr.mean(0) - proj_b_tr.mean(0)) / (proj_a_tr.std(0) + proj_b_tr.std(0))
    sep = torch.nan_to_num(sep)  # 層 0 は末尾トークンの埋め込みそのもので全件同じなので差が 0 になり nan が出る
    return {"norm": norm, "rel": rel, "sep": sep, "acc": acc}


def choose_candidate_layer(acc, rel, n_layers: int) -> int:
    """候補層: 確認正答率が最大に達した層のうち、|差|/|残差| (rel) が最初に山を作る層。

    候補集合は range(2, n_layers-2) のうち acc が最大(許容 1e-6)の層。その中で
    rel[i] >= rel[i-1] かつ rel[i] > rel[i+1](隣は全層の配列から見る。左は ≧、右は >)を満たす最初の i を返す。
    見つからなければ候補集合の先頭。

    分離度は言葉に近い深い層ほど伸びるので、分離度最大で選ぶと出力寄りの深い層に引かれる。
    消したいのは言葉でなく、その手前にある判断なので、確認正答率が最大に達した後に |差|/|残差| が最初に山を作る層
    (方向がまず固まる層)を採る。あくまで目安で、実際に使う層は実験 04 で消してみて決める。
    """
    top = max(float(acc[i]) for i in range(2, n_layers - 2))
    cand = [i for i in range(2, n_layers - 2) if float(acc[i]) >= top - 1e-6]
    for i in cand:
        if float(rel[i]) >= float(rel[i - 1]) and float(rel[i]) > float(rel[i + 1]):
            return i
    return cand[0]


def choose_sweep_range(acc, best: int, n_layers: int) -> list[int]:
    """実験 04 でスイープする層の範囲: 確認正答率が最大になる最初の層 〜 候補層 + 2(上限は最終層)。"""
    top = max(float(acc[i]) for i in range(2, n_layers - 2))
    first = next(i for i in range(2, n_layers - 2) if float(acc[i]) >= top - 1e-6)
    return list(range(first, min(best + 2, n_layers - 1) + 1))


def direction_lens(model, unit: torch.Tensor, n_layers: int) -> list[dict]:
    """各層の単位方向 u に最終層の読み出し(ln_final + unembed。02 の lens_at_layer と同じ経路)を当てる。

    RMSNorm なので u の長さは効かない。戻り値は層ごとの {top: 上位語の文字列, zmax, zmean}。
    z = (logit - 語彙全体の平均) / 語彙全体の標準偏差。拒否語は LENS_REFUSAL_WORDS の各語を
    tokenizer.encode(word, add_special_tokens=False) した先頭トークンで見る。
    """
    tok = model.tokenizer
    ref = next(model.parameters())
    word_ids = [tok.encode(w, add_special_tokens=False)[0] for w in LENS_REFUSAL_WORDS]
    rows = []
    for i in range(n_layers):
        u = unit[i].to(ref.device, ref.dtype)
        with torch.no_grad():
            logits = model.unembed(model.ln_final(u[None, None]))[0, 0].float()
        z = (logits - logits.mean()) / logits.std()
        zw = z[word_ids]
        top = logits.topk(LENS_TOP_K).indices.tolist()
        rows.append({"top": ", ".join(repr(tok.decode([t])) for t in top),
                     "zmax": float(zw.max()), "zmean": float(zw.mean())})
    return rows


def lens_marks(i: int, best: int, sep_best: int) -> str:
    """表の印の列: 候補層 = 候補、分離度最大層 = 分離度最大(両方なら併記)。"""
    return " ".join(m for m, hit in (("候補", i == best), ("分離度最大", i == sep_best)) if hit)


def write_lens_table(args, tag: str, lens: list[dict], best: int, sep_best: int) -> None:
    """方向の logit lens を Markdown の表にして保存する。"""
    lines = [f"# 実験 03: 方向の logit lens ({args.model})", "",
             "z = (logit − 語彙全体の平均) / 語彙全体の標準偏差。拒否語 z は下の語の先頭トークンの z(最大 / 平均)。",
             "拒否語: " + ", ".join(f"`{w}`" for w in LENS_REFUSAL_WORDS), "",
             f"| 層 | 拒否語 z(最大) | 拒否語 z(平均) | 印 | 上位 {LENS_TOP_K} トークン |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {r['zmax']:.2f} | {r['zmean']:.2f} | {lens_marks(i, best, sep_best)} | {md_cell(r['top'])} |"
              for i, r in enumerate(lens)]
    (OUT_DIR / f"03_direction_lens_{tag}.md").write_text("\n".join(lines) + "\n")


def write_table(args, tag: str, n_harmful: int, n_harmless: int, best: int, sweep: list[int], m: dict, cos, n_layers: int) -> None:
    """層ごとの指標を Markdown の表にして保存する。cos は候補層の方向との余弦類似度(1 なら同じ向き、0 なら直交)。"""
    lines = [f"# 実験 03: 拒否方向 ({args.model})", "",
             f"harmful {n_harmful} 件 / harmless {n_harmless} 件(末尾 {N_TEST} 件ずつは確認用)。候補層: **{best}**。スイープ範囲: **{sweep[0]}〜{sweep[-1]}**", "",
             "| 層 | \\|差\\| | \\|差\\|/\\|残差\\| | 分離度 | 確認正答率 | 候補層との余弦類似度 |", "|---|---|---|---|---|---|"]
    lines += [f"| {i} | {m['norm'][i]:.2f} | {m['rel'][i]:.3f} | {m['sep'][i]:.2f} | {m['acc'][i]:.2f} | {cos[i]:.2f} |" for i in range(n_layers)]
    (OUT_DIR / f"03_refusal_direction_{tag}.md").write_text("\n".join(lines) + "\n")


def plot(args, tag: str, best: int, sweep: list[int], m: dict) -> None:
    """rel / sep / acc を層ごとに 3 枚並べて描く。点線が候補層、薄い帯がスイープ範囲。"""
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(m["rel"], marker="o"); ax[0].set_title("|有害平均 − 無害平均| / |残差|"); ax[0].set_xlabel("層")
    ax[1].plot(m["sep"], marker="o"); ax[1].set_title("分離度(クラス間距離 / ばらつき)"); ax[1].set_xlabel("層")
    ax[2].plot(m["acc"], marker="o"); ax[2].set_ylim(0, 1.05); ax[2].set_title("確認データの正答率"); ax[2].set_xlabel("層")
    for a_ in ax:
        a_.axvspan(sweep[0] - 0.5, sweep[-1] + 0.5, color="gray", alpha=0.12)
        a_.axvline(best, color="gray", ls="--", alpha=0.6); a_.grid(alpha=0.3)
    fig.suptitle(f"拒否方向の候補 ({args.model})"); fig.tight_layout()
    fig.savefig(OUT_DIR / f"03_refusal_direction_{tag}.png", dpi=120)


def plot_projection_hist(args, tag: str, best: int, a_tr, b_tr, a_te, b_te, unit) -> None:
    """候補層の残差を候補層の方向 u に射影し、有害(赤)と無害(青)のヒストグラムにする。

    棒は全件、三角は方向の計算に使わなかった確認用の件。点線は判定に使う中点(訓練データの両クラス平均の真ん中)。
    """
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    u = unit[best]
    pa, pb, pa_te, pb_te = ((x[:, best] @ u).tolist() for x in (a_tr, b_tr, a_te, b_te))  # 射影 = 内積 h·u
    mid = (sum(pa) / len(pa) + sum(pb) / len(pb)) / 2
    every = pa + pb + pa_te + pb_te
    span = (min(every), max(every))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(pb + pb_te, bins=30, range=span, color="tab:blue", alpha=0.6, label="無害")
    ax.hist(pa + pa_te, bins=30, range=span, color="tab:red", alpha=0.6, label="有害")
    ax.plot(pb_te, [-0.6] * len(pb_te), "^", color="tab:blue", label="無害(確認用)")
    ax.plot(pa_te, [-0.6] * len(pa_te), "^", color="tab:red", label="有害(確認用)")
    ax.axvline(mid, color="gray", ls="--", label="中点")
    ax.set_xlabel(f"層 {best} の方向 u への射影 h·u"); ax.set_ylabel("件数")
    ax.set_title(f"拒否方向への射影 ({args.model}, 候補層 {best})"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"03_projection_hist_{tag}.png", dpi=120)


def main() -> None:
    args = parse_args()
    harmful = json.loads(Path("data/harmful.json").read_text())
    harmless = json.loads(Path("data/harmless.json").read_text())
    if args.n:
        harmful, harmless = harmful[: args.n], harmless[: args.n]
    assert len(harmful) > N_TEST and len(harmless) > N_TEST, "各クラス 17 件以上必要"
    print(f"model: {args.model}  harmful: {len(harmful)}  harmless: {len(harmless)}  (末尾 {N_TEST} 件ずつは確認用)")

    # 1) 残差を集める
    model = TransformerBridge.boot_transformers(args.model, device="mps", dtype=torch.bfloat16)
    print("collecting harmful ...");  a = collect(model, harmful)
    print("collecting harmless ..."); b = collect(model, harmless)

    # 2) 訓練用と確認用に分け、層ごとの「有害平均 − 無害平均」を方向の候補にする
    a_tr, a_te = a[:-N_TEST], a[-N_TEST:]
    b_tr, b_te = b[:-N_TEST], b[-N_TEST:]
    dirs = a_tr.mean(0) - b_tr.mean(0)                     # (n_layers, d_model) 拒否方向の候補
    unit = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # 長さ 1 にした方向 u

    # 3) 層ごとの指標と候補層(候補層は |差|/|残差| の最初の山。分離度最大層は対比用)
    n_layers = model.cfg.n_layers
    m = layer_metrics(a_tr, b_tr, a_te, b_te, dirs, unit)
    print(f"\n{'層':>3} {'|差|':>8} {'|差|/|残差|':>10} {'分離度':>7} {'確認正答率':>9}")
    for i in range(n_layers):
        print(f"{i:3d} {m['norm'][i]:8.2f} {m['rel'][i]:10.3f} {m['sep'][i]:7.2f} {m['acc'][i]:9.2f}")
    best = choose_candidate_layer(m["acc"], m["rel"], n_layers)
    sweep = choose_sweep_range(m["acc"], best, n_layers)
    sep_best = max((i for i in range(2, n_layers - 2) if float(m["acc"][i]) >= float(m["acc"][2:-2].max()) - 1e-6),
                   key=lambda i: float(m["sep"][i]))  # 分離度が最大の層。参考として表に印を付けるだけで、本では使わない
    print(f"\n候補層: {best}  (|差|={m['norm'][best]:.2f}, 分離度={m['sep'][best]:.2f}, 確認正答率={m['acc'][best]:.2f})")
    print(f"分離度最大層: {sep_best}")
    print(f"スイープ範囲: {sweep[0]}〜{sweep[-1]}")

    # 4) 方向の logit lens(全層。1 層 1 回の行列積なので安価)
    tag = model_tag(args.model)
    OUT_DIR.mkdir(exist_ok=True)
    lens = direction_lens(model, unit, n_layers)
    write_lens_table(args, tag, lens, best, sep_best)
    print(f"\n方向の logit lens(拒否語 z の最大/平均。上位 {LENS_TOP_K} 語は outputs/03_direction_lens_{tag}.md)")
    for name, i in (("候補層", best), ("分離度最大層", sep_best), ("最終層", n_layers - 1)):
        print(f"  {name} {i:2d}: z最大={lens[i]['zmax']:.2f} z平均={lens[i]['zmean']:.2f}  {lens[i]['top'][:60]}")

    # 5) 保存(方向ベクトルは実験 04〜07 で使う)
    torch.save({"model": args.model, "dirs": dirs, "unit": unit, "best_layer": best, "sweep_range": sweep,
                "norm": m["norm"], "rel": m["rel"], "sep": m["sep"], "acc": m["acc"],
                "n_harmful": len(harmful), "n_harmless": len(harmless)}, OUT_DIR / f"03_refusal_dirs_{tag}.pt")
    cos = unit @ unit[best]  # 各層の方向と候補層の方向の余弦類似度(どちらも長さ 1 なので内積がそのまま余弦)
    write_table(args, tag, len(harmful), len(harmless), best, sweep, m, cos, n_layers)
    plot(args, tag, best, sweep, m)
    plot_projection_hist(args, tag, best, a_tr, b_tr, a_te, b_te, unit)
    print(f"wrote outputs/03_refusal_dirs_{tag}.pt, 03_refusal_direction_{tag}.md, 03_refusal_direction_{tag}.png, "
          f"03_projection_hist_{tag}.png, 03_direction_lens_{tag}.md")


if __name__ == "__main__":
    main()
