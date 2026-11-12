"""実験スクリプトで共用する小さな道具(チャット形式の組み立てと Markdown 出力)。

各スクリプトは `sys.path.insert(0, <scripts の場所>)` をしてからこのモジュールを読む。
"""
from pathlib import Path


def chat_prompt(tokenizer, user_text: str) -> str:
    """ユーザーの発言 1 つをチャット形式に包み、返答の直前までの文字列を返す。

    Qwen3 は既定で thinking モード(答える前に長く考える)が入るので、
    実験では enable_thinking=False で切る。返ってくる文字列の末尾は
    「次はアシスタントの番」という印で終わる。
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def model_tag(model_name: str) -> str:
    """'Qwen/Qwen3-1.7B' → 'Qwen3-1.7B'。出力ファイル名や保存先に使う短い名前。"""
    return model_name.split("/")[-1]


def md_cell(text: str) -> str:
    """Markdown の表のセルに入れても壊れないように、改行を空白に、| をエスケープする。"""
    return text.replace("\n", " ").replace("|", "\\|")


def write_markdown(path: str | Path, lines: list[str]) -> Path:
    """行のリストを Markdown ファイルとして書く(親ディレクトリがなければ作る)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
