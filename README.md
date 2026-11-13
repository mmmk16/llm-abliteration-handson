# 作りながら学ぶAbliteratedモデル

このリポジトリは「作りながら学ぶAbliteratedモデル」の実験を行うためのソースコードをまとめています。
各実験の内容と結果の読み方は書籍を参照してください。

## このリポジトリには以下を含みません
- 加工済みモデル
- データセット: 公開データセットから各自の環境で取得をお願いします。
- ベクトルファイル: 3章のスクリプトを動かして作ります。

## 動作環境
- Appleシリコンを積んだMac。筆者は32GBメモリのM1MacBookProで行いました。4Bのモデルを扱う章はメモリを8GBほど使うため、16GB以上のメモリを推奨します。
- Python3.14と[uv](https://docs.astral.sh/uv/)。
- 8章では[Ollama](https://ollama.com/)と、GGUF変換のために[llama.cpp](https://github.com/ggml-org/llama.cpp)のソースを使います。どちらも下の「インストール」で入れます。

## 実行のしかた
- コードブロックはそのままターミナルに貼り付けて実行できます。
- すべてリポジトリ直下から実行してください。
- 結果は `outputs/` に出ます。書籍に載せた実行結果は `expected/` に同じファイル名で置いています。
- 生成は温度0の貪欲法ですが、GPU(MPS)の計算誤差で数値の末尾や文の細部が変わることがあります。
- `HF_HUB_OFFLINE=1` を常に設定している場合は、モデルとデータを取得するときだけ `HF_HUB_OFFLINE=0` を付けてください。

## インストール
1章から8章までに必要なものを、ここでまとめて入れます。

リポジトリを取得して環境を作ります。`--group mlx` は、7章で使うMLXも一緒に入れる指定です。
```sh
git clone https://github.com/mmmk16/llm-abliteration-handson.git
cd llm-abliteration-handson
uv sync --group mlx
```

8章で使うOllamaを入れて起動します。公式サイトのアプリを入れて開いても構いません。
```sh
brew install ollama
brew services start ollama
```

8章で使うllama.cppの変換スクリプトを取得します(ビルドは不要)。
```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
```

動作確認(MPS、transformers、TransformerLens):
```sh
uv run python scripts/check_mps.py
uv run python scripts/generate_hf.py
uv run python scripts/check_transformer_lens.py
```

モデルは初回実行時にHugging Faceから取得します(1.7Bで約4GB、4Bで約8GB)。

## データの取得
3章から6章で使う有害/無害の指示を `data/` に取得します。
```sh
uv run python scripts/03_make_data.py
uv run python scripts/06_make_eval_data.py
```

## 章ごとの実行

### 1章 モデルの大きさと性能
```sh
uv run python scripts/01_compare_models.py
```

### 2章 モデルの内部を覗いてみる
1.7B:
```sh
uv run python scripts/02_logit_lens.py
```
4B:
```sh
uv run python scripts/02_logit_lens.py --model Qwen/Qwen3-4B
```
質問と答えを変える例:
```sh
uv run python scripts/02_logit_lens.py --prompt "日本の首都の名前だけ答えて。" --answer 東京
```

### 3章 モデルの拒否方向を探す
1.7B:
```sh
uv run python scripts/03_refusal_direction.py
```
4B:
```sh
uv run python scripts/03_refusal_direction.py --model Qwen/Qwen3-4B
```

### 4章 モデルの拒否方向を消す、あるいは足す
層スイープのあと、採用層で消す/足すを行います。

1.7B:
```sh
uv run python scripts/04_ablate_direction.py --sweep-layers 16 17 18 19 20 21 22
uv run python scripts/04_ablate_direction.py --layer 19
```
4B:
```sh
uv run python scripts/04_ablate_direction.py --model Qwen/Qwen3-4B --sweep-layers 20 21 22 23
uv run python scripts/04_ablate_direction.py --model Qwen/Qwen3-4B --layer 20
```

### 5章 モデルの重みを書き換えて、拒否方向を消す
加工後のモデルは `models/` に保存されます。

1.7B:
```sh
uv run python scripts/05_bake_weights.py --layer 19
```
4B:
```sh
uv run python scripts/05_bake_weights.py --model Qwen/Qwen3-4B --layer 20
```
加工後のモデルと会話する:
```sh
uv run python scripts/chat_repl.py models/Qwen3-1.7B-abliterated
```

### 6章 モデルの拒否方向を消すと何が失われるか
1.7B:
```sh
uv run python scripts/06_evaluate.py
```
4B:
```sh
uv run python scripts/06_evaluate.py --model Qwen/Qwen3-4B
```

### 7章 Macに最適化されたMLXを試してみる
1.7B:
```sh
uv run --group mlx python scripts/07_mlx_ablate.py --layer 19
```
4B:
```sh
uv run --group mlx python scripts/07_mlx_ablate.py --model Qwen/Qwen3-4B --layer 20
```

### 8章 つくったモデルをOllamaで使いやすくする
5章で作ったモデルと加工前のモデルを、それぞれGGUFに変換してOllamaに登録します。変換には、インストールで取得した `~/llama.cpp` のスクリプトを使います。
ここではオフライン指定(`HF_HUB_OFFLINE=1`)を付けないでください。

#### 4B
変換と登録:
```sh
uv run --with gguf --with sentencepiece --with protobuf python ~/llama.cpp/convert_hf_to_gguf.py \
    models/Qwen3-4B-abliterated --outfile models/Qwen3-4B-abliterated-bf16.gguf --outtype bf16
ollama create qwen3-4b-abliterated -f ollama/Modelfile.qwen3-4b

SRC=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-4B'))")
uv run --with gguf --with sentencepiece --with protobuf python ~/llama.cpp/convert_hf_to_gguf.py \
    "$SRC" --outfile models/Qwen3-4B-original-bf16.gguf --outtype bf16
ollama create qwen3-4b-original -f ollama/Modelfile.qwen3-4b-original
```
比較(Ollamaが起動していること):
```sh
uv run python scripts/08_compare_ollama.py --base qwen3-4b-original --abl qwen3-4b-abliterated
```
会話(thinkingを切るには `/set nothink`):
```sh
ollama run qwen3-4b-abliterated
```

#### 1.7B
変換と登録:
```sh
uv run --with gguf --with sentencepiece --with protobuf python ~/llama.cpp/convert_hf_to_gguf.py \
    models/Qwen3-1.7B-abliterated --outfile models/Qwen3-1.7B-abliterated-bf16.gguf --outtype bf16
ollama create qwen3-1.7b-abliterated -f ollama/Modelfile.qwen3-1.7b

SRC=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-1.7B'))")
uv run --with gguf --with sentencepiece --with protobuf python ~/llama.cpp/convert_hf_to_gguf.py \
    "$SRC" --outfile models/Qwen3-1.7B-original-bf16.gguf --outtype bf16
ollama create qwen3-1.7b-original -f ollama/Modelfile.qwen3-1.7b-original
```
比較(Ollamaが起動していること):
```sh
uv run python scripts/08_compare_ollama.py --base qwen3-1.7b-original --abl qwen3-1.7b-abliterated
```
会話(thinkingを切るには `/set nothink`):
```sh
ollama run qwen3-1.7b-abliterated
```

## ファイル構成
| 場所 | 内容 |
|---|---|
| `scripts/01_*.py` 〜 `scripts/08_*.py` | 各章の実験。先頭のdocstringに目的と使い方を書いています |
| `scripts/common.py` | チャット形式の組み立てとMarkdown出力の共用部品 |
| `scripts/refusal_labels.py` | 生成文に拒否/応答/崩壊のラベルを付ける判定器 |
| `scripts/chat_repl.py` | 加工前後のモデルと会話するREPL |
| `scripts/check_*.py`, `scripts/generate_hf.py` | 環境確認 |
| `ollama/` | 8章のModelfileとOllama公式のqwen3テンプレート |
| `expected/` | 本に載せた実行結果。`outputs/` と見比べるためのもの |
| `outputs/`, `data/`, `models/` | 実行時に作られます。git管理外 |

## データとライセンス
- 有害な指示は[AdvBench](https://github.com/llm-attacks/llm-attacks)(MIT)、無害な指示は[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)(CC BY-NC 4.0)から、3章と6章のスクリプトが各自のマシンに取得します。このリポジトリには含めていません。
- モデルは[Qwen3](https://huggingface.co/Qwen)(Apache 2.0)をHugging Faceから取得します。
- このリポジトリのコードは[MITライセンス](LICENSE)です。

## 安全のために
拒否を外す操作は、安全のための訓練を外す操作でもあります。
4〜8章で有害な評価指示(AdvBenchから取った指示)に対する返答を扱う際は、本文を画面にもファイルにも出さず、ラベルと割合だけを記録します。
1章の鍵の質問(筆者が書いた日本語の質問)は扱いが別で、返答の冒頭を画面に表示し、全文を `outputs/01_compare_models.md` に保存します。`expected/` にもその結果を収録しています。6章では、この質問もラベルだけを記録します。
加工済みのモデルは配布しません。作ったモデルは手元で、モデルの仕組みを理解するためにお使いください。
