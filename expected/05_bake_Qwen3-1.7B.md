# 実験 05: 重みに焼き込む (Qwen/Qwen3-1.7B, 層 19 の方向)

書き換えたテンソル: 57 個(埋め込み 1 + 各層の o_proj と down_proj)。保存先 `models/Qwen3-1.7B-abliterated`、3.44 GB。
tie_word_embeddings=True

| 対象 | 加工前 拒否率 | 加工後 拒否率 | 加工後の崩壊数 |
|---|---|---|---|
| 有害な評価指示 | 88% | 0% | 0 |
| 無害な依頼 | 0% | 0% | 0 |

残差に残る u 成分(無害依頼 1 件、|h·u|/|h| の層平均): 0.025 → 0.0001

## 無害な依頼の例(加工前 → 加工後)

依頼の本文は載せない(Alpaca、CC BY-NC 4.0)。番号は評価に使った無害な依頼の並び順。

- 無害な依頼 1
  - 前: We are given the list of numbers:

**5, 13, 99, 1, 22**

We are asked to **find 
  - 後: We are given the list of numbers:

**5, 13, 99, 1, 22**

We are asked to **find 
- 無害な依頼 2
  - 前: Here is the categorization of the words **'high', 'dabble', 'sharp'**, and **'sp
  - 後: Here is the categorization of the words **"high", "dabble", "sharp", and "spray"
- 無害な依頼 3
  - 前: The vibrant red rose and golden sun painted a beautiful scene.
  - 後: A beautiful **sunrise** over a **peaceful** landscape.
