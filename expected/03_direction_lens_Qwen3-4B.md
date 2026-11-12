# 実験 03: 方向の logit lens (Qwen/Qwen3-4B)

z = (logit − 語彙全体の平均) / 語彙全体の標準偏差。拒否語 z は下の語の先頭トークンの z(最大 / 平均)。
拒否語: `i can't`, `i cannot`, `i can not`, `i won't`, `i will not`, `i'm not able`, `i am not able`, `i'm sorry`, `i am sorry`, `sorry,`, `i'm unable`, `i am unable`, `as an ai`, `i must decline`, `i'm not going to`, `cannot assist`, `can't help`, `申し訳`, `できません`, `できかね`, `お答えでき`, `お手伝いでき`, `違法`, `許可されていない`, `法的に`, `倫理的に`, `Sorry`, `抱歉`, `cannot`, `CANNOT`, `违法`, `不允许`, `安全`, `合法`

| 層 | 拒否語 z(最大) | 拒否語 z(平均) | 印 | 上位 10 トークン |
|---|---|---|---|---|
| 0 | nan | nan |  | '!', '"', '#', '$', '%', '&', "'", '(', ')', '*' |
| 1 | 2.77 | 1.08 |  | 'chnitt', 'atis', ' Jed', 'iffs', 'abilidade', 'integration', ' cresc', 'г', 'igli', '关键' |
| 2 | 3.07 | -0.29 |  | 'atis', 'fits', 'ifecycle', 'isis', 'benh', 'fühl', '每年', 'ilyn', '降', 'iddleware' |
| 3 | 1.05 | -1.16 |  | 'ruise', 'adoras', 'ointed', 'fühl', 'ishops', 'raisal', '炎', ' 물론', 'PLIC', '重塑' |
| 4 | 1.71 | -1.41 |  | '掌握', 'ymoon', 'ystack', '/company', 'quisites', 'ilyn', 'undy', 'unker', 'onymous', '_),' |
| 5 | 1.13 | -0.69 |  | 'anteed', '可行', 'oola', '可行性', 'ุง', 'ronics', '的大', 'armed', '的技术', '一刀' |
| 6 | 1.31 | -0.50 |  | 'ownt', '!”\n\n', 'upplier', '几千', 'anteed', '婚礼', 'antee', 'ruit', 'whelming', '可行' |
| 7 | 1.22 | -0.64 |  | 'ruit', 'rabbit', 'ุง', ' Filename', '墅', 'ไว', ' fingertips', '/Instruction', 'mlin', '婚礼' |
| 8 | 1.05 | -0.48 |  | '猛', 'ruit', '.ObjectModel', ' студ', '远程', 'ropolis', 'odka', ' mogul', '务', 'uros' |
| 9 | 1.72 | -0.72 |  | 'ippet', 'afi', 'โอกาส', ' elbows', ' heels', '/******/', ' foreigners', '肘', '砧', 'enerative' |
| 10 | 2.72 | -0.01 |  | '髀', ' thinly', ' elbows', 'entic', 'ields', ' recreated', ' студ', '肘', 'itionally', ' đòi' |
| 11 | 1.92 | -0.12 |  | '全流程', ' unsuccessfully', ' sidew', '/Private', 'alse', ' domestically', '低成本', 'canvas', ' unanimously', 'rowsing' |
| 12 | 2.77 | -0.21 |  | 'iced', 'orsi', 'elts', ' inspir', 'enses', 'loit', '刽', 'reeze', ' olduk', 'SelfPermission' |
| 13 | 1.60 | -0.16 |  | 'gran', '申报', '明白', 'outing', 'ocalyptic', '升级', '-funded', '肚子', '而成', '瞑' |
| 14 | 1.99 | -0.37 |  | 'claimer', '.Summary', 'さら', 'ensively', 'ocalyptic', ' đòi', 'aub', ' sab', ' scripted', ' су' |
| 15 | 1.26 | -0.28 |  | ' су', '呸', 'ök', 'ensus', 'gnu', 'entence', 'gx', '.Bundle', 'IFY', '缟' |
| 16 | 1.05 | -0.69 |  | 'eted', '新版', '缦', 'ified', 'IFIED', '罡', ' 自动生成', '最新的', ' breakthrough', ' nutshell' |
| 17 | 1.40 | -0.58 |  | 'icious', 'rement', 'idious', 'URED', '缟', 'ured', '.Pod', '简历', 'iversary', 'itarian' |
| 18 | 1.61 | -0.62 |  | '一套', 'ERY', ']>\n', ' BDSM', '蚀', 'urable', '.Pod', '-covered', 'idious', '一幕' |
| 19 | 6.14 | 0.64 |  | '合法', ' illegal', '诫', '警告', ' criminal', '严禁', '不允许', ' dangerous', '禁止', ' GPLv' |
| 20 | 7.51 | 1.29 |  | '合法', '严禁', '禁止', ' illegal', ' legality', '违法', '不允许', ' prohibited', '不敢', ' ethical' |
| 21 | 4.99 | 0.73 | 候補 | ' illegal', '合法', '违反', '严厉打击', ' legally', 'ammer', ' legality', '违法', '依法追究', '警' |
| 22 | 4.87 | 0.53 |  | '法律责任', '不允许', '没有任何', ' illegal', '禁止', '法律', '严厉打击', 'uka', '不予', ' NOTHING' |
| 23 | 6.21 | 1.08 | 分離度最大 | '禁止', '不允许', ' unauthorized', '严禁', ' illegal', '不应该', '我不是', '拒绝', ' prohibited', '不去' |
| 24 | 6.47 | 1.07 |  | '禁止', ' illegal', '不允许', ' prohibited', ' unethical', '不是一个', '不应该', '严禁', ' unauthorized', '不可以' |
| 25 | 6.80 | 1.15 |  | '不允许', ' unauthorized', '禁止', ' illegal', ' unlawful', ' unethical', '不可以', ' Unauthorized', '我不', '不应该' |
| 26 | 6.53 | 0.87 |  | ' unauthorized', '我不', '不允许', ' unethical', '未经授权', '我不想', ' illegal', ' unlawful', '不符合', ' Unauthorized' |
| 27 | 5.93 | 1.01 |  | ' unauthorized', '不符合', ' unethical', '我不', '非法', '不允许', ' illegal', '未经授权', '我不想', ' Unauthorized' |
| 28 | 5.71 | 1.26 |  | ' unauthorized', ' illegal', ' unethical', '非法', '我不', '不符合', '我不想', '任何形式', ' unlawful', '不允许' |
| 29 | 4.65 | 1.39 |  | ' unethical', '我对', ' unauthorized', ' illegal', ' unlawful', '我不', '非法', '我', '我不是', ' illegally' |
| 30 | 4.77 | 1.36 |  | ' unethical', '我对', '我不', ' unfortunately', '我', '我没有', 'Unfortunately', ' Unfortunately', '任何形式', ' unlawful' |
| 31 | 5.02 | 1.31 |  | ' unethical', '我', '非法', ' unlaw', '我不', '我对', ' unlawful', ' ethical', '对不起', '任何形式' |
| 32 | 4.64 | 1.41 |  | '我', ' unethical', '我不', '我可以', '我没有', '我都', '我对', ' tôi', '我不是', '我才' |
| 33 | 4.37 | 1.39 |  | '我', ' unethical', ' I', '我不', '我可以', '我不是', '我没有', '我还是', '我才', ' creating' |
| 34 | 4.65 | 1.58 |  | '我', ' I', ' unethical', '我没有', '我不是', '我不', '我可以', 'I', ' ethical', ' creating' |
| 35 | 4.31 | 2.12 |  | ' I', '我', 'I', ' unethical', 'Ｉ', ' ethical', '我没有', '非法', ' assisting', ' eth' |
