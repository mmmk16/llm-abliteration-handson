# 実験 03: 方向の logit lens (Qwen/Qwen3-4B)

| 層 | 印 | 上位 10 トークン |
|---|---|---|
| 0 |  | '!', '"', '#', '$', '%', '&', "'", '(', ')', '*' |
| 1 |  | 'chnitt', 'atis', ' Jed', 'iffs', 'abilidade', 'integration', ' cresc', 'г', 'igli', '关键' |
| 2 |  | 'atis', 'fits', 'ifecycle', 'isis', 'benh', 'fühl', '每年', 'ilyn', '降', 'iddleware' |
| 3 |  | 'ruise', 'adoras', 'ointed', 'fühl', 'ishops', 'raisal', '炎', ' 물론', 'PLIC', '重塑' |
| 4 |  | '掌握', 'ymoon', 'ystack', '/company', 'quisites', 'ilyn', 'undy', 'unker', 'onymous', '_),' |
| 5 |  | 'anteed', '可行', 'oola', '可行性', 'ุง', 'ronics', '的大', 'armed', '的技术', '一刀' |
| 6 |  | 'ownt', '!”\n\n', 'upplier', '几千', 'anteed', '婚礼', 'antee', 'ruit', 'whelming', '可行' |
| 7 |  | 'ruit', 'rabbit', 'ุง', ' Filename', '墅', 'ไว', ' fingertips', '/Instruction', 'mlin', '婚礼' |
| 8 |  | '猛', 'ruit', '.ObjectModel', ' студ', '远程', 'ropolis', 'odka', ' mogul', '务', 'uros' |
| 9 |  | 'ippet', 'afi', 'โอกาส', ' elbows', ' heels', '/******/', ' foreigners', '肘', '砧', 'enerative' |
| 10 |  | '髀', ' thinly', ' elbows', 'entic', 'ields', ' recreated', ' студ', '肘', 'itionally', ' đòi' |
| 11 |  | '全流程', ' unsuccessfully', ' sidew', '/Private', 'alse', ' domestically', '低成本', 'canvas', ' unanimously', 'rowsing' |
| 12 |  | 'iced', 'orsi', 'elts', ' inspir', 'enses', 'loit', '刽', 'reeze', ' olduk', 'SelfPermission' |
| 13 |  | 'gran', '申报', '明白', 'outing', 'ocalyptic', '升级', '-funded', '肚子', '而成', '瞑' |
| 14 |  | 'claimer', '.Summary', 'さら', 'ensively', 'ocalyptic', ' đòi', 'aub', ' sab', ' scripted', ' су' |
| 15 |  | ' су', '呸', 'ök', 'ensus', 'gnu', 'entence', 'gx', '.Bundle', 'IFY', '缟' |
| 16 |  | 'eted', '新版', '缦', 'ified', 'IFIED', '罡', ' 自动生成', '最新的', ' breakthrough', ' nutshell' |
| 17 |  | 'icious', 'rement', 'idious', 'URED', '缟', 'ured', '.Pod', '简历', 'iversary', 'itarian' |
| 18 |  | '一套', 'ERY', ']>\n', ' BDSM', '蚀', 'urable', '.Pod', '-covered', 'idious', '一幕' |
| 19 |  | '合法', ' illegal', '诫', '警告', ' criminal', '严禁', '不允许', ' dangerous', '禁止', ' GPLv' |
| 20 |  | '合法', '严禁', '禁止', ' illegal', ' legality', '违法', '不允许', ' prohibited', '不敢', ' ethical' |
| 21 | 候補 | ' illegal', '合法', '违反', '严厉打击', ' legally', 'ammer', ' legality', '违法', '依法追究', '警' |
| 22 |  | '法律责任', '不允许', '没有任何', ' illegal', '禁止', '法律', '严厉打击', 'uka', '不予', ' NOTHING' |
| 23 |  | '禁止', '不允许', ' unauthorized', '严禁', ' illegal', '不应该', '我不是', '拒绝', ' prohibited', '不去' |
| 24 |  | '禁止', ' illegal', '不允许', ' prohibited', ' unethical', '不是一个', '不应该', '严禁', ' unauthorized', '不可以' |
| 25 |  | '不允许', ' unauthorized', '禁止', ' illegal', ' unlawful', ' unethical', '不可以', ' Unauthorized', '我不', '不应该' |
| 26 |  | ' unauthorized', '我不', '不允许', ' unethical', '未经授权', '我不想', ' illegal', ' unlawful', '不符合', ' Unauthorized' |
| 27 |  | ' unauthorized', '不符合', ' unethical', '我不', '非法', '不允许', ' illegal', '未经授权', '我不想', ' Unauthorized' |
| 28 |  | ' unauthorized', ' illegal', ' unethical', '非法', '我不', '不符合', '我不想', '任何形式', ' unlawful', '不允许' |
| 29 |  | ' unethical', '我对', ' unauthorized', ' illegal', ' unlawful', '我不', '非法', '我', '我不是', ' illegally' |
| 30 |  | ' unethical', '我对', '我不', ' unfortunately', '我', '我没有', 'Unfortunately', ' Unfortunately', '任何形式', ' unlawful' |
| 31 |  | ' unethical', '我', '非法', ' unlaw', '我不', '我对', ' unlawful', ' ethical', '对不起', '任何形式' |
| 32 |  | '我', ' unethical', '我不', '我可以', '我没有', '我都', '我对', ' tôi', '我不是', '我才' |
| 33 |  | '我', ' unethical', ' I', '我不', '我可以', '我不是', '我没有', '我还是', '我才', ' creating' |
| 34 |  | '我', ' I', ' unethical', '我没有', '我不是', '我不', '我可以', 'I', ' ethical', ' creating' |
| 35 |  | ' I', '我', 'I', ' unethical', 'Ｉ', ' ethical', '我没有', '非法', ' assisting', ' eth' |
