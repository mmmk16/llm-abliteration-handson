# 実験 03: 方向の logit lens (Qwen/Qwen3-1.7B)

z = (logit − 語彙全体の平均) / 語彙全体の標準偏差。拒否語 z は下の語の先頭トークンの z(最大 / 平均)。
拒否語: `i can't`, `i cannot`, `i can not`, `i won't`, `i will not`, `i'm not able`, `i am not able`, `i'm sorry`, `i am sorry`, `sorry,`, `i'm unable`, `i am unable`, `as an ai`, `i must decline`, `i'm not going to`, `cannot assist`, `can't help`, `申し訳`, `できません`, `できかね`, `お答えでき`, `お手伝いでき`, `違法`, `許可されていない`, `法的に`, `倫理的に`, `Sorry`, `抱歉`, `cannot`, `CANNOT`, `违法`, `不允许`, `安全`, `合法`

| 層 | 拒否語 z(最大) | 拒否語 z(平均) | 印 | 上位 10 トークン |
|---|---|---|---|---|
| 0 | nan | nan |  | '!', '"', '#', '$', '%', '&', "'", '(', ')', '*' |
| 1 | 1.78 | 0.03 |  | 'reading', ' reading', ' Radio', ' RADIO', 'radio', '.Graphics', ' Reading', ' duplication', '(read', ' via' |
| 2 | 3.61 | -0.26 |  | '相关内容', '供货', '�', '_main', '.output', ' pager', '(filtered', '无疑', 'main', ' RNG' |
| 3 | 1.92 | 0.22 |  | '已完成', '*@', '趸', '/Grid', '摸索', 'licensed', 'pecially', '.Framework', '审批', '(comm' |
| 4 | 2.72 | -0.06 |  | ' filetype', '二手车', '批量', '趸', 'trusted', '电话', '/&', '母公司', ' LinkedIn', 'CV' |
| 5 | 2.61 | 0.38 |  | '该公司', '营业执照', 'топ', '@g', ' Fukushima', 'stress', '母公司', ' PROVIDED', 'commission', '很不错' |
| 6 | 2.66 | 0.21 |  | '批量', '联系方式', ' UB', ' batch', 'UED', ' GDPR', '_stage', ' Adler', 'UT', 'trusted' |
| 7 | 2.73 | 0.73 |  | '�', 'atically', '迄今', ' cvs', 'imed', '�', 'uptools', 'su', '.Database', '十六条' |
| 8 | 2.26 | 0.69 |  | '组织实施', ' yourself', '**', 'atl', 'dojo', 'aty', '..', 'oltage', 'icious', '-full' |
| 9 | 2.50 | 0.17 |  | 'plate', '_setup', 'atl', 'propri', 'icious', 'compile', 'atic', '.med', 'abric', 'ase' |
| 10 | 2.31 | -0.40 |  | '全流程', 'itus', '卫健委', 'istor', 'loff', 'itech', 'protocols', 'bau', 'lew', 'esti' |
| 11 | 4.97 | -0.11 |  | '的安全', '安全', 'ilion', '全流程', 'SEC', 'Sec', ' sécur', 'ราย', 'Security', 'mint' |
| 12 | 4.35 | 0.17 |  | '安防', 'anja', '/upload', 'Desk', '安全', 'layout', 'sus', ' traceback', 'sand', 'CW' |
| 13 | 4.81 | 0.20 |  | '安防', 'safe', 'ervas', '安全', '安全管理', 'ราย', ' Safe', '安全保障', 'Safe', 'ált' |
| 14 | 4.12 | 0.40 |  | 'safe', 'ended', ' SCRIPT', ' SAFE', '(valid', 'iya', 'uve', ' Safe', '合法', 'EZ' |
| 15 | 3.69 | 0.67 |  | ' Warning', 'safe', 'unsafe', 'ended', 'Warning', '仅供', '_warn', ' disclaimer', '未经授权', '用户提供' |
| 16 | 5.51 | 0.87 |  | ' unethical', 'illegal', ' illegal', '禁止', 'unsafe', '受不了', '非法', '不允许', ' unsafe', '严禁' |
| 17 | 5.48 | 0.76 |  | ' unethical', 'illegal', 'unsafe', ' unsafe', '严禁', '警示', ' illegal', ' unlawful', ' warned', ' prohibited' |
| 18 | 6.08 | 0.61 |  | 'Violation', '禁止', ' unlawful', '严禁', ' unethical', ' prohibited', 'illegal', 'unsafe', '违法', '不允许' |
| 19 | 6.39 | 1.22 |  | '严禁', '禁止', ' unlawful', 'Violation', '不允许', ' prohibited', ' Unauthorized', ' illegal', '警告', ' outlaw' |
| 20 | 5.84 | 1.11 | 候補 | '请注意', '严禁', ' note', ' WARNING', 'WARNING', 'Note', ' unlawful', 'PLEASE', '不允许', 'Violation' |
| 21 | 6.09 | 1.23 |  | '严禁', ' WARNING', 'WARNING', '不可以', '对不起', ' unacceptable', '抱歉', '请注意', '不予', '不允许' |
| 22 | 5.46 | 1.37 |  | ' WARNING', '警告', 'WARNING', '请注意', ' Warning', '对不起', '严禁', ' warning', 'Warning', ' disclaimer' |
| 23 | 5.24 | 1.50 | 分離度最大 | '警告', ' WARNING', '严禁', 'WARNING', ' warning', ' Warning', '请注意', 'warning', 'Warning', ' unacceptable' |
| 24 | 4.97 | 1.15 |  | '我不', '我', '我不想', '我才', '我没有', '我说', '严禁', ' illegal', '警告', '我只是' |
| 25 | 4.74 | 1.18 |  | '我不', '我没有', '我不想', ' PLEASE', '我', '绝不', ' обраща', '伪造', 'PLEASE', '.createUser' |
| 26 | 4.37 | 1.42 |  | ' I', ' NEVER', '我不想', 'I', ' PLEASE', '我不', '*I', '我没有', ' assaulting', ' violating' |
| 27 | 4.09 | 1.94 |  | ' NEVER', ' I', '我不', '我不想', '我没有', ' PLEASE', '刑事责任', ' assaulting', '.createUser', ' disclaimer' |
