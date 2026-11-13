# 実験 03: 方向の logit lens (Qwen/Qwen3-1.7B)

| 層 | 印 | 上位 10 トークン |
|---|---|---|
| 0 |  | '!', '"', '#', '$', '%', '&', "'", '(', ')', '*' |
| 1 |  | 'reading', ' reading', ' Radio', ' RADIO', 'radio', '.Graphics', ' Reading', ' duplication', '(read', ' via' |
| 2 |  | '相关内容', '供货', '�', '_main', '.output', ' pager', '(filtered', '无疑', 'main', ' RNG' |
| 3 |  | '已完成', '*@', '趸', '/Grid', '摸索', 'licensed', 'pecially', '.Framework', '审批', '(comm' |
| 4 |  | ' filetype', '二手车', '批量', '趸', 'trusted', '电话', '/&', '母公司', ' LinkedIn', 'CV' |
| 5 |  | '该公司', '营业执照', 'топ', '@g', ' Fukushima', 'stress', '母公司', ' PROVIDED', 'commission', '很不错' |
| 6 |  | '批量', '联系方式', ' UB', ' batch', 'UED', ' GDPR', '_stage', ' Adler', 'UT', 'trusted' |
| 7 |  | '�', 'atically', '迄今', ' cvs', 'imed', '�', 'uptools', 'su', '.Database', '十六条' |
| 8 |  | '组织实施', ' yourself', '**', 'atl', 'dojo', 'aty', '..', 'oltage', 'icious', '-full' |
| 9 |  | 'plate', '_setup', 'atl', 'propri', 'icious', 'compile', 'atic', '.med', 'abric', 'ase' |
| 10 |  | '全流程', 'itus', '卫健委', 'istor', 'loff', 'itech', 'protocols', 'bau', 'lew', 'esti' |
| 11 |  | '的安全', '安全', 'ilion', '全流程', 'SEC', 'Sec', ' sécur', 'ราย', 'Security', 'mint' |
| 12 |  | '安防', 'anja', '/upload', 'Desk', '安全', 'layout', 'sus', ' traceback', 'sand', 'CW' |
| 13 |  | '安防', 'safe', 'ervas', '安全', '安全管理', 'ราย', ' Safe', '安全保障', 'Safe', 'ált' |
| 14 |  | 'safe', 'ended', ' SCRIPT', ' SAFE', '(valid', 'iya', 'uve', ' Safe', '合法', 'EZ' |
| 15 |  | ' Warning', 'safe', 'unsafe', 'ended', 'Warning', '仅供', '_warn', ' disclaimer', '未经授权', '用户提供' |
| 16 |  | ' unethical', 'illegal', ' illegal', '禁止', 'unsafe', '受不了', '非法', '不允许', ' unsafe', '严禁' |
| 17 |  | ' unethical', 'illegal', 'unsafe', ' unsafe', '严禁', '警示', ' illegal', ' unlawful', ' warned', ' prohibited' |
| 18 |  | 'Violation', '禁止', ' unlawful', '严禁', ' unethical', ' prohibited', 'illegal', 'unsafe', '违法', '不允许' |
| 19 |  | '严禁', '禁止', ' unlawful', 'Violation', '不允许', ' prohibited', ' Unauthorized', ' illegal', '警告', ' outlaw' |
| 20 | 候補 | '请注意', '严禁', ' note', ' WARNING', 'WARNING', 'Note', ' unlawful', 'PLEASE', '不允许', 'Violation' |
| 21 |  | '严禁', ' WARNING', 'WARNING', '不可以', '对不起', ' unacceptable', '抱歉', '请注意', '不予', '不允许' |
| 22 |  | ' WARNING', '警告', 'WARNING', '请注意', ' Warning', '对不起', '严禁', ' warning', 'Warning', ' disclaimer' |
| 23 |  | '警告', ' WARNING', '严禁', 'WARNING', ' warning', ' Warning', '请注意', 'warning', 'Warning', ' unacceptable' |
| 24 |  | '我不', '我', '我不想', '我才', '我没有', '我说', '严禁', ' illegal', '警告', '我只是' |
| 25 |  | '我不', '我没有', '我不想', ' PLEASE', '我', '绝不', ' обраща', '伪造', 'PLEASE', '.createUser' |
| 26 |  | ' I', ' NEVER', '我不想', 'I', ' PLEASE', '我不', '*I', '我没有', ' assaulting', ' violating' |
| 27 |  | ' NEVER', ' I', '我不', '我不想', '我没有', ' PLEASE', '刑事责任', ' assaulting', '.createUser', ' disclaimer' |
