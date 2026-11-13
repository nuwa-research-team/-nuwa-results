"""Combine frozen predictions for the original D4 10%, 30%, and 50% masks."""
import json
from html import escape
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUT = Path(__file__).resolve().parent
P = OUT.parent
BASE = P.parent / 'd4_full_aligned_20261113' / 'baseline'
NAMES = {'cnn':'CNN', 'lstm':'LSTM', 'gru':'GRU', 'tst':'TST', 'sdae':'SDAE', 'fsnet':'FS-Net'}
ARMS = ['原双路', '短包替换', '完整 TFI']
RATES = [10, 30, 50]


def table(headers, rows):
    def row(cells, tag):
        return '<tr>' + ''.join(f'<{tag}>{escape(str(c))}</{tag}>' for c in cells) + '</tr>'
    return '<table><thead>' + row(headers, 'th') + '</thead><tbody>' + ''.join(row(r, 'td') for r in rows) + '</tbody></table>'


def aligned(path, ids, y, allow_extra=False):
    with np.load(path) as z:
        lookup = {k:i for i,k in enumerate(z['ids'].astype(str))}
        assert len(lookup) == len(z['ids']) and set(ids).issubset(lookup)
        assert allow_extra or len(lookup) == len(ids)
        indices = [lookup[k] for k in ids]
        assert np.array_equal(y, z['y'][indices])
        pred = z['pred'][indices]
        assert np.all(np.isin(pred, np.arange(8)))
        return pred


def main():
    old = json.loads((P/'comparison_data.json').read_bytes())
    refs = {(v['rate'],v['model']):v for t in old['tables'] if t['id']=='full'
            for r in t['rows'] for v in [r['values']]}
    old50 = json.loads((P/'full_joint_50/results.json').read_bytes())
    rows, focus, out = [], [], {}
    with np.load(P/'full_joint_50/fsnet.npz') as z:
        ids, y = z['ids'].astype(str), z['y']
    assert len(set(ids)) == len(ids) == 13150 and set(y) == set(range(8))
    mi, ht = y == 1, y == 0
    assert mi.sum() == 127 and ht.sum() == 254
    for rate in RATES:
        folder = P/'full_joint_50' if rate == 50 else OUT/str(rate)
        obj = {'protocol':json.loads((folder/'audit.json').read_bytes()), 'classifiers':{}}
        for name, label in NAMES.items():
            paths = [P/f'{name}_direct_{rate}.npz', folder/f'{name}.npz', BASE/f'{name}_tfi_{rate}.npz']
            preds = [aligned(path, ids, y, allow_extra=(i == 2)) for i, path in enumerate(paths)]
            scores = [[100*accuracy_score(y,a),100*f1_score(y,a,labels=np.arange(8),average='macro',zero_division=0)] for a in preds]
            b,h,t = scores
            ref = refs[rate,label]
            assert np.allclose([*b,*t], [ref[k] for k in ('b','bf','t','tf')])
            if rate == 50:
                assert np.allclose(scores, [old50['classifiers'][label][k] for k in ('B','joint','TFI')])
            obj['classifiers'][label] = scores
            if name == 'fsnet':
                obj['fsnet'] = [{'miuref_tp':int((mi & (a==1)).sum()),'miuref_support':int(mi.sum()),
                                 'htbot_to_virut':int((ht & (a==5)).sum()), 'htbot_support':int(ht.sum())} for a in preds]
                obj['transitions'] = dict(miuref_gained=int((mi & (preds[0]!=1) & (preds[1]==1)).sum()),
                                          miuref_lost=int((mi & (preds[0]==1) & (preds[1]!=1)).sum()),
                                          htbot_to_virut_resolved=int((ht & (preds[0]==5) & (preds[1]!=5)).sum()),
                                          htbot_to_virut_new=int((ht & (preds[0]!=5) & (preds[1]==5)).sum()))
                for arm,d in zip(ARMS,obj['fsnet']):
                    focus.append([f'{rate}%',arm,f"{d['miuref_tp']}/127",f"{100*d['miuref_tp']/127:.2f}%",f"{d['htbot_to_virut']}/254"])
        obj['mean'] = np.mean(list(obj['classifiers'].values()),axis=0).tolist()
        for label, scores in [*obj['classifiers'].items(),('六模型均值',obj['mean'])]:
            rows.append([f'{rate}%',label,*[f'{a:.2f} / {f:.2f}' for a,f in scores]])
        out[rate]=obj
    payload = dict(arms=ARMS,metrics=['accuracy_pct','macro_f1_pct'],rates=out)
    html = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 三档丢包对照</title><style>body{font:16px system-ui;max-width:1050px;margin:36px auto;padding:0 22px;color:#182939}table{border-collapse:collapse;width:100%;border-block:2px solid;margin-bottom:30px}td,th{padding:7px 12px;text-align:right}thead{border-bottom:1px solid}td:nth-child(2),th:nth-child(2){text-align:left}tbody tr:nth-child(7n){font-weight:bold}</style><h1>D4 · 短包联合替换</h1><p>同一批 13,150 条流，沿用各档缺包位置和冻结权重。双路预测绝对包长 ≤128 字节且 TFI 预测非零时，用 TFI 的带方向包长替换。各格为准确率 / 八类 macro-F1（%）。均值为六个分类器的算术平均。</p>'
    html += table(['丢包率','分类器',*ARMS],rows) + '<h2>FS-Net 逐类结果</h2>' + table(['丢包率','补全方法','Miuref 判对/总数','Miuref 召回率','Htbot 判为 Virut/总数'],focus)+'</html>'
    (OUT/'comparison.html').write_text(html, encoding='utf8')
    (OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    for row in rows + focus:
        print(*row,sep=' | ')

if __name__ == '__main__':
    main()

