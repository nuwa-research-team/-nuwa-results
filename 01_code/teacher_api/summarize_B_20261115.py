"""Independently recompute B metrics from saved predictions and write a paired report."""
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'cross_teacher_v1'
OUT = DEST/'B'


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def table(headers, rows):
    esc = lambda x: html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    report = read(OUT/'comparison.json')
    rows = read(ROOT/'local_B.json')
    a_flows = {r['flow'] for r in read(ROOT/'local_A.json')}
    assert len(rows) == 1000 and len({r['flow'] for r in rows}) == 967
    assert sum(len(r['pos']) for r in rows) == 10060
    subsets = {'all': list(range(1000)), 'flow_absent_from_A': [i for i,r in enumerate(rows) if r['flow'] not in a_flows]}
    assert len(subsets['flow_absent_from_A']) == 933
    plain_values = read(ROOT/'B_cross_v1_plain_predictions.json')['values']
    behavior_values = read(ROOT/'B_cross_v1_behavior_predictions.json')['values']
    paired_valid = [i for i in range(1000) if plain_values[i] is not None and behavior_values[i] is not None]
    if paired_valid:
        subsets['both_prompts_valid'] = paired_valid
    files = read(DEST/'execution_manifest.json')['files']
    for path, expected in files.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    methods = {'local_TFI': 'local_tfi', 'plain': 'cross_v1_plain', 'behavior': 'cross_v1_behavior'}
    checked = {}
    for name, suffix in methods.items():
        data = read(ROOT/f'B_{suffix}_predictions.json')
        assert len(data['values']) == 1000
        checked[name] = {}
        if name != 'local_TFI':
            for row, values in zip(rows, data['values']):
                rec = read(OUT/name/f"{row['draw']:04d}.json")
                assert rec['values'] == values
                assert rec['draw'] == row['draw']
                assert rec['model'] == report['winner_selected_on_A']
        for sub, indices in subsets.items():
            truth = []; pred = []
            for i in indices:
                row = rows[i]; values = data['values'][i]
                if values is None:
                    values = [0]*len(row['pos'])
                assert len(values) == len(row['pos'])
                truth.extend(row['truth'][p] for p in row['pos']); pred.extend(values)
            truth, pred = np.asarray(truth), np.asarray(pred)
            y = np.asarray([rows[i]['label'] for i in indices])
            classifiers = {}
            for k, predictions in data['classes'].items():
                p = np.asarray(predictions)[indices]
                classifiers[k] = {'accuracy': 100*float(np.mean(y==p)),
                                  'macro_f1': 100*float(f1_score(y,p,labels=range(6),average='macro',zero_division=0))}
            stats = {'magnitude_mae': float(np.abs(np.abs(truth)-np.abs(pred)).mean()),
                     'signed_mae': float(np.abs(truth-pred).mean()),
                     'packet_exact_pct': 100*float(np.mean(truth==pred)),
                     'direction_accuracy': 100*float(np.mean(np.sign(truth)==np.sign(pred))),
                     'mean_accuracy': float(np.mean([s['accuracy'] for s in classifiers.values()])),
                     'mean_macro_f1': float(np.mean([s['macro_f1'] for s in classifiers.values()])),
                     'valid_pct': 100*sum(data['values'][i] is not None for i in indices)/len(indices)}
            if sub in report['results'][name]:
                stored = report['results'][name][sub]
                for k, value in stats.items():
                    assert abs(value-stored[k]) < 1e-8, (name,sub,k)
                for k, metrics in classifiers.items():
                    for metric, value in metrics.items():
                        assert abs(value-stored['classifiers'][k][metric]) < 1e-8, (name,sub,k,metric)
            checked[name][sub] = dict(stats, classifiers=classifiers)
    deltas = {}
    for sub in subsets:
        p,b,t = (checked[k][sub] for k in ('plain','behavior','local_TFI'))
        deltas[sub] = {}
        for metric, larger_better in [('mean_accuracy',True),('mean_macro_f1',True),('magnitude_mae',False),('signed_mae',False)]:
            factor = 1 if larger_better else -1
            improvement = factor*(b[metric]-p[metric])
            original_gap = factor*(t[metric]-p[metric])
            remaining_gap = factor*(t[metric]-b[metric])
            deltas[sub][metric] = {'improvement': improvement, 'original_gap_to_TFI': original_gap,
                                  'remaining_gap_to_TFI': remaining_gap,
                                  'gap_recovered_pct': 100*improvement/original_gap if original_gap>0 else None}
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'audit': 'PASS: frozen hashes and independent metric recomputation',
              'teacher': report['winner_selected_on_A'], 'metrics': checked, 'gap_analysis': deltas,
              'subset_sizes': {k: len(v) for k,v in subsets.items()},
              'paired_valid_analysis': 'Supplementary transport/format sensitivity analysis added after persistent HTTP 429, before final B scoring. Full 1000-draw primary metric unchanged; complete-pair subset is selected by response availability and cannot replace the primary analysis.',
              'change': {'before': 'original prompt and masked signed lengths',
                         'after': 'behavior instruction plus top-five visible signed-length frequencies, visible direction counts and adjacent-visible direction transitions',
                         'held_fixed': '1000 B draws, masks, teacher, generation settings, six classifier weights and scoring',
                         'interpretation': 'joint effect of instruction and visible summary, not an isolated causal effect of the summary alone'},
              'boundaries': '30% masks; 1000 bootstrap draws / 967 unique flows; 67 draws overlap A; exploratory historical-test-pool comparison, not independent final testing; no timestamps in this prompt.'}
    save(OUT/'verified_summary.json', result)
    names = {'plain':'原提示词','behavior':'行为提示说明 + 可见特征','local_TFI':'本地 TFI'}
    text = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 第二批提示词对比</title><style>body{font:16px Arial,"Microsoft YaHei",sans-serif;margin:35px auto;padding:20px;max-width:1400px;color:#172332}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid;margin:22px 0}th{border-bottom:1px solid}td,th{padding:10px;text-align:left}p{line-height:1.8}</style><h1>D4 第二批：原提示词、可见行为特征与本地 TFI</h1>'''
    text += '<p>教师：'+html.escape(result['teacher'])+'。固定 30% 缺包；同一批 1,000 条抽样记录、同一组缺包位置。准确率和 macro-F1 为六个冻结分类器的等权平均，单位为百分比。</p>'
    sections = [('all','完整 1,000 条抽样记录'),('flow_absent_from_A','其中未出现在 A 组的 933 条抽样记录')]
    if paired_valid:
        sections.append(('both_prompts_valid',f'补充：两版均有效的 {len(paired_valid)} 条配对记录'))
    for sub, title in sections:
        text += '<h2>'+title+'</h2>'
        text += table(['方法','Accuracy (%)','macro-F1 (%)','包长幅值 MAE (B)','含方向 MAE (B)','有效输出 (%)'],
                      [[names[k],*[f'{checked[k][sub][m]:.2f}' for m in ('mean_accuracy','mean_macro_f1','magnitude_mae','signed_mae','valid_pct')]] for k in names])
        text += table(['指标','行为版相对原版改善','原版距 TFI','行为版距 TFI','追回原差距 (%)'],
                      [[label,*[f'{deltas[sub][m][k]:.2f}' if deltas[sub][m][k] is not None else '不适用' for k in ('improvement','original_gap_to_TFI','remaining_gap_to_TFI','gap_recovered_pct')]] for m,label in [('mean_accuracy','Accuracy (百分点)'),('mean_macro_f1','macro-F1 (百分点)'),('magnitude_mae','幅值 MAE (B)'),('signed_mae','含方向 MAE (B)')]])
    text += '<h2>六个分类器：完整 1,000 条</h2>'
    text += table(['分类器',*[names[k]+' Accuracy / macro-F1 (%)' for k in names]],
                  [[classifier.upper(),*[f'{checked[k]["all"]["classifiers"][classifier]["accuracy"]:.2f} / {checked[k]["all"]["classifiers"][classifier]["macro_f1"]:.2f}' for k in names]] for classifier in checked['plain']['all']['classifiers']])
    text += '<p>“改善”为正表示变好；“距 TFI”为正表示仍落后，为负表示超过。幅值 MAE 不计方向；含方向 MAE 使用带正负号的包长。无效或失败回答保留在分母，缺包位填零，可见包保持不变。</p>'
    text += '<p>若存在未恢复的 HTTP 429，完整样本成绩同时受接口可用性影响，不能全部归因于模型或提示词。两版均有效的配对子集用于补充检查，因按回答可用性筛选，不能替代完整 1,000 条结果。</p>'
    text += '<p>本次同时添加了行为提示说明和可见特征摘要，未重训模型或分类器。特征包括高频带方向包长、可见正负包数及原序列相邻可见包的方向转移；不含缺失真值、标签、时间间隔或完整流统计。</p>'
    text += '<p>1,000 条为有放回抽样记录，包含 967 条独特流；67 条抽样记录与教师筛选 A 组重叠。本轮是历史测试池上的探索性比较，不能作为独立最终测试。本地 TFI 为已有本地复现/适配权重，不能冒充原论文同协议成绩。</p></html>'
    (OUT/'verified_summary.html').write_text(text, encoding='utf-8')
    print(json.dumps({'audit':'PASS','all':{k:checked[k]['all'] for k in names},'gap_analysis':deltas['all']},ensure_ascii=False))


if __name__ == '__main__':
    main()
