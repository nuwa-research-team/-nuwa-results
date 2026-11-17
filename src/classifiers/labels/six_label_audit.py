"""Post-hoc label harmonization using Nüwa's timestamp-preparation script."""
import ast,json,pickle,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from controls_report import DISPLAY
from report import metrics,paper
p=Path(__file__).resolve().parent;repo=p.parents[1];project=repo/'nuwa_reproduction_20261109'
src=project/'upstream/Nuwa/code/data_process/generate_timestamps.py'
maps=[ast.literal_eval(n.value) for n in ast.walk(ast.parse(src.read_text(encoding='utf-8-sig'))) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='USTC_Label2num6' for v in n.targets)]
assert len(maps)==1
mapping=maps[0];raw=pickle.loads((project/'audit/paired_snapshot_20261110/work/d4_aligned/raw.pkl').read_bytes());names=raw['labels']
def g(n):return mapping[n+'-1'] if n in ('Weibo','SMB') else mapping[n]
m=np.array([g(s) for s in names]);assert set(m)==set(range(6))
rows=[];paper={**paper,20:[(80.52,80.07),(80.65,79.34),(79.80,79.04),(81.54,81.04),(65.17,54.68),(81.00,83.53)]}
lines=['# 作者脚本的六类映射与本地敏感性分析','',
'作者的时间序列准备脚本使用以下六类：Weibo、SMB、Htbot、Virut、Miuref，以及合并的 Neris / Nsis-ay / Zeus / Geodo / Shifu。本地这批只有最后一组中的前三个来源。作者的通用包长训练准备脚本另列十类字典。附录表 XVIII 没有给出逐流身份，无法把时间准备脚本的六类划分确认为该表实际使用的逐流划分。[作者六类字典](https://github.com/Timeless-zfqi/Nuwa/blob/main/code/data_process/generate_timestamps.py) · [通用训练准备脚本](https://github.com/Timeless-zfqi/Nuwa/blob/main/code/data_process/train_test_data.py)。','',
'在同一批 13,150 条流上，将八分类器的硬预测与真值按字典合并后重新计分。分类器权重、输入和输出决策均沿用八类版本。此检查量化标签合并对分数的影响，直接训练六分类器是另一项实验。','',
'|丢包|分类器|双路，合并六类 Acc / macro-F1|TFI，合并六类|双路 − TFI（百分点）|论文表 XVIII，Acc / F1|','|---|---|---:|---:|---:|---:|']
for r in (10,20,30,50):
    arm=p/'controls/paper20' if r==20 else p
    for j,k in enumerate(DISPLAY):
        a=np.load(arm/f'outputs/d4_downstream_random/{k}_aligned_{r}.npz')
        b=np.load(p/('extra_inputs' if r==20 else 'baseline')/f'{k}_tfi_{r}.npz')
        ids=a['ids'].astype(str);order={s:i for i,s in enumerate(b['ids'].astype(str))};ix=[order[s] for s in ids]
        y=a['y'];assert np.array_equal(y,b['y'][ix]) and len(set(ids))==len(ids)==13150 and set(y)==set(range(8))
        u=metrics(m[y],m[a['pred']]);t=metrics(m[y],m[b['pred'][ix]])
        u8=metrics(y,a['pred']);t8=metrics(y,b['pred'][ix])
        assert u[0]>=u8[0] and t[0]>=t8[0]
        rows.append(dict(rate=r,clf=k,ours=u.tolist(),TFI=t.tolist(),ours8=u8.tolist(),TFI8=t8.tolist()))
        fmt=lambda x:' / '.join(f'{v:.2f}' for v in x)
        lines.append(f'|{r}%|{DISPLAY[k]}|{fmt(u)}|{fmt(t)}|{fmt(u-t)}|'+(fmt(paper[r][j]) if r in paper else '未报告')+'|')
    sub=rows[-6:];a=np.mean([q['ours'] for q in sub],axis=0);b=np.mean([q['TFI'] for q in sub],axis=0)
    lines.append(f'|{r}%|**六分类器均值**|{fmt(a)}|{fmt(b)}|{fmt(a-b)}|'+(fmt(np.mean(paper[r],axis=0)) if r in paper else '未报告')+'|')
bs=defaultdict(Counter)
for flow in map(json.loads,(p.parent/'d4_downstream_random/flows.jsonl').open()):bs[tuple(flow['signed'])][int(m[flow['y']])]+=1
correct=sum(max(b.values()) for b in bs.values())
lines+=['',f'在合并后的六类真值上，同一完整带方向包长序列强制给出同一预测时，这批流最多分对 {correct:,}/13,150 条（{100*correct/13150:.2f}%）。']
(p/'six_label_audit.json').write_text(json.dumps(dict(mapping=mapping,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),support=Counter(map(int,m[y])),ceiling_correct=correct,results=rows),indent=2))
(p/'six_label_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(s for s in lines if '均值**' in s));print('CEILING',correct)
