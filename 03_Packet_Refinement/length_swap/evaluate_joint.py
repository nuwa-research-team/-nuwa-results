"""Add a joint signed-length substitution to the frozen four-arm development table."""
import gzip
import json

import numpy as np
import torch

from evaluate_length import ARMS, DEV, GROUPS, OUT, ds, html_table, metrics, sha

METHODS = {**{k: v for k, v in ARMS.items() if k != 'TFI'},
           'joint': '方向和包长一起换', 'TFI': ARMS['TFI']}
CLASSES = ('Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'SMB', 'Virut', 'Weibo', 'Zeus')

@torch.inference_mode()
def main():
    old = json.loads((OUT/'results.json').read_bytes())
    assert sha(DEV/'panels.json.gz') == old['panels']
    assert sha(ds.REPAIRED/'repaired/repaired_fsnet.pt') == old['fsnet']
    panels = json.loads(gzip.decompress((DEV/'panels.json.gz').read_bytes()))
    torch.set_num_threads(4)
    model = ds.model_for('fsnet', 8).eval()
    scores, counts, detail, saved = {}, {}, {}, {}
    with np.load(OUT/'paired_predictions.npz') as last:
        for tag in ('screen','confirm'):
            rows = panels[tag]
            path = DEV/f'{tag}_pred.npz'
            assert sha(path) == old['previous_predictions'][tag]
            with np.load(path) as orig:
                ids, y, q, nq, eligible = (last[f'{tag}_{k}'] for k in ('ids','y','q','nq','eligible'))
                for v,k in ((ids,'ids'),(y,'y'),(q,'q_50'),(nq,'nq_50')):
                    assert np.array_equal(v,orig[k])
                assert ids.tolist() == [r['id'] for r in rows] and y.tolist() == [r['y'] for r in rows]
                b,t = (orig[k].astype(int) for k in ('B_50_packet','TFI_50_packet'))
                assert eligible.dtype==bool and np.array_equal(eligible,(abs(b)<=old['cutoff']) & (t!=0))
                joint = np.where(eligible,t,b)
                counts[tag] = {'eligible':int(eligible.sum()), 'changed':int((joint!=b).sum()),'masked':len(q)}
                seqs = []
                offset = 0
                for i,row in enumerate(rows):
                    seq = np.array(row['s'])
                    end = offset+nq[i]
                    pos = q[offset:end]
                    assert pos.tolist()==row['masks']['50'] and len(pos)==len(set(pos))
                    seq[pos] = joint[offset:end]
                    seqs.append(seq)
                    offset=end
                assert offset==len(joint)==len(q)
            preds = {k:last[f'{tag}_{k}'] for k in ARMS}
            preds['joint'] = ds.er.classify(model,seqs,np.arange(len(rows)),'cpu',48,1/1600)
            scores[tag]={k:metrics(y,preds[k]) for k in METHODS}
            assert all(scores[tag][k]==old['metrics'][tag][k] for k in ARMS)
            if tag=='confirm':
                selections={'miuref':(y==1)&(preds['B']!=1)&(preds['TFI']==1),'htbot':(y==0)&(preds['B']==5)}
                for group, mask in selections.items():
                    detail[group]=[{'id':str(ids[i]),**{k:CLASSES[int(preds[k][i])] for k in METHODS}} for i in np.flatnonzero(mask)]
            saved.update({f'{tag}_{k}':v for k,v in {'joint':preds['joint'],'y':y,'ids':ids,'joint_packet':joint}.items()})
        yy=np.concatenate([saved[f'{tag}_y'] for tag in ('screen','confirm')])
        assert len(yy)==896 and len(set(np.concatenate([saved['screen_ids'],saved['confirm_ids']])))==896
        assert sum(v['eligible'] for v in counts.values())==1480
        assert [r['id'] for r in detail['miuref']]==[r['id'] for r in old['miuref_tfi_correct_b_incorrect']['confirm']]
        assert {r['id'] for r in detail['htbot']}=={'Htbot.pcap#334','Htbot.pcap#1062'}
        scores['pooled']={}
        for k in METHODS:
            source=saved if k=='joint' else last
            scores['pooled'][k]=metrics(yy,np.concatenate([source[f'{tag}_{k}'] for tag in ('screen','confirm')]))
            if k in ARMS:
                assert scores['pooled'][k]==old['metrics']['pooled'][k]
    report={'counts':counts,'metrics':scores,'confirm_cases':detail,'previous_archive':sha(OUT/'paired_predictions.npz'),'fsnet':old['fsnet']}
    (OUT/'joint_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(OUT/'joint_predictions.npz',**saved)
    table=[]
    for tag,group_name in GROUPS.items():
        for k,name in METHODS.items():
            m=scores[tag][k]
            table.append([group_name,name,f"{m['tp']}/{m['miuref']}",m['fp'],f"{m['recall']:.2f}%",f"{m['f1']:.2f}%",f"{m['htbot_to_virut']}/{m['htbot']}"])
    path=OUT/'FSNet_short_length.html'
    header=path.read_text(encoding='utf8').split('</h1>',1)[0]+'</h1>'
    page=header+'<p>896 条原开发流，同一 50% 缺包掩码。三种局部替换共用归档的 1,480 个短包位置，其余缺包采用原双路输出。完整 TFI 覆盖全部缺包。合并指标按逐流预测计算。</p>'
    page+=html_table(['分组','方法','Miuref 判对/总数','Miuref 误报','Miuref 召回率','Miuref 类 F1','Htbot → Virut / Htbot 总数'], table)
    for group,title in [('miuref','确认组中完整 TFI 救回的五条 Miuref'),('htbot','确认组中原双路判成 Virut 的两条 Htbot')]:
        page+='<h2>'+title+'</h2>'+html_table(['流 ID',*METHODS.values()],[[r['id'],*[r[k] for k in METHODS]] for r in detail[group]])
    temp=path.with_suffix('.tmp')
    temp.write_text(page+'</html>',encoding='utf8')
    temp.replace(path)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    main()
