import gzip
import json
from html import escape

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from pair_timing import describe
from experiment import DATA, FULL, IAT, P, RATES, ds, sha


def table(headers, rows):
    def cell(tag, x):
        return f'<{tag}>{escape(str(x))}</{tag}>'
    return '<table><thead><tr>'+''.join(cell('th',h) for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(cell('td',x) for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def metric(y,p):
    return np.array([accuracy_score(y,p),f1_score(y,p,labels=range(6),average='macro',zero_division=0)])*100


def display(v):
    return ' / '.join(f'{n:.2f}' for n in v)


def main():
    prior=np.load(IAT/'predictions.npz')
    raw=np.load(P/'full_predictions.npz')
    pair=np.load(P/'pair_full_predictions.npz')
    rates=json.loads((P/'full.json').read_bytes())['rates']
    report=json.loads((P/'pair_full.json').read_bytes())
    ref=json.loads((IAT/'metrics.json').read_bytes())['audit']
    for f in (raw,pair):
        assert np.array_equal(prior['ids'],f['ids']) and np.array_equal(prior['y'],f['y'])
    y=raw['y']
    seen=raw['exact_train_template']
    times=json.loads(gzip.decompress((FULL/'verified.json.gz').read_bytes()))
    flows=list(map(json.loads,(DATA/'flows.jsonl').open()))
    head=joblib.load(P/'pair_heads.joblib')['augmented']
    models=['base','joint','coarse','augmented','tfi','tfi_pair','clean','clean_pair']
    names=['原双路','联合投票','投票＋基础时间判别','投票＋方向条件时间判别','本地 TFI','TFI＋同一时间判别','真实包长和方向','真实包长和方向＋同一时间判别']
    summary=[]; detail=[]; overlap=[]; swaps=[]; audits=[]; transfer=[]
    for rate in RATES:
        path=DATA/f'mask{rate}.npz'
        assert sha(path)==ref[str(rate)]['mask_sha']
        with np.load(path) as mf:
            assert mf['mask'].sum()==rates[str(rate)]['joint']['slots']
            xx=np.array([describe(row['signed'],times[row['id']][1],np.flatnonzero(mask)) for row,mask in zip(flows,mf['mask'],strict=True)])
        binary=head.predict(xx)
        pool={k:[] for k in models}
        for kind in ds.NAMES:
            pp={k:raw[f'{kind}_{k}{rate}'] for k in ('base','joint','tfi','clean')}
            for k in ('coarse','augmented'): pp[k]=pair[f'{kind}_{k}{rate}']
            for k in ('tfi','clean'): pp[k+'_pair']=np.where(pp[k]<2,binary,pp[k])
            assert np.array_equal(np.where(pp['joint']<2,binary,pp['joint']),pp['augmented'])
            for k in ('base','tfi'):
                prior_key=f'{kind}_{"dual" if k=="base" else "tfi"}{rate}_after'
                audits.append(int(np.count_nonzero(pp[k]!=prior[prior_key])))
            for k,v in pp.items():
                m=metric(y,v)
                expected=(rates[str(rate)].get(k) or report[str(rate)].get(k))
                if expected:
                    saved=expected['models'][kind]
                    assert np.allclose(m,[saved['acc'],saved['f1']])
                pool[k].append(v)
            detail.append([rate,kind]+[display(metric(y,pp[k])) for k in models])
            if rate==50 and kind in ('fsnet','lstm'):
                for k in ('base','joint','augmented','tfi','tfi_pair','clean_pair'):
                    v=pp[k]
                    ok=lambda label: int(((y==label)&(v==label)).sum())
                    err=lambda a,b: int(((y==a)&(v==b)).sum())
                    swaps.append([kind,k,ok(4),ok(2),err(2,3),err(2,5),err(0,1),err(1,0)])
        summary.append([rate]+[display(np.mean([metric(y,v) for v in pool[k]],axis=0)) for k in models])
        for mask,desc in [(seen,'模板相同'),(~seen,'模板未见')]:
            overlap.append([rate,desc,int(mask.sum())]+[display(np.mean([metric(y[mask],v[mask]) for v in pool[k]],axis=0)) for k in ('base','joint','augmented','tfi_pair')])
        for k in ('base','joint','augmented','tfi_pair'):
            transfer.append([rate,k]+[round(np.mean([np.sum((y==a)&(v==b)) for v in pool[k]]),1) for a,b in ((0,1),(1,0))])
    print('averages',summary,'overlap',overlap,'pair_errors',transfer,'baseline replay mismatches',sum(audits),'across',len(audits),flush=True)
    sections=[('D4 三档评估',table(['丢包率 %']+names,summary)),('六个分类器',table(['丢包率 %','分类器']+names,detail)),
        ('同标签以外也可能共享包长模板；按完整序列是否在训练库中出现分层',table(['丢包率 %','分层','流数','原双路','投票','投票＋方向条件时间判别','TFI＋同一判别'],overlap)),
        ('SMB 和 Weibo 互判（六个分类器平均条数）',table(['丢包率 %','方法','Weibo → SMB','SMB → Weibo'],transfer)),
        ('50% 丢包重点类别（实际流数）',table(['分类器','方法','Miuref 正确 /127','Htbot 正确 /254','Htbot → Virut','Htbot → 其他','Weibo → SMB','SMB → Weibo'],swaps))]
    pkt=[]
    for rate in RATES:
        for k in ('base','joint','tfi'):
            s=rates[str(rate)][k]
            pkt.append([rate,k,s['slots'],f"{s['mae']:.2f}",f"{s['direction']:.2f}"])
    sections.append(('缺包位置上的补全指标',table(['丢包率 %','方法','位置数','绝对包长 MAE（字节）','方向正确率 %'],pkt)))
    sections.append(('Nüwa 补充材料 Table XVIII，d4，六分类器平均',table(['丢包率 %','官方 Accuracy %','官方 F1 %'],[[10,79.38,77.51],[20,78.11,76.28],[30,76.82,74.86],[50,'未报告','未报告']])))
    dev=[]
    for tag in ('screen','confirm'):
        orig=json.loads((P/f'{tag}.json').read_bytes())['rates']
        rerank=json.loads((P/f'pair_{tag}.json').read_bytes())
        for k in ('base','direction','magnitude','joint','coarse','augmented'):
            rs=[(orig[r].get(k) or rerank[r][k]) for r in orig]
            dev.append([tag,k,display([np.mean([d[f] for d in rs]) for f in ('acc','f1')])])
    sections.append(('开发集消融（三档等权）',table(['组别','修改','Accuracy / macro-F1 %'],dev)))
    txt='<!doctype html><html lang="zh"><meta charset="utf-8"><title>D4 优化实验</title><style>body{font:16px system-ui;max-width:1450px;margin:40px auto;padding:15px;color:#20252b}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid}th,td{padding:6px 9px;text-align:left}thead{border-bottom:1px solid}h2{margin-top:34px}</style><h1>D4 补全与分类优化实验</h1><p>13,150 条流，六分类。分数为 Accuracy / macro-F1（%），各分类器等权。所有方法使用相同可见间隔和原掩码。训练模板仅来自 4,096 条训练流。时间判别只修改原预测落在 SMB / Weibo 的流。clean 使用完整带方向包长和掩码后可见间隔。</p>'
    txt+=''.join(f'<h2>{escape(title)}</h2>{t}' for title,t in sections)
    txt+=f'<p>36 组原双路及 TFI 预测和上一轮逐流复核，共 {sum(audits)} 个分类结果不同。原文选样、下游输入和 F1 平均方式未完全对齐。PCAP 分片留出详见 pair_transfer.json。目前尚无独立采集结果。</p></html>'
    (P/'D4_optimization.html').write_text(txt,encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=3): main()
