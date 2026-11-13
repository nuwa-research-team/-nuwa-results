"""Replay A at 50% only; inspect paired flips without fitting any estimator."""
import json
from pathlib import Path
from html import escape
from collections import Counter

import numpy as np
import joblib
from threadpoolctl import threadpool_limits

import evaluate as e

P=e.OUT
GROUPS=['Weibo 改对','Weibo 新错','SMB 救回后又丢']
RULES=['分歧时始终保留逐包','逐包绝对分数更大','有观测到的换向且每次换向都跨缺口','换向至多一次','逐包绝对分数更大且换向至多一次','逐包判 SMB 且绝对分数更大','逐包判 Weibo 且绝对分数更大']


def observation(r,q):
    visible=np.ones(len(r['s']),bool)
    visible[q]=False
    pos=np.flatnonzero(visible)
    dt=np.diff(np.asarray(r['t'],np.int64)[pos])
    gap=np.diff(pos)-1
    turns=np.diff(np.sign(np.asarray(r['s'])[pos]))!=0
    sizes=np.diff(np.r_[0,np.flatnonzero(turns)+1,len(pos)])
    def median(a):
        return float(np.median(a)) if len(a) else None
    return {'可见包':len(pos),'收发段':len(sizes),'最长段包数':int(max(sizes)),
        '段内间隔中位数 μs':median(dt[(~turns)&(dt>=0)]),'换向间隔中位数 μs':median(dt[turns&(dt>=0)]),
        '段内跨过缺包总数':int(gap[~turns].sum()),'换向跨过缺包总数':int(gap[turns].sum()),
        '每次换向都跨缺口':bool(turns.any() and (gap[turns]>0).all())}


def contributions(model,x):
    """Count-weighted descendant leaf expectation; telescope on each decision path."""
    const=float(model._baseline_prediction.item())
    out=np.zeros(x.shape)
    for (tree,) in model._predictors:
        ns=tree.nodes
        assert not ns['is_categorical'].any()
        ev=np.array(ns['value'])
        for i in reversed(range(len(ns))):
            n=ns[i]
            if not n['is_leaf']:
                a,b=n['left'],n['right']
                ev[i]=(ev[a]*ns[a]['count']+ev[b]*ns[b]['count'])/n['count']
        const+=ev[0]
        at=np.zeros(len(x),int)
        while True:
            rows=np.flatnonzero(ns['is_leaf'][at]==0)
            if not len(rows): break
            node=ns[at[rows]]
            col=node['feature_idx'].astype(int)
            value=x[rows,col]
            go_left=np.where(np.isnan(value),node['missing_go_to_left'],value<=node['num_threshold'])
            child=np.where(go_left,node['left'],node['right'])
            out[rows,col]+=ev[child]-ev[at[rows]]
            at[rows]=child
    assert np.allclose(const+out.sum(axis=1),model.decision_function(x))
    return const,out


def features_by_group(width,kept,vec):
    summary=['流级时间统计']*19+['方向转换与窗口统计']*28
    packet=(['逐包方向','逐包间隔','逐包缺包数']*(width//3))
    groups=list(np.array(summary+packet)[kept])
    for key in vec.get_feature_names_out():
        part=key.split('_',1)[1]
        if part=='incoming_dt': tag='段间换向间隔'
        elif part=='incoming_missing': tag='段间缺包数'
        elif 'dt' in part: tag='段内时间间隔'
        elif 'gap' in part or part=='missing': tag='段内缺包数'
        else: tag='段方向和可见包数'
        groups.append(tag)
    return np.asarray(groups)


def table(headers,rows):
    def cells(tag,line):
        return ''.join(f'<{tag}>{escape(str(x))}</{tag}>' for x in line)
    return '<table><thead><tr>'+cells('th',headers)+'</tr></thead><tbody>'+''.join('<tr>'+cells('td',r)+'</tr>' for r in rows)+'</tbody></table>'


def main():
    with np.load(P/'paired.npz') as z:
        mask=z['fold']==0
        ids,y,base,old,joint=[z[k][mask] for k in ('ids','y','summary50','ordered50','combined50')]
    rmap={r['id']:r for r in e.old.read_panels()['train']}
    rows=[rmap[i] for i in ids]
    assert {e.old.file_of(r) for r in rows}==e.old.FILES[0]
    q=[e.old.fixed_mask(r,50) for r in rows]
    x=e.prepare(rows,q)
    width,kept,_,om=joblib.load(e.seg.PREV/'downstream_heads.joblib')[0]
    vec=joblib.load(e.PREV/'segment_heads.joblib')[0][0]
    jm=joblib.load(P/'combined_heads.joblib')[0]
    xx=e.matrix(x,(width,kept,vec)); ox=xx[:,:kept.sum()]
    so=om.decision_function(ox); sj=jm.decision_function(xx)
    assert np.array_equal(om.predict(ox),old) and np.array_equal(jm.predict(xx),joint)
    obs=[observation(r,h) for r,h in zip(rows,q)]
    sets=[(y==0)&(old==1)&(joint==0),(y==0)&(old==0)&(joint==1),
          (y==1)&(base==0)&(old==1)&(joint==0)]
    assert [s.sum() for s in sets]==[32,31,33]
    use=np.logical_or.reduce(sets)
    obscols=list(obs[0])[:-1]
    desc=[]
    for key in obscols:
        vals=[[obs[i][key] for i in np.flatnonzero(s) if obs[i][key] is not None] for s in sets]
        desc.append([key,*[f'{np.median(v):g} [{np.percentile(v,25):g}, {np.percentile(v,75):g}] (n={len(v)})' if v else '无' for v in vals]])
    o0,oc=contributions(om,ox[use])
    j0,jc=contributions(jm,xx[use])
    jc[:,:oc.shape[1]]-=oc
    group=features_by_group(width,kept,vec)
    deltas={k:jc[:,group==k].sum(1) for k in dict.fromkeys(group)}
    deltas['模型基准分']=np.full(use.sum(),j0-o0)
    assert np.allclose(sum(deltas.values()),sj[use]-so[use])
    drivers=[[k,*[round(float(np.mean(v[s[use]])),3) for s in sets]] for k,v in deltas.items()]
    agg=[*(f'可见间隔{i+1}' for i in range(12)),'流均值','流标准差','流最小值','流Q1','流中位数','流Q3','流最大值']
    for label in ('负负','负正','正负','正正','前1/3','中1/3','后1/3'):
        agg.extend(label+z for z in ('比例','标准化间隔Q1','标准化间隔中位数','标准化间隔Q3'))
    pkt=[f'可见包{i+1}{field}' for i in range(width//3) for field in ('方向','入间隔','中间缺数')]
    names=list(np.asarray(agg+pkt)[kept])+list(vec.get_feature_names_out())
    specific=[]
    for name,s in zip(GROUPS,sets):
        avg=jc[s[use]].mean(axis=0)
        specific.append([name,*[f'{names[i]} {avg[i]:+.3f}' for i in np.argsort(-abs(avg))[:5]]])
    logits=[[round(float(v[s].mean()),3) for s in sets] for v in (so,sj,sj-so)]
    disagree=old!=joint
    turns=np.array([z['收发段']-1 for z in obs])
    strong=abs(so)>abs(sj)
    choices=[disagree,strong,np.array([z['每次换向都跨缺口'] for z in obs]),turns<=1,strong&(turns<=1),strong&(old==1),strong&(old==0)]
    checks=[]
    for rule,choice in zip(RULES,choices):
        switch=choice&disagree
        win=switch&(old==y);loss=switch&(joint==y)
        checks.append([rule,int(switch.sum()),*[int((switch&s).sum()) for s in sets],
            int(win.sum()),int(loss.sum()),int(win.sum()-loss.sum()),
            int((win&(y==1)).sum()-(loss&(y==1)).sum()),int((win&(y==0)).sum()-(loss&(y==0)).sum())])
    # Show whether same apparent margin rule behaves symmetrically in both directions.
    directions=[]
    for direction in (0,1):
        switch=strong&disagree&(old==direction)
        directions.append([int(direction),int(switch.sum()),int((switch&(old==y)).sum()),int((switch&(joint==y)).sum())])
    records=[]
    for k,i in enumerate(np.flatnonzero(use)):
        records.append(dict(id=ids[i],group=next(name for name,s in zip(GROUPS,sets) if s[i]),
            ordered=float(so[i]),combined=float(sj[i]),visible=obs[i],delta={name:float(v[k]) for name,v in deltas.items()}))
    summary=dict(descriptive=desc,drivers=drivers,specific=specific,mean_scores=logits,checks=checks,directions=directions,
        n=len(y),disagree=int(disagree.sum()),records=records)
    (P/'audit_A50.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding='utf-8')
    print('descriptive',desc,'drivers',drivers,'means',logits,'checks',checks,'directions',directions,'all',disagree.sum(),flush=True)
    body='<h1>50% 丢包 A 折逐流翻判复核</h1><p>正分预测 SMB，负分预测 Weibo。可见数据来自原流和原掩码。区间为各流描述量的第 25 至 75 百分位。</p>'
    body+=table(['可见条件',*GROUPS],desc)
    body+='<h2>分数变化</h2><p>两棵树集成分别从训练计数加权的叶分数均值沿实际路径累计变化；下表取联合减逐包的组平均差。特征相关时路径贡献随模型和分组变化。</p>'
    body+=table(['SMB 对数几率',*GROUPS],[[t,*v] for t,v in zip(('逐包','联合','差'),logits)]+drivers)
    body+='<h2>每组平均分数变化最大的具体特征</h2><p>r0 为第一段；dt0 为段内第一对可见包间隔。标准化间隔为相邻可见包时间差除以槽位跨度后取 log1p。</p>'+table(['组','1','2','3','4','5'],specific)
    body+='<h2>候选保留条件，全部 A 折分歧</h2>'+table(['条件','触发','撤销改对 Weibo','救回新错 Weibo','救回目标 SMB','总救回','总损失','净多对','SMB 净多对','Weibo 净多对'],checks)
    body+='<h2>96 条流的模型分数及首要路径贡献</h2>'+table(['流 ID','组','逐包分','联合分','绝对值最大三项'],[[r['id'],r['group'],round(r['ordered'],3),round(r['combined'],3),
        '，'.join(f'{n} {v:+.2f}' for n,v in sorted(r['delta'].items(),key=lambda z:-abs(z[1]))[:3])] for r in records])
    (P/'A50_flip_audit.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A 折翻判</title><style>body{font:16px system-ui;margin:3% auto;max-width:1300px}table{border-collapse:collapse;border-block:2px solid}thead{border-bottom:1px solid}td,th{padding:.4em .65em;text-align:left}</style>'+body+'</html>',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=3):main()
