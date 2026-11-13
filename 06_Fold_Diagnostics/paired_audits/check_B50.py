import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import evaluate as e


def main():
    with np.load(e.OUT/'paired.npz') as data:
        select = data['fold'] == 1
        ids, y, base, packet, joint = [
            data[k][select] for k in ('ids','y','summary50','ordered50','combined50')]
    lookup = {r['id']:r for r in e.old.read_panels()['train']}
    rows = [lookup[s] for s in ids]
    assert len(set(ids)) == len(ids) and {e.old.file_of(r) for r in rows} == e.old.FILES[1]
    assert np.array_equal(y,[e.old.LABEL[e.old.file_of(r)] for r in rows])
    masks = [e.old.fixed_mask(r,50) for r in rows]
    width, used, _, ordered_head = joblib.load(e.seg.PREV/'downstream_heads.joblib')[1]
    vectorizer = joblib.load(e.PREV/'segment_heads.joblib')[1][0]
    joint_head = joblib.load(e.OUT/'combined_heads.joblib')[1]
    x = e.matrix(e.prepare(rows,masks),(width,used,vectorizer))
    x_packet = x[:, :used.sum()]
    a = ordered_head.decision_function(x_packet)
    b = joint_head.decision_function(x)
    assert np.array_equal(ordered_head.predict(x_packet),packet) and np.array_equal(joint_head.predict(x),joint)
    trigger = (packet==1) & (joint==0) & (np.abs(a)>np.abs(b))
    after = np.where(trigger,packet,joint)
    target = (y==1) & (base==0) & (packet==1)
    assert target.sum()==19 and (target & (joint==1)).sum()==11
    before_stats=e.old.metrics(y,joint,2)
    after_stats=e.old.metrics(y,after,2)
    smb=int((trigger & (y==1)).sum())
    weibo=int((trigger & (y==0)).sum())
    assert before_stats['sw']-after_stats['sw']==smb and after_stats['ws']-before_stats['ws']==weibo
    result=dict(n=len(ids),smb=int(y.sum()),weibo=int((y==0).sum()),trigger=int(trigger.sum()),rescued_smb=smb,
        extra_weibo_errors=weibo,target_before=11,target_added=int((target & trigger).sum()),target_after=int((target & (after==1)).sum()),
        before=before_stats,after=after_stats)
    np.savez_compressed(e.OUT/'B50_fallback_predictions.npz',ids=ids,y=y,ordered_score=a,joint_score=b,
        target=target,trigger=trigger,ordered=packet,before=joint,after=after)
    (e.OUT/'B50_fallback.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=3):
        main()
