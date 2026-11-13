"""Visible-context signed-length head gated by the frozen dual-path prediction."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

def features(visible, positions):
    # Input uses zero only for a hidden slot. No labels or hidden values enter here.
    n=len(visible)
    at=np.flatnonzero(visible)
    val=visible[at]
    cut=np.searchsorted(at,positions)
    cols=[positions,positions/max(n-1,1),np.full(len(positions),n)]
    for delta in [-4,-3,-2,-1,0,1,2,3]:
        k=cut+delta
        ok=(k>=0)&(k<len(at))
        k=k.clip(0,len(at)-1)
        cols.extend([np.where(ok,val[k],0),np.where(ok,at[k]-positions,0)])
    for radius in (8,32,n):
        lo=np.searchsorted(at,positions-radius)
        hi=np.searchsorted(at,positions+radius,side='right')
        for mask in [val>0,val<0,(val>0)&(val<=128),(val<0)&(val>=-128),abs(val)>=1400]:
            c=np.r_[0,np.cumsum(mask)]
            cols.append((c[hi]-c[lo])/np.maximum(1,hi-lo))
    for side in [1,-1]:
        short=val[(side*val>0)&(side*val<=128)]
        mode=np.bincount(abs(short)).argmax()*side if len(short) else 0
        cols.append(np.full(len(positions),mode))
    return np.column_stack(cols).astype(np.float32)

def fit(rows):
    x,y=[],[]
    for row in rows:
        s=np.array(row['s'])
        q=np.array(next(iter(row['masks'].values())))
        short=q[(abs(s[q])<=128)&(s[q]!=0)]
        if not len(short): continue
        vis=s.copy()
        vis[q]=0
        x.append(features(vis,short))
        y.extend(s[short])
    y=np.array(y)
    codes,counts=np.unique(y,return_counts=True)
    common=codes[np.argsort(-counts,kind='stable')[:48]]
    keep=np.isin(y,common)
    model=ExtraTreesClassifier(n_estimators=64,max_depth=14,min_samples_leaf=3,
                               random_state=17,n_jobs=4)
    model.fit(np.concatenate(x)[keep],y[keep])
    return model,dict(train_flows=len(rows),short_targets=len(y),fitted_targets=int(keep.sum()))

def apply(model,rows,positions,base):
    out=base.copy()
    chunks=[];indices=[]
    off=0
    for row,q in zip(rows,positions,strict=True):
        q=np.asarray(q,dtype=int)
        idx=np.flatnonzero((abs(base[off:off+len(q)])<=128)&(base[off:off+len(q)]!=0))
        if len(idx):
            vis=np.array(row)
            vis[q]=0
            chunks.append(features(vis,q[idx]))
            indices.extend(off+idx)
        off+=len(q)
    assert off==len(base)
    if chunks:
        x=np.concatenate(chunks)
        for start in range(0,len(x),2048):
            p=model.predict_proba(x[start:start+2048])
            k=p.argmax(axis=1)
            ok=p[np.arange(len(k)),k]>=0.45
            target=np.array(indices[start:start+len(k)])[ok]
            out[target]=model.classes_[k[ok]]
    assert np.array_equal(out[abs(base)>128],base[abs(base)>128])
    return out
