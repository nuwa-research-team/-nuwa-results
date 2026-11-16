"""Bounded Luna prompt study; frozen paired masks, train-only retrieval, no fitting."""
import argparse, concurrent.futures as cf, csv, html, json, math, msvcrt, os
import pickle, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import compare as api

ROOT=api.ROOT
DATA=api.DATA
OUT=DATA/'luna_prompt_dev200_20261116'
MODEL='gpt-6-luna'
ARMS=('P0','P1','P2','P3')
sys.path.insert(0,str(DATA))
import experiment as exp

LOCAL=(' Local contexts are redundant views of the same observed input. Each index and distance '
       'uses original packet slots, including missing slots. Consecutive missing slots are not '
       'observations. Use both sides when available; a missing boundary is unknown. Infer each '
       'packet direction and then its magnitude using the local send/receive pattern. Do not '
       'assume every gap repeats its neighbors. Return values only for the target missing slots, '
       'in exactly the listed order; do not return explanations or observed packets.')
BEHAVIOR=(' Behavioral summaries use observed packets only, and include support counts. '
          'Global common lengths are evidence, not mandatory outputs. Prefer relevant local '
          'patterns over the global majority when they conflict. Runs end at a missing slot '
          'or an observed direction change; do not treat a gap as proof of continuity.')
EXAMPLES=(' Examples are other training flows with their own masks and answers. They illustrate '
          'the task, not the target answer. Infer the target independently; do not copy an '
          'example answer or its number of slots. Only the target requires an output.')

def visible(r):
    s=list(map(int,r['truth']))
    for p in r['pos']: s[p]=None
    return s

def local(s):
    result=[]
    obs=[i for i,v in enumerate(s) if v is not None]
    for p,v in enumerate(s):
        if v is not None:continue
        left=[i for i in obs if i<p][-3:]
        right=[i for i in obs if i>p][:3]
        lo=p;hi=p
        while lo>0 and s[lo-1] is None:lo-=1
        while hi+1<len(s) and s[hi+1] is None:hi+=1
        result.append({'index':p,'gap_start':lo,'gap_size':hi-lo+1,
                       'left':[[i,s[i],p-i] for i in left],
                       'right':[[i,s[i],i-p] for i in right]})
    return result

def summary(s):
    vs=[v for v in s if v is not None]
    pairs=[(a,b) for a,b in zip(s,s[1:]) if a is not None and b is not None]
    runs=[];start=None;sign=None
    for i,v in enumerate(s+[None]):
        sg=None if v is None else int(v>0)
        if start is not None and sg!=sign:
            runs.append([start,i-1,1 if sign else -1,i-start]);start=None
        if sg is not None and start is None:start=i;sign=sg
    def block(xs):
        vv=[x for x in xs if x is not None]
        return {'slots':len(xs),'observed':len(vv),'positive':sum(x>0 for x in vv),
                'negative':sum(x<0 for x in vv),'frequent_signed':Counter(vv).most_common(4)}
    return {'global':block(s),'positive_magnitude_counts':Counter(v for v in vs if v>0).most_common(5),
            'negative_magnitude_counts':Counter(-v for v in vs if v<0).most_common(5),
            'observed_adjacent_pairs':len(pairs),'same_sign_pairs':sum((a>0)==(b>0) for a,b in pairs),
            'observed_runs_start_end_direction_count':runs,
            'windows16':[dict(start=i,**block(s[i:i+16])) for i in range(0,len(s),16)]}

def feature(s):
    vs=np.array([x for x in s if x is not None],dtype=float)
    bins=np.array([0,64,128,256,512,1024,1400,1601])
    hist=[]
    for sign in (1,-1):
        hist.extend((np.histogram(np.abs(vs[vs*sign>0]),bins)[0]/max(1,len(vs))).tolist())
    pairs=[(a,b) for a,b in zip(s,s[1:]) if a is not None and b is not None]
    return np.array(hist+[math.log2(len(s))/9,len(vs)/len(s),
        sum((a>0)==(b>0) for a,b in pairs)/max(1,len(pairs))],dtype=float)

def payload(r,arm,examples=None):
    s=visible(r)
    obj={'signed':s,'missing_positions_zero_based':r['pos'],'missing_count':len(r['pos'])}
    if arm!='P0':
        obj['local_context_triplets_index_signed_length_distance']=local(s)
    if arm in ('P2','P3'):obj['behavior']=summary(s)
    if arm=='P3':obj={'examples':examples,'target':obj}
    base=api.read(DATA/'protocol.json')['prompts']['plain']
    system=base+(LOCAL if arm!='P0' else '')+(BEHAVIOR if arm in ('P2','P3') else '')+(EXAMPLES if arm=='P3' else '')
    return {'model':MODEL,'messages':[{'role':'system','content':system},
        {'role':'user','content':json.dumps(obj,separators=(',',':'))}],
        'max_completion_tokens':4096,'reasoning_effort':'low','stream':False,
        'response_format':{'type':'json_object'}}

def pick(rows,n,seed):
    """Proportionate six-label allocation by largest remainder, independent of outcomes."""
    rng=np.random.default_rng(seed);counts=Counter(r['label'] for r in rows)
    raw={k:n*v/len(rows) for k,v in counts.items()};alloc={k:int(v) for k,v in raw.items()}
    for k in sorted(raw,key=lambda k:(-(raw[k]-alloc[k]),k))[:n-sum(alloc.values())]:alloc[k]+=1
    ix=[]
    for k in sorted(alloc):ix.extend(rng.choice([i for i,r in enumerate(rows) if r['label']==k],alloc[k],False).tolist())
    return sorted(ix)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        for p,h in api.read(OUT/'manifest.json')['files'].items():assert api.sha(Path(p).read_bytes())==h,p
        return
    for p,h in api.read(DATA/'commandcode_compare_v1/execution_manifest.json')['files'].items():
        assert api.sha(Path(p).read_bytes())==h,p
    a=api.read(DATA/'local_A.json');b=api.read(DATA/'local_B.json')
    devix=pick(a,200,2026111601);dev=[a[i] for i in devix]
    rawpath=exp.SNAP/'work/d4_aligned/raw.pkl';rawbytes=rawpath.read_bytes()
    assert api.sha(rawbytes)==api.read(DATA/'protocol.json')['source_sha256']
    d=pickle.loads(rawbytes);test=list(map(int,d['split']['test']));train=list(map(int,d['split']['train']))
    assert not set(test)&set(train)
    excluded={r['flow'] for r in a+b}
    mapping=api.read(DATA.parent/'d4_full_aligned_20261113/six_label_audit.json')['mapping']
    with np.load(exp.PE/'mask_30.npz') as z:
        assert list(z['ids'])==[d['ids'][i] for i in test]
        masks=z['mask'].copy()
    pool=[]
    for j,i in enumerate(test):
        flow=str(d['ids'][i])
        if flow in excluded:continue
        truth=list(map(int,d['seqs'][i]));pos=np.flatnonzero(masks[j,:len(truth)]).tolist()
        pool.append(dict(draw=len(pool),flow=flow,truth=truth,pos=pos,label=mapping[flow.split('.pcap#')[0]]))
    confirm=[pool[i] for i in pick(pool,200,2026111602)]
    assert len({r['flow'] for r in dev+confirm})==400
    testseq={tuple(map(int,d['seqs'][i])) for i in test}
    bank=[];features=[];seen=set()
    for i in train:
        seq=tuple(map(int,d['seqs'][i]))
        if seq in testseq or seq in seen:continue
        seen.add(seq);pos=sorted(np.random.default_rng([2026111603,i]).choice(len(seq),max(1,round(.3*len(seq))),False).tolist())
        r={'truth':list(seq),'pos':pos,'flow':str(d['ids'][i]),'raw_index':i}
        bank.append(r);features.append(feature(visible(r)))
    features=np.stack(features);assert len(bank)>2
    provenance={};requests={}
    for stage,rows in [('dev',dev),('confirm',confirm)]:
        requests[stage]={arm:[] for arm in ARMS};provenance[stage]=[]
        for r in rows:
            distances=np.sum((features-feature(visible(r)))**2,axis=1)
            inds=np.argsort(distances,kind='stable')[:2];examples=[];ids=[]
            for i in inds:
                br=bank[int(i)];s=visible(br)
                examples.append({'signed':s,'missing_positions_zero_based':br['pos'],
                    'answer':{'values':[br['truth'][p] for p in br['pos']]}})
                ids.append({'flow':br['flow'],'raw_index':br['raw_index'],'distance':float(distances[i])})
            provenance[stage].append(ids)
            changed=dict(r,truth=r['truth'].copy())
            for p in r['pos']:changed['truth'][p]=1599
            assert np.array_equal(feature(visible(r)),feature(visible(changed)))
            for arm in ARMS:
                req=payload(r,arm,examples)
                assert req==payload(changed,arm,examples),'Hidden value leaked'
                requests[stage][arm].append(req)
    previous=api.read(DATA/'commandcode_compare_v1/requests.json')[MODEL]
    for j,i in enumerate(devix):assert requests['dev']['P0'][j]==previous[i]
    cohorts={'dev':dev,'confirm':confirm}
    api.save(OUT/'cohorts.json',cohorts);api.save(OUT/'requests.json',requests)
    api.save(OUT/'retrieval_provenance.json',provenance)
    spec={'utc':api.utc(),'model':MODEL,'mask_rate':.3,'dev_A_indices':devix,
          'dev_support':dict(Counter(r['label'] for r in dev)),
          'confirmation_support':dict(Counter(r['label'] for r in confirm)),
          'arms':{'P0':'original, archived matching requests reused on dev',
                  'P1':'P0 + nearest three observed packets per side with original-slot distances and gap lengths',
                  'P2':'P1 + observed-only direction-specific modes, adjacency, contiguous runs, 16-slot summaries and coverage',
                  'P3':'P2 + two nearest training demonstrations, visible-only histogram/length/coverage/adjacency retrieval'},
          'bank_size':len(bank),'training_demo_exclusion':'exclude all exact complete-sequence matches to test, deduplicate train sequences',
          'primary':'equal mean of six frozen classifier accuracies',
          'gate':'best dev mean_accuracy strictly exceeds P0 and macro-F1 is not lower; tie by macro-F1 then MAE then arm name',
          'confirmation':'one selected arm versus P0 and TFI on frozen 200 flows absent from prior A/B; no changes after selection',
          'budget':'600 new dev response slots; conditional 400 confirm response slots; maximum 3 transport attempts each; no invalid-answer retries',
          'decode':'4096 max completion tokens, low reasoning, JSON object, same strict validation as original',
          'invalid':'zero fill missing slots, retain every flow; observed values unchanged',
          'boundary':'historical test pool, exploratory development and held-out-for-this-prompt-study confirmation; not independent final benchmark',
          'no_training':True,'no_extra_timestamps':True,'concurrency':6}
    api.save(OUT/'protocol.json',spec)
    files={str(p):api.sha(p.read_bytes()) for p in [Path(__file__),Path(api.__file__),Path(api.__file__).with_name('client.py'),
        rawpath,OUT/'cohorts.json',OUT/'requests.json',OUT/'retrieval_provenance.json',OUT/'protocol.json',
        DATA/'A_local_tfi_predictions.json',DATA/'A_commandcode_v1_gpt_6_luna_predictions.json',exp.PE/'tfi_imputed_30.npz']}
    files.update(api.read(DATA/'commandcode_compare_v1/execution_manifest.json')['files'])
    api.save(OUT/'manifest.json',{'utc':api.utc(),'files':files})
    api.save(OUT/'preflight.json',{'passed':True,'hidden_target_mutation_invariance':True,
        'train_split_and_exact_test_sequence_exclusion':True,'P0_request_identity':True,
        'dev_hidden':sum(len(r['pos']) for r in dev),'confirm_hidden':sum(len(r['pos']) for r in confirm)})
    print('Prepared',spec['dev_support'],'training bank',len(bank),flush=True)

def metrics(rows,values,classes=None):
    from sklearn.metrics import f1_score,recall_score
    seqs=[];t=[];p=[]
    for r,v in zip(rows,values,strict=True):
        s=r['truth'].copy();v=v if v is not None else [0]*len(r['pos'])
        assert len(v)==len(r['pos'])
        for i,x in zip(r['pos'],v):t.append(s[i]);p.append(x);s[i]=x
        assert all(s[i]==r['truth'][i] for i in range(len(s)) if i not in r['pos'])
        seqs.append(s)
    if classes is None:classes=exp.classify(seqs)
    y=np.array([r['label'] for r in rows]);details={}
    for k,c in classes.items():
        assert len(c)==len(y)
        details[k]={'accuracy':float(100*np.mean(np.array(c)==y)),
            'macro_f1':100*f1_score(y,c,average='macro',labels=range(6),zero_division=0),
            'recall_by_class':(100*recall_score(y,c,average=None,labels=range(6),zero_division=0)).tolist()}
    t=np.array(t);p=np.array(p)
    return {'draws':len(rows),'hidden':len(t),'mean_accuracy':float(np.mean([v['accuracy'] for v in details.values()])),
        'mean_macro_f1':float(np.mean([v['macro_f1'] for v in details.values()])),
        'magnitude_mae':float(np.mean(np.abs(np.abs(t)-np.abs(p)))),
        'signed_mae':float(np.mean(np.abs(t-p))),'exact_signed_pct':float(100*np.mean(t==p)),
        'direction_pct':float(100*np.mean((t*p)>0)),'valid_pct':100*sum(v is not None for v in values)/len(values),
        'classifiers':details,'classes':classes}

def evaluate(stage,arm):
    path=OUT/stage/(arm+'_scores.json')
    if path.exists():return api.read(path)
    rows=api.read(OUT/'cohorts.json')[stage];classes=None
    if stage=='dev' and arm in ('P0','TFI'):
        name='A_local_tfi_predictions.json' if arm=='TFI' else 'A_commandcode_v1_gpt_6_luna_predictions.json'
        old=api.read(DATA/name);ix=api.read(OUT/'protocol.json')['dev_A_indices']
        values=[old['values'][i] for i in ix];classes={k:[cs[i] for i in ix] for k,cs in old['classes'].items()}
    elif arm=='TFI':
        with np.load(exp.PE/'tfi_imputed_30.npz') as z:lookup=dict(zip(z['ids'],z['values']))
        values=[[int(lookup[r['flow']][p]) for p in r['pos']] for r in rows]
    else:
        reqs=api.read(OUT/'requests.json')[stage][arm];values=[]
        for i,r in enumerate(rows):
            rec=api.read(OUT/stage/arm/f'{i:04d}.json')
            assert rec['request_sha256']==api.sha(api.enc(reqs[i]))
            v,error=api.validate(rec['response'],len(r['pos'])) if rec['http_status']==200 else (None,'http_failure')
            assert v==rec['values'] and error==rec['validation_error']
            values.append(v)
    result=metrics(rows,values,classes)
    api.save(OUT/stage/(arm+'_predictions.json'),{'values':values,'classes':result.pop('classes')})
    api.save(path,result);report(stage)
    print(stage,arm,'accuracy',result['mean_accuracy'],'F1',result['mean_macro_f1'],'MAE',result['magnitude_mae'],flush=True)
    return result

def report(stage):
    results={p.name.replace('_scores.json',''):api.read(p) for p in (OUT/stage).glob('*_scores.json')}
    columns=['Method','Accuracy (%)','Macro-F1 (%)','Magnitude MAE (bytes)','Signed MAE (bytes)','Valid (%)']
    table=[];detail=[['Method','Classifier','Accuracy (%)','Macro-F1 (%)']]
    for arm in ('TFI',*ARMS):
        if arm not in results:continue
        s=results[arm];table.append([arm,s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['signed_mae'],s['valid_pct']])
        for k,v in s['classifiers'].items():detail.append([arm,k,v['accuracy'],v['macro_f1']])
    for name,rs in [('summary.csv',[columns,*table]),('classifiers.csv',detail)]:
        with (OUT/stage/name).open('w',newline='',encoding='utf-8-sig') as f:csv.writer(f).writerows(rs)
    def tab(rs):return '<table>'+''.join('<tr>'+''.join('<'+('th' if j==0 else 'td')+'>'+html.escape(f'{v:.3f}' if isinstance(v,float) else str(v))+'</'+('th' if j==0 else 'td')+'>' for v in row)+'</tr>' for j,row in enumerate(rs))+'</table>'
    (OUT/stage/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px Arial;margin:40px}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:25px 0}th{border-bottom:1px solid}td,th{padding:10px}</style><h1>Luna prompt study: '+stage+'</h1><p>200 paired D4 flows, original 30% masks. Accuracy and macro-F1: equal mean of six frozen classifiers. Invalid answers retained with zero fill. P0 original; P1 local context; P2 adds behavior; P3 adds two train-only examples. Historical test pool: exploratory, not an independent final benchmark.</p>'+tab([columns,*table])+tab(detail),encoding='utf-8')

def run_api(stage,arms):
    api.OUT=OUT/stage;api.STOP.clear()
    reqs=api.read(OUT/'requests.json')[stage];rows=api.read(OUT/'cohorts.json')[stage]
    secret=api.key();jobs=[]
    for i in range(len(rows)):
        for arm in arms:
            path=OUT/stage/arm/f'{i:04d}.json'
            if path.exists():assert api.read(path)['request_sha256']==api.sha(api.enc(reqs[arm][i]))
            else:jobs.append((arm,i))
    start=time.time();initial=len(jobs);completed=0;pending={};it=iter(jobs)
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        while True:
            while len(pending)<6 and not api.STOP.is_set():
                job=next(it,None)
                if job is None:break
                arm,i=job
                pending[pool.submit(api.fetch,arm,i,reqs[arm][i],len(rows[i]['pos']),secret)]=job
            counts={a:sum((OUT/stage/a/f'{i:04d}.json').exists() for i in range(len(rows))) for a in arms}
            elapsed=time.time()-start
            api.save(OUT/'progress.json',{'utc':api.utc(),'pid':os.getpid(),'stage':stage,'counts':counts,
                'inflight':len(pending),'elapsed_seconds':elapsed,
                'remaining_seconds':(initial-completed)*elapsed/completed if completed else None,
                'state':'blocked' if api.STOP.is_set() else 'running'})
            if not pending:break
            done,_=cf.wait(pending,timeout=15,return_when=cf.FIRST_COMPLETED)
            for f in done:
                arm,i=pending.pop(f)
                try:f.result();completed+=1
                except Exception as exc:api.STOP.set();print('STOP',type(exc).__name__,str(exc),flush=True)
    if api.STOP.is_set():raise RuntimeError('Provider block; preserve completed work')
    for arm in arms:evaluate(stage,arm)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    prepare()
    if args.prepare:return
    api.save(OUT/'started.json',{'utc':api.utc(),'pid':os.getpid()})
    evaluate('dev','P0');evaluate('dev','TFI')
    run_api('dev',('P1','P2','P3'))
    scores={a:api.read(OUT/'dev'/(a+'_scores.json')) for a in ARMS}
    eligible=[a for a in ARMS[1:] if scores[a]['mean_accuracy']>scores['P0']['mean_accuracy'] and scores[a]['mean_macro_f1']>=scores['P0']['mean_macro_f1']]
    chosen=min(eligible,key=lambda a:(-scores[a]['mean_accuracy'],-scores[a]['mean_macro_f1'],scores[a]['magnitude_mae'],a)) if eligible else None
    decision={'utc':api.utc(),'selected':chosen,'rule':api.read(OUT/'protocol.json')['gate'],'eligible':eligible}
    api.save(OUT/'selection.json',decision)
    if chosen:
        evaluate('confirm','TFI');run_api('confirm',('P0',chosen))
    api.save(OUT/'complete.json',{'utc':api.utc(),'selected':chosen,'confirmation_run':bool(chosen)})
    api.save(OUT/'progress.json',{'utc':api.utc(),'state':'complete','selected':chosen})

if __name__=='__main__':main()
