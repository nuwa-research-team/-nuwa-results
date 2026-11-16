"""Frozen D4 A comparison: two CommandCode teachers, no fitting or prompt selection."""
import argparse, concurrent.futures as cf, hashlib, json, msvcrt, os
from pathlib import Path
import subprocess, sys, threading, time, urllib.request, urllib.error
from datetime import datetime, timezone
from client import BASE, key

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'outputs/d4_teacher_api_20261114'
OUT=DATA/'commandcode_compare_v1'
MODELS=('gpt-6-luna','gpt-6-astra')
STOP=threading.Event()
def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_bytes())
def enc(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(enc(v))
    # Windows readers can briefly hold a destination handle without delete sharing.
    for attempt in range(50):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)
def body(model,google):
    return {'model':model,'messages':[
        {'role':'system','content':google['systemInstruction']['parts'][0]['text']},
        {'role':'user','content':google['contents'][0]['parts'][0]['text']}],
        'max_completion_tokens':4096,'reasoning_effort':'low','stream':False,
        'response_format':{'type':'json_object'}}
def validate(res,n):
    choices=res.get('choices',[])
    if not choices:return None,'no_candidate'
    c=choices[0]
    if c.get('finish_reason')!='stop':return None,'finish_'+str(c.get('finish_reason'))
    try:v=json.loads(c.get('message',{}).get('content',''))
    except (ValueError,TypeError):return None,'invalid_json'
    if not isinstance(v,dict) or set(v)!={'values'}:return None,'wrong_keys'
    vs=v['values']
    if not isinstance(vs,list) or len(vs)!=n:return None,'wrong_count'
    if not all(type(x) is int and 0<abs(x)<=1600 for x in vs):return None,'invalid_integer'
    return vs,None
def prepare():
    old=read(DATA/'cross_teacher_v1/execution_manifest.json')['files']
    for p,h in old.items():assert sha(Path(p).read_bytes())==h, 'Frozen file changed: '+p
    rows=read(DATA/'local_A.json');protocol=read(DATA/'protocol.json')
    assert sha((DATA/'local_A.json').read_bytes())==protocol['hashes']['A']
    assert len(rows)==len({r['flow'] for r in rows})==1000
    assert [r['draw'] for r in rows]==list(range(1000))
    assert sum(len(r['pos']) for r in rows)==10488
    google=read(DATA/'gemini_compare_v2/requests.json')
    assert len(google)==1000
    for r,g in zip(rows,google):
        seq=r['truth'].copy()
        assert r['pos']==sorted(set(r['pos']))
        for pos in r['pos']:seq[pos]=None
        expected=json.dumps({'signed':seq,'missing_positions_zero_based':r['pos'],'missing_count':len(r['pos'])},separators=(',',':'))
        assert g['contents'][0]['parts'][0]['text']==expected
        assert g['systemInstruction']['parts'][0]['text']==protocol['prompts']['plain']
    requests={m:[body(m,g) for g in google] for m in MODELS}
    spec={'models':MODELS,'cohort':'A','draws_per_model':1000,'unique_flows':1000,'mask_rate':.3,'hidden_slots':10488,
          'request_hashes':{m:sha(enc(bs)) for m,bs in requests.items()},'concurrency_per_model':6,
          'response_slots':2000,'attempts_per_slot_max':3,'timeout_seconds':180,
          'retries':'Only transport/429/5xx; no retry of HTTP 200 invalid answers. Auth, payment or configuration errors stop dispatch.',
          'invalid':'Retain every draw, fill hidden slots with zero. Visible values unchanged.',
          'primary':'Equal-weight mean Accuracy of the six frozen classifiers; macro-F1 and MAE also reported.',
          'boundary':'Historical test pool used for exploratory teacher selection; not independent final evaluation.',
          'platform':'CommandCode reports model identities; upstream identity not independently attested. JSON-object mode plus strict local validation; Gemini used native schema. Token limits do not equalize compute.'}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==json.loads(enc(spec))
    else:save(OUT/'protocol.json',spec)
    save(OUT/'requests.json',requests)
    files=dict(old)
    for p in [Path(__file__),Path(__file__).with_name('report_compare.py'),DATA/'gemini_compare_v2/requests.json',DATA/'cross_teacher_v1/A/comparison.json',OUT/'protocol.json',OUT/'requests.json']:
        files[str(p)]=sha(p.read_bytes())
    if (OUT/'execution_manifest.json').exists():assert read(OUT/'execution_manifest.json')['files']==files
    else:save(OUT/'execution_manifest.json',{'utc':utc(),'files':files})
    fake=lambda vs:{'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'values':vs})}}]}
    assert validate(fake([1500,-52]),2)==([1500,-52],None)
    for vs in ([True,-52],[0,-52],[1601,-52],[1.0,-52],[1500]):assert validate(fake(vs),2)[0] is None
    return rows,requests

def fetch(model,draw,b,n,secret):
    path=OUT/model/f'{draw:04d}.json';rh=sha(enc(b))
    if path.exists():
        rec=read(path);assert rec['request_sha256']==rh
        return rec
    journal=OUT/model/f'{draw:04d}.attempts.json'
    attempts=read(journal) if journal.exists() else []
    response={};status=0
    while len(attempts)<3 and not STOP.is_set():
        start=time.time();a={'started_utc':utc(),'attempt':len(attempts)+1};attempts.append(a);save(journal,attempts)
        req=urllib.request.Request(BASE+'/chat/completions',data=enc(b),headers={
            'Content-Type':'application/json','User-Agent':'NIDS-Research-CommandCode/1.0','Authorization':'Bearer '+secret})
        try:
            with urllib.request.urlopen(req,timeout=180) as r:status=r.status;response=json.load(r)
            a['http_status']=status
        except urllib.error.HTTPError as exc:
            status=exc.code;a['http_status']=status
            a['error']=exc.read().decode(errors='replace').replace(secret,'[REDACTED]')[:1500]
        except Exception as exc:
            status=0;a['http_status']=0;a['error']=(type(exc).__name__+': '+str(exc)).replace(secret,'[REDACTED]')[:1000]
        a['seconds']=time.time()-start;a['finished_utc']=utc();save(journal,attempts)
        if status==200:break
        if status in (400,401,402,403,404,422):
            STOP.set();save(OUT/'blocked.json',{'utc':utc(),'model':model,'draw':draw,'attempt':a})
            raise RuntimeError(f'Provider blocked HTTP {status}; see redacted blocked.json')
        if len(attempts)<3:STOP.wait(2**len(attempts))
    if STOP.is_set() and status!=200:raise RuntimeError('Stopped without terminal response')
    values,error=validate(response,n) if status==200 else (None,'http_failure')
    rec={'utc':utc(),'model':model,'draw':draw,'request_sha256':rh,'http_status':status,'response':response,'values':values,'validation_error':error,'attempts':attempts}
    save(path,rec);return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));args=ap.parse_args()
    active=tuple(args.models)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    rows,requests=prepare()
    print('Preflight PASS: 1000 identical flows/messages, 10488 masks, frozen classifiers and baseline hashes',flush=True)
    if args.prepare:return
    secret=key();start=time.time();done={m:{} for m in active};pending={};scorers={};logs=[]
    for m in active:
        for i in range(1000):
            p=OUT/m/f'{i:04d}.json'
            if p.exists():
                r=read(p);assert r['request_sha256']==sha(enc(requests[m][i]));done[m][i]=r
    initial={m:len(done[m]) for m in active}
    save(OUT/'started.json',{'utc':utc(),'pid':os.getpid(),'active_models':active,'initial_completed':initial})
    # First actual sample per model tests endpoint options; retained as draw 0.
    for m in active:
        if 0 not in done[m]:done[m][0]=fetch(m,0,requests[m][0],len(rows[0]['pos']),secret)
        print(m,'first response HTTP',done[m][0]['http_status'],'validation',done[m][0]['validation_error'],flush=True)
    todo={m:iter([i for i in range(1000) if i not in done[m]]) for m in active}
    def progress(state):
        elapsed=time.time()-start;info={}
        for m in active:
            count=len(done[m]);delta=count-initial[m];rate=delta/elapsed if elapsed else 0
            info[m]={'completed':count,'total':1000,'valid':sum(r['values'] is not None for r in done[m].values()),
                'http_success':sum(r['http_status']==200 for r in done[m].values()),'inflight':sum(x[0]==m for x in pending.values()),
                'elapsed_seconds':elapsed,'samples_per_minute':rate*60,'api_remaining_seconds':(1000-count)/rate if rate else None,
                'scored':(OUT/m/'scores.json').exists()}
        save(OUT/'progress.json',{'utc':utc(),'pid':os.getpid(),'state':state,'models':info,'excluded_models':[m for m in MODELS if m not in active]})
        print(json.dumps({'state':state,'elapsed_s':round(elapsed),'counts':{m:x['completed'] for m,x in info.items()} }),flush=True)
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        while True:
            for m in active:
                while not STOP.is_set() and sum(x[0]==m for x in pending.values())<6:
                    i=next(todo[m],None)
                    if i is None:break
                    pending[pool.submit(fetch,m,i,requests[m][i],len(rows[i]['pos']),secret)]=(m,i)
                if len(done[m])==1000 and m not in scorers:
                    f=(OUT/(m+'_evaluation.log')).open('a',encoding='utf-8');logs.append(f)
                    scorers[m]=subprocess.Popen([sys.executable,str(Path(__file__).with_name('report_compare.py')),'--evaluate',m],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            progress('blocked' if STOP.is_set() else 'running')
            if not pending:break
            ready,_=cf.wait(pending,timeout=20,return_when=cf.FIRST_COMPLETED)
            for future in ready:
                m,i=pending.pop(future)
                try:done[m][i]=future.result()
                except Exception as exc:STOP.set();print(type(exc).__name__+': '+str(exc),flush=True)
            # Bound status/log churn while keeping slots occupied.
            time.sleep(.2)
    if STOP.is_set():progress('blocked');return
    progress('evaluating')
    for m,p in scorers.items():
        if p.wait()!=0:raise RuntimeError('Classifier evaluation failed for '+m)
    subprocess.run([sys.executable,str(Path(__file__).with_name('report_compare.py'))],check=True,cwd=ROOT)
    save(OUT/('complete.json' if all((OUT/m/'scores.json').exists() for m in MODELS) else 'active_models_complete.json'),
         {'utc':utc(),'active_models':active,'draws_this_scope':1000*len(active),'seconds_this_session':time.time()-start})
    progress('complete')
if __name__=='__main__':main()
