"""Fixed B-cohort prompt comparison, gated on completed A-cohort selection."""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import gemini_compare_v2 as g

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'gemini_prompt_B_v1'
VARIANTS=('plain','behavior')
CPU_PYTHON=ROOT.parents[1]/'.venv/Scripts/python.exe'
STOP=threading.Event()

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())

def body_for(row,variant,protocol):
    body=g.body_for(row,protocol['prompts'][variant])
    payload=json.loads(body['contents'][0]['parts'][0]['text'])
    if variant=='behavior':
        seq=payload['signed']
        visible=[v for v in seq if v is not None]
        pairs=[(a,b) for a,b in zip(seq,seq[1:]) if a is not None and b is not None]
        same=sum((a>0)==(b>0) for a,b in pairs)
        payload['summary']={'frequent':Counter(visible).most_common(5),
            'positive':sum(v>0 for v in visible),'negative':sum(v<0 for v in visible),
            'adjacent_same':same,'adjacent_switch':len(pairs)-same}
        body['contents'][0]['parts'][0]['text']=json.dumps(payload,separators=(',',':'))
    return body

def prepare():
    old=read(ROOT/'protocol.json')
    rows=read(ROOT/'local_B.json')
    assert g.sha((ROOT/'local_B.json').read_bytes())==old['hashes']['B']
    assert len(rows)==1000 and sum(len(r['pos']) for r in rows)==10060
    assert len({r['flow'] for r in rows})==967
    assert [r['draw'] for r in rows]==list(range(1000))
    bodies={v:[body_for(r,v,old) for r in rows] for v in VARIANTS}
    for row,a,b in zip(rows,bodies['plain'],bodies['behavior']):
        assert a['generationConfig']==b['generationConfig']
        pa=json.loads(a['contents'][0]['parts'][0]['text'])
        pb=json.loads(b['contents'][0]['parts'][0]['text']);pb.pop('summary')
        assert pa==pb
        changed=dict(row,truth=row['truth'].copy(),label=-999,flow='unavailable')
        for p in row['pos']:changed['truth'][p]=-123
        for v in VARIANTS:
            assert body_for(row,v,old)==body_for(changed,v,old), 'Hidden truth or identity leaked'
    # Gaps are not collapsed when counting adjacent visible transitions.
    sample={'truth':[1500,40,-52,1500],'pos':[1]}
    s=json.loads(body_for(sample,'behavior',old)['contents'][0]['parts'][0]['text'])['summary']
    assert s['positive']==2 and s['negative']==1 and s['adjacent_same']==0 and s['adjacent_switch']==1
    protocol={'version':1,'cohort':'B','draws':1000,'unique_flows':967,'duplicate_draws':33,
        'overlap_unique_with_A':67,'mask_rate':0.30,'hidden_positions':10060,
        'cohort_sha256':old['hashes']['B'],'legacy_protocol_sha256':g.sha((ROOT/'protocol.json').read_bytes()),
        'selection':'Wait for completed A including transport recovery and all six-classifier scores; use its predeclared ranking',
        'variants':VARIANTS,'prompts':old['prompts'],
        'changed':'Only add behavior instruction and visible-only summary to the same masked input',
        'features':['five most frequent visible signed lengths and counts','visible positive/negative counts',
            'same-sign/sign-change counts for originally adjacent visible packets; skip pairs touching masks'],
        'not_included':'No timestamps/IAT: absent from frozen B payload; no hidden truth, class, flow or PCAP identity in API requests',
        'request_hashes':{v:g.sha(g.encoded(bs)) for v,bs in bodies.items()},
        'generationConfig_example':bodies['plain'][0]['generationConfig'],
        'primary':'equal-weight mean accuracy of six frozen classifiers; also report mean macro-F1, magnitude/signed MAE, direction and exact hit rate',
        'invalid':'retain all draws; invalid or failed responses fill masked slots with zero; no format repair',
        'budget':'Two prompts x 1000 draws = 2000 response slots, no added candidates or prompt tuning; at most 3 HTTP attempts plus one 3-attempt transport-only recovery per slot',
        'concurrency':4,'stopping':'stop on fatal HTTP 400/401/403/404; otherwise finish both fixed variants then score once',
        'report_subsets':['all 1000 draws','933 draws whose flows were absent from A'],
        'boundary':'Exploratory historical-test-pool prompt comparison; B is with replacement, overlaps A, and is not an independent final test'}
    DEST.mkdir(exist_ok=True)
    p=DEST/'protocol.json'
    if p.exists():assert read(p)==json.loads(g.encoded(protocol)), 'Frozen B protocol changed'
    else:
        g.save(p,protocol)
        g.save(DEST/'created.json',{'utc':utc(),'code_sha256':g.sha(Path(__file__).read_bytes())})
        for v in VARIANTS:g.save(DEST/f'{v}_requests.json',bodies[v])
        g.save(DEST/'change_log.json',{'created_utc':utc(),'status':'registered_before_execution',
            'before':'Original plain prompt, masked signed lengths and mask positions',
            'after':'Same input plus visible signed-length frequency, sign counts and adjacent direction transition counts',
            'held_fixed':['teacher selected on A','B draws','mask positions','output schema','LOW thinking','4096 output token cap','six classifier weights'],
            'results':'pending; no result-based prompt edits in this batch'})
    return rows,bodies

def endpoint(model):return f'https://aiplatform.googleapis.com/v1/projects/{g.PROJECT}/locations/global/publishers/google/models/{model}:generateContent'

def one(model,variant,row,body,recover=False):
    path=DEST/variant/f"{row['draw']:04d}.json"
    key=g.sha(g.encoded({'endpoint':endpoint(model),'body':body}))
    previous=None
    if path.exists():
        r=read(path);assert r['request_sha256']==key
        if not recover or r['http_status']==200 or r.get('recovery_performed'):return r
        previous=r;g.save(DEST/'transport_first_pass'/variant/path.name,r)
    if STOP.is_set():return None
    start=time.monotonic();attempts=[];status=None;result={}
    for attempt in range(3):
        if STOP.is_set():return None
        g.throttle(model)
        g.save(DEST/'inflight'/variant/path.name,{'draw':row['draw'],'attempt':attempt+1,'utc':utc(),'recovery':recover})
        try:
            resp=g.session().post(endpoint(model),json=body,timeout=(20,90))
            status=resp.status_code
            try:result=resp.json()
            except ValueError:result={'error':{'message':'Non-JSON HTTP response'}}
            attempts.append({'http_status':status,'elapsed_seconds':time.monotonic()-start})
        except Exception as exc:
            status=None;result={'error':{'message':type(exc).__name__}}
            attempts.append({'exception_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-start})
        if status not in (None,429,500,502,503,504):break
        if status==429:
            with g.LOCK:g.NEXT_START[model]=max(g.NEXT_START[model],time.monotonic()+30)
        if attempt<2:time.sleep(15*(attempt+1))
    values,error=g.validate(result,len(row['pos'])) if status==200 else (None,'http_failure')
    rec={'model':model,'variant':variant,'draw':row['draw'],'request_sha256':key,
        'http_status':status,'values':values,'validation_error':error,'response':result,
        'attempts':(previous['attempts'] if previous else [])+attempts,
        'elapsed_seconds':time.monotonic()-start+(previous['elapsed_seconds'] if previous else 0),
        'completed_utc':utc(),'recovery_performed':previous is not None}
    g.save(path,rec);(DEST/'inflight'/variant/path.name).unlink(missing_ok=True)
    if status in (400,401,403,404):STOP.set()
    return rec

def progress(model,stage):
    out={'utc':utc(),'model':model,'stage':stage,'variants':{}}
    for v in VARIANTS:
        rs=[read(p) for p in (DEST/v).glob('[0-9][0-9][0-9][0-9].json')]
        out['variants'][v]={'processed':len(rs),'target':1000,'valid':sum(r['values'] is not None for r in rs),
            'http_success':sum(r['http_status']==200 for r in rs),
            'errors':dict(Counter(r['validation_error'] for r in rs if r['validation_error']))}
    g.save(DEST/'status.json',out);print(json.dumps(out),flush=True)

def await_winner():
    while True:
        if (g.DEST/'api_complete.json').exists() and (g.DEST/'comparison.json').exists():
            report=read(g.DEST/'comparison.json')
            assert all(report['results'][m]['draws']==1000 for m in g.MODELS)
            ranking=sorted(g.MODELS,key=lambda m:(-report['results'][m]['mean_accuracy'],
                -report['results'][m]['packet_exact_pct'],-report['results'][m]['valid_pct'],m))
            assert ranking==report['ranking']
            selected={'utc':utc(),'winner':ranking[0],'ranking':ranking,
                'A_report_sha256':g.sha((g.DEST/'comparison.json').read_bytes()),
                'A_scores':{m:report['results'][m] for m in g.MODELS}}
            p=DEST/'selection.json'
            if p.exists():assert read(p)['A_report_sha256']==selected['A_report_sha256']
            else:g.save(p,selected)
            return ranking[0]
        g.save(DEST/'status.json',{'utc':utc(),'stage':'waiting_for_A_final_scores','API_calls_in_B':0})
        time.sleep(30)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);args=ap.parse_args()
    rows,bodies=prepare()
    if args.action=='prepare':print('PASS: B identity, 1000 draws/967 flows, 10060 fixed masks, paired inputs, visible-only summaries and unchanged JSON schema');return
    lock=DEST/'running.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        model=await_winner()
        for path,digest in read(DEST/'execution_manifest.json')['files'].items():
            assert g.sha(Path(path).read_bytes())==digest, 'Frozen execution file changed: '+path
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for recovery in (False,True):
                jobs=[]
                for i,row in enumerate(rows):
                    # Alternate order by draw so neither prompt always goes first.
                    for v in (VARIANTS if i%2==0 else VARIANTS[::-1]):
                        jobs.append(pool.submit(one,model,v,row,bodies[v][i],recovery))
                pending=set(jobs)
                while pending:
                    done,pending=cf.wait(pending,timeout=30,return_when=cf.ALL_COMPLETED)
                    for f in done:f.result()
                    progress(model,'transport_recovery' if recovery else 'first_pass')
                if STOP.is_set():raise RuntimeError('Fatal HTTP error; see cached response and status')
        g.save(DEST/'api_complete.json',{'utc':utc(),'winner':model})
        progress(model,'scoring')
        subprocess.run([str(CPU_PYTHON),str(ROOT/'report_prompt_B.py')],check=True)
        progress(model,'complete')
    except Exception as exc:
        g.save(DEST/'failure.json',{'utc':utc(),'error_type':type(exc).__name__})
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
