"""DeepSeek A, then best-teacher paired B. Credentials stay in process memory."""
import concurrent.futures as cf
import getpass
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
import gemini_prompt_B as b

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'cross_teacher_v1'
MODEL='deepseek-v4-pro'
URL='https://api.deepseek.com/chat/completions'
CPU=ROOT.parents[1]/'.venv/Scripts/python.exe'
STOP=threading.Event()
PACE=threading.Lock()
NEXT=0.0
SECRET=None
TLS=threading.local()

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def state(stage,**kw):g.save(DEST/'status.json',{'utc':utc(),'stage':stage,**kw})

def request_for(model,google_body):
    if model!=MODEL:
        return b.endpoint(model), google_body
    return URL, {'model':model,'messages':[
        {'role':'system','content':google_body['systemInstruction']['parts'][0]['text']},
        {'role':'user','content':google_body['contents'][0]['parts'][0]['text']}],
        'reasoning_effort':'low','max_tokens':4096,'stream':False,
        'response_format':{'type':'json_object'}}

def validate(model,response,n):
    if model!=MODEL:return g.validate(response,n)
    cs=response.get('choices',[])
    if not cs:return None,'no_candidate'
    c=cs[0]
    if c.get('finish_reason')!='stop':return None,'finish_'+str(c.get('finish_reason'))
    try:o=json.loads(c.get('message',{}).get('content',''))
    except (ValueError,TypeError):return None,'invalid_json'
    if not isinstance(o,dict) or set(o)!={'values'}:return None,'wrong_keys'
    vs=o['values']
    if not isinstance(vs,list) or len(vs)!=n:return None,'wrong_count'
    if not all(type(v) is int and 0<abs(v)<=1600 for v in vs):return None,'invalid_integer'
    return vs,None

def prepare():
    rows,abodies=g.prepare()
    brows,bbodies=b.prepare()
    DEST.mkdir(exist_ok=True)
    p={'version':1,'added_teacher':MODEL,'A':{'draws':1000,'masks':10488,'rate':0.30},
       'B':{'draws':1000,'unique':967,'masks':10060,'rate':0.30,'overlap_A':67},
       'sequence':'Complete Gemini A and final scores; DeepSeek A; rank all four teachers; selected teacher B plain/behavior',
       'selection':'mean Accuracy of six frozen classifiers; ties signed exact hit, valid rate, name',
       'request':'identical system prompt and masked user content to Gemini; no truth, class or flow ID',
       'platform_difference':'DeepSeek JSON-object mode plus strict local validation; Gemini native JSON schema. Both low reasoning and 4096 output cap; compute budgets not identical.',
       'invalid':'retain all draws; zero-fill missing slots on HTTP/format failures; never repair or retry HTTP-200 answers',
       'budget':'1000 DeepSeek A + 2000 selected-teacher B response slots; 3 attempts plus one 3-attempt transport recovery; no extra prompt candidates',
       'features':read(b.DEST/'protocol.json')['features'],
       'boundary':'Exploratory historical-test-pool model/prompt selection, not independent final test; no new training',
       'A_request_sha256':g.sha(g.encoded([request_for(MODEL,x) for x in abodies])),
       'B_request_sha256':{m:{v:g.sha(g.encoded([request_for(m,x) for x in bs])) for v,bs in bbodies.items()} for m in (*g.MODELS,MODEL)}}
    path=DEST/'protocol.json'
    if path.exists():assert read(path)==p
    else:g.save(path,p)
    # Reuse and verify the full frozen input/classifier identity manifest.
    files=read(b.DEST/'execution_manifest.json')['files'].copy()
    for name in ('A_local_tfi_predictions.json','A_local_tfi_scores.json','cross_teacher_chain.py','report_cross_teacher.py'):
        path=ROOT/name;files[str(path)]=g.sha(path.read_bytes())
    if (DEST/'execution_manifest.json').exists():assert read(DEST/'execution_manifest.json')['files']==files
    else:g.save(DEST/'execution_manifest.json',{'files':files,'utc':utc()})
    for path,digest in files.items():assert g.sha(Path(path).read_bytes())==digest,path
    for row,body in zip(rows,abodies):
        changed=dict(row,truth=row['truth'].copy(),label=-999,flow='hidden')
        for pos in row['pos']:changed['truth'][pos]=999999
        assert request_for(MODEL,body)==request_for(MODEL,g.body_for(changed,read(ROOT/'protocol.json')['prompts']['plain']))
    def fake(v,finish='stop'):return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps({'values':v})}}]}
    assert validate(MODEL,fake([1500,-52]),2)==([1500,-52],None)
    for v in ([1500],[0,-52],[True,-52],[1601,-52],[1.0,-52]):assert validate(MODEL,fake(v),2)[0] is None
    assert validate(MODEL,fake([1500,-52],'length'),2)[0] is None
    return rows,abodies,brows,bbodies

def one(model,phase,variant,row,google_body,recovery=False):
    import requests
    global NEXT
    folder=DEST/phase/variant
    path=folder/f"{row['draw']:04d}.json"
    endpoint,body=request_for(model,google_body)
    digest=g.sha(g.encoded({'endpoint':endpoint,'body':body}))
    previous=None
    if path.exists():
        previous=read(path);assert previous['request_sha256']==digest
        if not recovery or previous['http_status']==200 or previous.get('recovery_performed'):return
    if STOP.is_set():return
    attempts=list(previous['attempts']) if previous else []
    response={};status=0;started=time.monotonic()
    for attempt in range(3):
        if STOP.is_set():return
        with PACE:
            time.sleep(max(0,NEXT-time.monotonic()));NEXT=time.monotonic()+2
        try:
            if model==MODEL:
                if not hasattr(TLS,'session'):TLS.session=requests.Session()
                r=TLS.session.post(endpoint,json=body,headers={'Authorization':'Bearer '+SECRET},timeout=(20,120))
            else:r=g.session().post(endpoint,json=body,timeout=(20,120))
            status=r.status_code
            try:response=r.json()
            except ValueError:response={}
            attempts.append({'utc':utc(),'http_status':status})
        except requests.RequestException as exc:
            status=0;response={};attempts.append({'utc':utc(),'http_status':0,'error_type':type(exc).__name__})
        if status==200:break
        if status not in (0,408,429,500,502,503,504):STOP.set();break
        if attempt<2:time.sleep(15*(attempt+1))
    values,error=validate(model,response,len(row['pos'])) if status==200 else (None,'http_failure')
    if previous:g.save(DEST/'transport_first_pass'/phase/variant/path.name,previous)
    # Do not persist headers, credentials, or raw non-success HTTP error bodies.
    g.save(path,{'draw':row['draw'],'model':model,'request_sha256':digest,'http_status':status,
        'response':response if status==200 else {},'values':values,'validation_error':error,
        'attempts':attempts,'recovery_performed':recovery,'elapsed_seconds':time.monotonic()-started,
        'completed_utc':utc()})

def progress(phase,model,variants,recovery):
    info={}
    for v in variants:
        rs=[read(p) for p in (DEST/phase/v).glob('[0-9][0-9][0-9][0-9].json')]
        recent=[r for r in rs if (datetime.now(timezone.utc)-datetime.fromisoformat(r['completed_utc'])).total_seconds()<300]
        info[v]={'completed':len(rs),'valid':sum(r['values'] is not None for r in rs),
          'http_success':sum(r['http_status']==200 for r in rs),'errors':dict(Counter(r['validation_error'] for r in rs if r['validation_error'])),
          'last_5min_count':len(recent)}
    state(phase+'_transport_recovery' if recovery else phase+'_first_pass',model=model,variants=info)

def run_phase(phase,model,rows,bodies):
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for recovery in (False,True):
            futures=[]
            for i,row in enumerate(rows):
                variants=list(bodies)
                if i%2:variants.reverse()
                for v in variants:futures.append(pool.submit(one,model,phase,v,row,bodies[v][i],recovery))
            pending=set(futures)
            while pending:
                done,pending=cf.wait(pending,timeout=30,return_when=cf.ALL_COMPLETED)
                for f in done:f.result()
                progress(phase,model,bodies,recovery)
            if STOP.is_set():raise RuntimeError('Fatal API response; inspect sanitized status files')
    g.save(DEST/phase/'api_complete.json',{'utc':utc(),'model':model})

def main():
    import requests
    global SECRET
    rows,abodies,brows,bbodies=prepare()
    if '--prepare' in sys.argv:print('PASS: frozen A/B, visible-only payloads, validation, manifests',flush=True);return
    lock=DEST/'running.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        SECRET=getpass.getpass('DeepSeek key (hidden): ')
        check=requests.get('https://api.deepseek.com/models',headers={'Authorization':'Bearer '+SECRET},timeout=(15,45))
        if check.status_code!=200:raise RuntimeError('DeepSeek authentication HTTP '+str(check.status_code))
        available=[m['id'] for m in check.json()['data']];assert MODEL in available
        g.save(DEST/'authentication_check.json',{'utc':utc(),'status':200,'models':available})
        while not (g.DEST/'api_complete.json').exists() or not (g.DEST/'comparison.json').exists():
            state('waiting_for_Gemini_A_final_scores',DeepSeek_D4_calls=0,B_calls=0)
            time.sleep(30)
        state('A_starting',model=MODEL)
        run_phase('A',MODEL,rows,{'plain':abodies})
        state('A_scoring',model=MODEL)
        subprocess.run([str(CPU),str(ROOT/'report_cross_teacher.py'),'A'],check=True)
        selected=read(DEST/'A/comparison.json')['ranking'][0]
        g.save(DEST/'selection.json',{'utc':utc(),'winner':selected,'A_report_sha256':g.sha((DEST/'A/comparison.json').read_bytes())})
        state('B_starting',model=selected)
        run_phase('B',selected,brows,bbodies)
        state('B_scoring',model=selected)
        subprocess.run([str(CPU),str(ROOT/'report_cross_teacher.py'),'B'],check=True)
        state('complete',winner=selected)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__)
        raise
    finally:
        SECRET=None;lock.unlink(missing_ok=True)

if __name__=='__main__':main()
