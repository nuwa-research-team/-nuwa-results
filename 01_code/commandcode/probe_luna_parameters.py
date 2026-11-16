"""Capability probes only, with synthetic output, no parameter choice by task scores."""
import concurrent.futures as cf,json,time,urllib.request,urllib.error
from pathlib import Path
import compare as a
OUT=a.DATA/'luna_parameter_vs_prompt_20261116';OUT.mkdir(parents=True,exist_ok=True)
OPTIONS={'medium':{'reasoning_effort':'medium'},'high':{'reasoning_effort':'high'},'temp0':{'reasoning_effort':'low','temperature':0}}
secret=a.key()
def run(name,opts):
    path=OUT/'capability_standard_client'/f'{name}.json'
    if path.exists():return a.read(path)
    body={'model':'gpt-6-luna','messages':[{'role':'system','content':'Return JSON only.'},{'role':'user','content':'Return exactly {"values":[40]}.'}],
          'max_completion_tokens':4096,'stream':False,'response_format':{'type':'json_object'},**opts}
    started=time.time();status=0;response={};error=None
    req=urllib.request.Request(a.BASE+'/chat/completions',data=a.enc(body),headers={'Content-Type':'application/json','User-Agent':'NIDS-Research-CommandCode/1.0','Authorization':'Bearer '+secret})
    try:
        with urllib.request.urlopen(req,timeout=180) as r:status=r.status;response=json.load(r)
    except urllib.error.HTTPError as e:status=e.code;error=e.read().decode(errors='replace').replace(secret,'[REDACTED]')[:1800]
    except Exception as e:error=(type(e).__name__+': '+str(e)).replace(secret,'[REDACTED]')[:1000]
    rec={'utc':a.utc(),'name':name,'request':body,'http_status':status,'response':response,'error':error,'seconds':time.time()-started,
         'boundary':'HTTP acceptance only; does not independently attest upstream parameter effect.'}
    a.save(path,rec);return rec
with cf.ThreadPoolExecutor(max_workers=3) as pool:
    for r in pool.map(lambda item:run(*item),OPTIONS.items()):
        print(json.dumps({k:r[k] for k in ('name','http_status','seconds','error')},ensure_ascii=False),flush=True)
