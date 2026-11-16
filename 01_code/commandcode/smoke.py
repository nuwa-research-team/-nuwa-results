"""One connectivity call and three fixed D4 requests. Never starts a full cohort."""
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from client import chat, request

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/d4_teacher_api_20261114'
OUT=SOURCE/'commandcode_smoke'

def main():
    OUT.mkdir(exist_ok=True)
    report={'time':datetime.now(timezone.utc).isoformat(), 'model_requested':'gpt-6-luna',
        'provider':'commandcode', 'purpose':'connectivity and format only', 'd4':[]}
    try:
        catalog=request('/models')
        (OUT/'models.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
        model=next(m for m in catalog['data'] if m['id']==report['model_requested'])
        assert '/chat/completions' in model['supported_endpoints']
        start=time.monotonic()
        response=chat(model['id'],[{'role':'user','content':'Return only this JSON object: {"ok":true}'}])
        report['connectivity']={'seconds':time.monotonic()-start,'response':response}
        print('Connectivity: OK',flush=True)
        protocol=json.loads((SOURCE/'protocol.json').read_bytes())
        raw=(SOURCE/'local_A.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==protocol['hashes']['A']
        report['source_sha256']=hashlib.sha256(raw).hexdigest()
        for row in json.loads(raw)[:3]:
            visible=row['truth'].copy()
            for p in row['pos']: visible[p]=None
            payload={'signed':visible,'missing_positions_zero_based':row['pos'],'missing_count':len(row['pos'])}
            messages=[{'role':'system','content':protocol['prompts']['plain']},
                {'role':'user','content':json.dumps(payload,separators=(',',':'))}]
            start=time.monotonic()
            result=chat(model['id'],messages)
            item={'draw':row['draw'],'seconds':time.monotonic()-start,'request':messages,'response':result,'valid':False}
            try:
                choice=result['choices'][0]
                obj=json.loads(choice['message']['content'])
                values=obj['values']
                item['valid']=(choice['finish_reason']=='stop' and set(obj)=={'values'} and
                    len(values)==len(row['pos']) and all(type(v) is int and 0<abs(v)<=1600 for v in values))
                if item['valid']:
                    truth=[row['truth'][p] for p in row['pos']]
                    item['magnitude_mae_bytes']=sum(abs(abs(a)-abs(b)) for a,b in zip(values,truth))/len(truth)
                    item['signed_exact_pct']=100*sum(a==b for a,b in zip(values,truth))/len(truth)
            except (ValueError,TypeError,KeyError): pass
            report['d4'].append(item)
            print('D4 draw',row['draw'],'valid:',item['valid'],flush=True)
        report['status']='complete'
    except Exception as exc:
        report['status']='blocked'
        report['error']=str(exc)
        print(report['error'],flush=True)
    finally:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (OUT/f'report_{stamp}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__': main()
