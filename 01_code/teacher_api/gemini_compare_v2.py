"""Paired, resumable Gemini teacher comparison. ADC only; no credentials logged."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'gemini_compare_v2'
PROJECT = 'project-ffdab9cb-2c47-436b-be1'
MODELS = ('gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-3.1-pro-preview')
LOCAL = threading.local()
STOP = set()
LOCK = threading.Lock()
NEXT_START = {m: 0.0 for m in MODELS}
INTERVAL = {'gemini-3.8-flash':2.0, 'gemini-3.5-flash':2.0, 'gemini-3.1-pro-preview':4.0}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def encoded(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_bytes(encoded(obj))
    temp.replace(path)


def body_for(row, prompt):
    seq = row['truth'].copy()
    for p in row['pos']:
        seq[p] = None
    n = len(row['pos'])
    payload = {'signed': seq, 'missing_positions_zero_based': row['pos'], 'missing_count': n}
    return {'systemInstruction': {'parts': [{'text': prompt}]},
            'contents': [{'role': 'user', 'parts': [{'text': json.dumps(payload, separators=(',', ':'))}]}],
            'generationConfig': {'maxOutputTokens': 4096, 'responseMimeType': 'application/json',
                'thinkingConfig': {'thinkingLevel': 'LOW'},
                'responseSchema': {'type': 'OBJECT', 'required': ['values'],
                    'properties': {'values': {'type': 'ARRAY', 'minItems': n, 'maxItems': n,
                        'items': {'type': 'INTEGER', 'minimum': -1600, 'maximum': 1600}}}}}}


def validate(response, n):
    candidates = response.get('candidates', [])
    if not candidates:
        return None, 'no_candidate'
    if candidates[0].get('finishReason') != 'STOP':
        return None, 'finish_' + str(candidates[0].get('finishReason'))
    parts = candidates[0].get('content', {}).get('parts', [])
    txt = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
    try:
        obj = json.loads(txt)
    except (ValueError, TypeError):
        return None, 'invalid_json'
    if not isinstance(obj, dict) or set(obj) != {'values'}:
        return None, 'wrong_keys'
    v = obj['values']
    if not isinstance(v, list) or len(v) != n:
        return None, 'wrong_count'
    if not all(type(x) is int and 0 < abs(x) <= 1600 for x in v):
        return None, 'invalid_integer'
    return v, None


def prepare():
    old = json.loads((ROOT / 'protocol.json').read_bytes())
    raw = (ROOT / 'local_A.json').read_bytes()
    assert sha(raw) == old['hashes']['A']
    rows = json.loads(raw)
    assert len(rows) == len({r['flow'] for r in rows}) == 1000
    assert [r['draw'] for r in rows] == list(range(1000))
    assert sum(len(r['pos']) for r in rows) == 10488
    for r in rows:
        assert r['pos'] == sorted(set(r['pos']))
        assert all(0 <= p < len(r['truth']) for p in r['pos'])
    bodies = [body_for(r, old['prompts']['plain']) for r in rows]
    for r, b in zip(rows, bodies):
        payload = json.loads(b['contents'][0]['parts'][0]['text'])
        assert set(payload) == {'signed', 'missing_positions_zero_based', 'missing_count'}
        assert [i for i,v in enumerate(payload['signed']) if v is None] == r['pos']
        # Perturb hidden truth and labels: API input must remain identical.
        changed = dict(r, truth=r['truth'].copy(), label=-999, flow='not_sent')
        for p in r['pos']:
            changed['truth'][p] = 123456
        assert body_for(changed, old['prompts']['plain']) == b
    protocol = {'version': 2, 'cohort': 'A', 'count': 1000, 'mask_rate': old['rate'],
        'source_sha256': sha(raw), 'legacy_protocol_sha256': sha((ROOT/'protocol.json').read_bytes()),
        'models': MODELS, 'project': PROJECT, 'location': 'global',
        'common_prompt': old['prompts']['plain'], 'requests_sha256': sha(encoded(bodies)),
        'generationConfig_example': bodies[0]['generationConfig'],
        'primary': old['primary'], 'ties': old['ties'], 'setting': old['setting'],
        'invalid': old['invalid'], 'format_repair': 'none; no format retries or truth-assisted repair',
        'http_retry': 'at most 3 attempts for 429/500/502/503/504 and transport exceptions',
        'budget': 'one response per flow per model; maximum 3000 successful responses, 9000 HTTP attempts',
        'prompt_optimization': 'not performed in this batch; cohort B untouched',
        'note': 'Models use identical LOW thinking level, not necessarily identical internal compute; no temperature set.'}
    p = DEST / 'protocol.json'
    if p.exists():
        assert json.loads(p.read_bytes()) == json.loads(encoded(protocol)), 'Frozen protocol changed'
    else:
        save(p, protocol)
        save(DEST/'requests.json', bodies)
        save(DEST/'created.json', {'utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha(Path(__file__).read_bytes())})
    def response(v):
        return {'candidates': [{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({'values':v})}]}}]}
    assert validate(response([1500, -52]),2) == ([1500,-52],None)
    for v in ([1500], [0,-52], [True,-52], [1601,-52], [1.0,-52]):
        assert validate(response(v),2)[0] is None
    return rows, bodies


def session():
    if not hasattr(LOCAL, 'session'):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        cred, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'], quota_project_id=PROJECT)
        LOCAL.session = AuthorizedSession(cred)
    return LOCAL.session


def throttle(model):
    while True:
        with LOCK:
            now=time.monotonic()
            delay=NEXT_START[model]-now
            if delay <= 0:
                NEXT_START[model]=now+INTERVAL[model]
                return
        time.sleep(min(delay,5))


def run_one(model, row, body, recover=False):
    endpoint = f'https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/publishers/google/models/{model}:generateContent'
    key = sha(encoded({'endpoint':endpoint,'body':body}))
    path = DEST / model / f"{row['draw']:04d}.json"
    previous=None
    if path.exists():
        rec = json.loads(path.read_bytes())
        assert rec['request_sha256'] == key
        if not recover or rec['http_status']==200 or rec.get('recovery_performed'):
            return rec
        previous=rec
        save(DEST/'transport_first_pass'/model/path.name, rec)
    with LOCK:
        if model in STOP:
            return None
    start = time.monotonic()
    attempts = []
    result = {}; status = None
    for attempt in range(3):
        throttle(model)
        save(DEST/'inflight'/model/path.name, {'draw':row['draw'],'request_sha256':key,
            'attempt':attempt+1,'utc':datetime.now(timezone.utc).isoformat(),'recovery':recover})
        try:
            resp = session().post(endpoint, json=body, timeout=(20,90))
            status = resp.status_code
            try:
                result = resp.json()
            except ValueError:
                result = {'error': {'message': 'Non-JSON HTTP response'}}
            attempts.append({'http_status':status, 'elapsed_seconds':time.monotonic()-start})
        except Exception as exc:
            # Only class name: arbitrary exception strings could contain request details.
            result = {'error': {'message': type(exc).__name__}}
            status = None
            attempts.append({'exception_type':type(exc).__name__, 'elapsed_seconds':time.monotonic()-start})
        if status not in (None,429,500,502,503,504):
            break
        if status==429:
            with LOCK:
                NEXT_START[model]=max(NEXT_START[model],time.monotonic()+30)
        if attempt < 2:
            time.sleep(15 * (attempt+1))
    values, error = validate(result, len(row['pos'])) if status == 200 else (None,'http_failure')
    rec = {'model':model, 'draw':row['draw'], 'request_sha256':key, 'expected_count':len(row['pos']),
        'http_status':status, 'elapsed_seconds':time.monotonic()-start, 'attempts':attempts,
        'values':values, 'validation_error':error, 'response':result,
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    save(path, rec)
    if previous is not None:
        rec['recovery_performed']=True
        rec['attempts']=previous['attempts']+rec['attempts']
        rec['elapsed_seconds']+=previous['elapsed_seconds']
        save(path,rec)
    (DEST/'inflight'/model/path.name).unlink(missing_ok=True)
    if status in (400,401,403,404):
        with LOCK:
            STOP.add(model)
    return rec


def progress(target):
    out = {}
    for model in MODELS:
        recs = [json.loads(p.read_bytes()) for p in (DEST/model).glob('[0-9][0-9][0-9][0-9].json')]
        out[model] = {'completed':len(recs),'target':target,
            'http_success':sum(r['http_status']==200 for r in recs),
            'valid':sum(r['values'] is not None for r in recs),
            'validation_errors':dict(Counter(r['validation_error'] for r in recs if r['validation_error'])),
            'elapsed_request_seconds':round(sum(r['elapsed_seconds'] for r in recs),1)}
    save(DEST/'progress.json', {'utc':datetime.now(timezone.utc).isoformat(),'models':out})
    print(json.dumps(out),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','probe','run'])
    ap.add_argument('--workers',type=int,default=9)
    args=ap.parse_args()
    rows,bodies=prepare()
    if args.action=='prepare':
        print('PASS: 1000 fixed flows; 10488 masks; request identity and no hidden-truth/label leakage; strict JSON validator')
        return
    count=3 if args.action=='probe' else 1000
    if args.action=='run':
        for model in MODELS:
            probes=[json.loads((DEST/model/f'{i:04d}.json').read_bytes()) for i in range(3)]
            assert all(r['http_status']==200 and r['values'] is not None for r in probes), f'Probe not passed: {model}'
    lock=DEST/'running.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        pools={m:cf.ThreadPoolExecutor(max_workers=12 if m=='gemini-3.5-flash' else 4) for m in MODELS}
        try:
            futures=[pools[m].submit(run_one,m,r,b) for r,b in zip(rows[:count],bodies[:count]) for m in MODELS]
            pending=set(futures)
            while pending:
                done,pending=cf.wait(pending,timeout=30,return_when=cf.ALL_COMPLETED)
                for f in done: f.result()
                progress(count)
            if args.action=='run':
                # Same recovery policy for every model and every transport failure.
                # Never retry a completed HTTP 200 response based on its score or format.
                futures=[pools[m].submit(run_one,m,r,b,True) for r,b in zip(rows,bodies) for m in MODELS]
                pending=set(futures)
                while pending:
                    done,pending=cf.wait(pending,timeout=30,return_when=cf.ALL_COMPLETED)
                    for f in done:f.result()
                    progress(count)
                if not STOP:
                    save(DEST/'api_complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'transport_recovery_complete':True})
        finally:
            for pool in pools.values():pool.shutdown(wait=True)
        progress(count)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
