"""Run providers on identical blind requests; resume completed responses by content hash."""
import concurrent.futures
import csv
import json
import os
import time
import urllib.error
import urllib.request

from experiment import OUT, NUWA, KINDS, digest, messages, read, score, write

PROVIDERS={
    "gpt":("OPENAI_API_KEY","https://api.openai.com/v1","gpt-6-astra"),
    "deepseek":("DEEPSEEK_API_KEY","https://api.deepseek.com","deepseek-v4-pro"),
}


def api_key(name):
    key=os.environ.get(name)
    if not key and os.name=="nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,"Environment") as h:
                key=winreg.QueryValueEx(h,name)[0]
        except FileNotFoundError:
            pass
    return key


def predict(provider,row,behavior,key):
    _,base,model=PROVIDERS[provider]
    request={"model":model,"messages":messages(row,behavior),"reasoning_effort":"low"}
    request["max_completion_tokens" if provider=="gpt" else "max_tokens"]=3072
    body=json.dumps(request,separators=(",",":")).encode()
    cache=OUT/"responses"/provider
    cache.mkdir(parents=True,exist_ok=True)
    # Include draw identity so duplicate blind inputs remain independent experimental draws.
    path=cache/(digest(body+str(row["draw"]).encode())+".json")
    if not path.exists():
        req=urllib.request.Request(base+"/chat/completions",data=body,
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req,timeout=150) as response:
                    result=json.load(response)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429,500,502,503,504) or attempt==3:
                    raise RuntimeError(f"{provider} returned HTTP {e.code}") from None
                time.sleep(2**(attempt+1))
        choice=result["choices"][0]
        write(path,dict(request_hash=digest(body),returned_model=result.get("model"),
            content=choice["message"].get("content"),finish=choice["finish_reason"],
            usage=result.get("usage"),request_id=result.get("id")))
    record=json.loads(path.read_bytes())
    if record["finish"]!="stop": return None
    try:
        v=json.loads(record["content"])["values"]
    except (ValueError,KeyError,TypeError):
        return None
    if isinstance(v,list) and len(v)==len(row["pos"]) and all(type(x) is int and 0<abs(x)<=1600 for x in v):
        return v
    return None


def evaluate(phase,provider,behavior,key):
    rows=read(f"local_{phase}.json")
    name=provider+("_behavior" if behavior else "_plain")
    # Check the first actual request before enqueueing a provider batch.
    first=predict(provider,rows[0],behavior,key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rest=list(pool.map(lambda r:predict(provider,r,behavior,key),rows[1:]))
    return score(phase,name,[first,*rest])


def table():
    paper={}
    ref=NUWA/"viewer/Nuwa_Experiment_Results_20261110/paper_reference_extraction.csv"
    with ref.open(encoding="utf-8-sig") as source:
        for r in csv.DictReader(source):
            if (r["dataset"],r["environment"],r["table"],r["imputer"])==("d4","simulated","XVIII","TFI") and float(r["loss_rate"])==.3:
                paper[r["classifier"].lower().replace("-",""),r["metric"]]=r["paper_value_percent"]
    with (OUT/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["phase","method","subset","classifier","flows","masked_slots","packet_exact_pct",
            "magnitude_MAE_bytes","flow_accuracy_pct","flow_macro_F1_pct","valid_response_pct",
            "paper_TFI_accuracy_reference_pct","paper_TFI_F1_averaging_unspecified_pct"])
        for path in sorted(OUT.glob("*_scores.json")):
            phase,method=path.stem.removesuffix("_scores").split("_",1)
            for subset,stats in json.loads(path.read_bytes()).items():
                for k in KINDS:
                    c=stats["classifiers"][k]
                    writer.writerow([phase,method,subset,k,stats["draws"],stats["hidden"],stats["packet_exact_pct"],
                        stats["magnitude_mae"],c["accuracy"],c["macro_f1"],stats["valid_pct"],
                        paper[k,"Accuracy"],paper[k,"F1"]])


def main():
    protocol=read("protocol.json")
    for phase in "AB":
        assert digest((OUT/f"local_{phase}.json").read_bytes())==protocol["hashes"][phase]
    ranking={}
    missing=[]
    for provider,(var,_,_) in PROVIDERS.items():
        key=api_key(var)
        if not key:
            missing.append(var)
            continue
        ranking[provider]=evaluate("A",provider,False,key)
        table()
    if missing:
        table()
        print("Waiting for credentials: "+", ".join(missing))
        return
    winner=sorted(ranking,key=lambda k:(-ranking[k]["mean_accuracy"],-ranking[k]["packet_exact_pct"],
        -ranking[k]["valid_pct"],k))[0]
    write(OUT/"selection.json",{"winner":winner,"scores":ranking})
    for variant in (False,True):
        evaluate("B",winner,variant,api_key(PROVIDERS[winner][0]))
        table()


if __name__=="__main__": main()
