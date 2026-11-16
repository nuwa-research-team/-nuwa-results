"""D4 paired teacher selection and a separate bootstrap prompt comparison."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NUWA = ROOT / "nuwa_reproduction_20261109"
SNAP = NUWA / "audit/paired_snapshot_20261110"
PE = SNAP / "work/paired_eval_v2_20261110"
SIX = OUT.parent / "d4_5090_visible_20261113"
KINDS = ("cnn", "lstm", "gru", "tst", "sdae", "fsnet")
BASE = ("Impute missing packet lengths in a single bidirectional network flow. "
        "The input array is chronological. Positive and negative denote opposite directions, "
        "magnitude is IP length in bytes (clipped at 1600). Each null is one missing packet. "
        "Return JSON only: an object with key values containing one nonzero signed integer "
        "in [-1600,1600] per null, in left-to-right order. Preserve the number of missing slots.")
BEHAVIOR = (" Use the signs and magnitudes on both sides of each gap, repeated signed "
            "lengths, and alternating or same-direction runs to resolve ambiguous slots. "
            "Summary fields list the most frequent signed lengths and counts, the number of "
            "visible packets of each sign, and observed adjacent same-sign and sign-changing pairs. "
            "Adjacency counts skip gaps.")


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(path)


def read(name):
    return json.loads((OUT / name).read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    raw = (SNAP / "work/d4_aligned/raw.pkl").read_bytes()
    assert digest(raw) == "312617767197cad4c931dbc57f8fdaead8185c13e6e9e10b07458ce3ffabb3d6"
    d = pickle.loads(raw)
    test = list(map(int, d["split"]["test"]))
    assert len(set(test)) == 13296 and not set(test).intersection(d["split"]["train"],) and not set(test).intersection(d["split"]["val"])
    mapping = json.loads((OUT.parent/"d4_full_aligned_20261113/six_label_audit.json").read_bytes())["mapping"]
    with np.load(PE/"mask_30.npz") as z:
        assert list(z["ids"]) == [d["ids"][i] for i in test]
        masks = z["mask"]
    cohorts = {}
    for phase, seed in (("A", 20261114), ("B", 20261115)):
        draws = np.random.default_rng(seed).choice(len(test), 1000, replace=phase=="B")
        rows = []
        for j, row in enumerate(draws):
            idx = test[row]
            truth = list(map(int, d["seqs"][idx]))
            n = len(truth)
            count = max(1, round(.30*n))
            rng = np.random.default_rng((42+idx) if phase=="A" else [20261116,j,idx])
            pos = sorted(map(int,rng.choice(n,count,replace=False)))
            if phase=="A":
                assert pos == np.flatnonzero(masks[row,:n]).tolist()
            flow = str(d["ids"][idx])
            rows.append(dict(draw=j,flow=flow,truth=truth,pos=pos,label=mapping[flow.split(".pcap#")[0]]))
        cohorts[phase] = rows
        write(OUT/f"local_{phase}.json", rows)
    a,b=({r["flow"] for r in cohorts[k]} for k in "AB")
    protocol=dict(source_sha256=digest(raw),pool=len(test),rate=.30,selection_seed=20261114,
                  bootstrap_seed=20261115,bootstrap_mask_seed=20261116,
                  primary="mean of the six frozen downstream classifier accuracies (equal classifier weights)",
                  ties="pooled exact signed-length hit rate; then fewer invalid responses; then provider name",
                  invalid="count all hidden slots as misses and fill zero for downstream evaluation",
                  setting="Exploratory teacher/prompt selection from the historical test split",
                  b_unique=len(b),b_duplicate_draws=1000-len(b),b_shared_unique=len(a&b),
                  b_draws_seen_in_a=sum(r["flow"] in a for r in cohorts["B"]),
                  hidden={k:sum(len(r["pos"]) for r in v) for k,v in cohorts.items()},
                  prompts={"plain":BASE,"behavior":BASE+BEHAVIOR},
                  hashes={k:digest((OUT/f"local_{k}.json").read_bytes()) for k in "AB"})
    write(OUT/"protocol.json",protocol)
    print(json.dumps({k:protocol[k] for k in ("pool","hidden","b_unique","b_shared_unique","b_draws_seen_in_a")}))


def messages(row, behavior):
    seq=row["truth"].copy()
    for p in row["pos"]:
        seq[p]=None
    payload={"signed":seq}
    if behavior:
        visible=[v for v in seq if v is not None]
        pairs=[(a,b) for a,b in zip(seq,seq[1:]) if a is not None and b is not None]
        same=sum((a>0)==(b>0) for a,b in pairs)
        payload["summary"]={"frequent":Counter(visible).most_common(5),
            "positive":sum(v>0 for v in visible),"negative":sum(v<0 for v in visible),
            "adjacent_same":same,"adjacent_switch":len(pairs)-same}
    return [{"role":"system","content":BASE+(BEHAVIOR if behavior else "")},
            {"role":"user","content":json.dumps(payload,separators=(",",":"))}]


def classify(seqs):
    import torch
    from torch import nn
    sys.path.insert(0,str(OUT.parent/"d4_downstream_random"))
    import downstream as ds
    torch.set_num_threads(4)
    buckets=defaultdict(list)
    for i,s in enumerate(seqs):
        buckets[max(32,1<<(len(s)-1).bit_length())].append(i)
    result={}
    for k in KINDS:
        net=ds.model_for(k,8)
        plain=k in KINDS[:3]
        if plain: net.fc=nn.Identity()
        elif k=="sdae": net.head=nn.Identity()
        elif k=="fsnet": net.head[-1]=nn.Identity()
        else: net.base.output_layer=nn.Identity()
        head=torch.load(SIX/f"six_{k}.pt",weights_only=True,map_location="cpu")
        pred=np.empty(len(seqs),dtype=int)
        with torch.inference_mode():
            for width,inds in buckets.items():
                for start in range(0,len(inds),24):
                    ix=inds[start:start+24]
                    lengths=torch.tensor([len(seqs[i]) for i in ix])
                    x=torch.zeros((len(ix),width))
                    for j,i in enumerate(ix):
                        x[j,:len(seqs[i])]=torch.tensor(seqs[i])
                    z=net(x[:,None,:]) if plain else net(x/1600,lengths)
                    logits=nn.functional.linear((z-head["mean"])/head["scale"],**head["state"])
                    pred[ix]=logits.argmax(1).numpy()
        result[k]=pred.tolist()
    return result


def score(phase, method, values):
    from sklearn.metrics import f1_score
    rows=read(f"local_{phase}.json")
    assert len(rows)==len(values)==1000
    complete=[]; exact=[]; abs_error=[]; valid=[]
    for r,v in zip(rows,values,strict=True):
        s=r["truth"].copy()
        valid.append(v is not None)
        if v is None: v=[0]*len(r["pos"])
        hit=[]; error=[]
        for p,b in zip(r["pos"],v,strict=True):
            hit.append(s[p]==b); error.append(abs(abs(s[p])-abs(b))); s[p]=b
        complete.append(s); exact.append(sum(hit)); abs_error.append(sum(error))
    preds=classify(complete)
    if method=="local_tfi" and phase=="A":
        ids=[json.loads(line)["id"] for line in (OUT.parent/"d4_downstream_random/flows.jsonl").open()]
        lookup={k:i for i,k in enumerate(ids)}
        ix=[i for i,r in enumerate(rows) if r["flow"] in lookup]
        ref_ix=[lookup[rows[i]["flow"]] for i in ix]
        with np.load(SIX/"six_predictions.npz") as z:
            for k in KINDS:
                assert np.array_equal(np.array(preds[k])[ix],z[k+"_tfi30"][ref_ix]),k
    summary={}
    split={"all":list(range(len(rows)))}
    if phase=="B":
        seen={r["flow"] for r in read("local_A.json")}
        split["flow_absent_from_A"]=[i for i,r in enumerate(rows) if r["flow"] not in seen]
    for tag,ix in split.items():
        y=np.array([rows[i]["label"] for i in ix])
        n=sum(len(rows[i]["pos"]) for i in ix)
        stats={}
        for k in KINDS:
            p=np.array(preds[k])[ix]
            stats[k]=dict(accuracy=100*(p==y).mean(),macro_f1=100*f1_score(y,p,average="macro",labels=range(6),zero_division=0))
        summary[tag]=dict(draws=len(ix),hidden=n,packet_exact_pct=100*sum(exact[i] for i in ix)/n,
            magnitude_mae=sum(abs_error[i] for i in ix)/n,valid_pct=100*sum(valid[i] for i in ix)/len(ix),
            mean_accuracy=float(np.mean([v["accuracy"] for v in stats.values()])),classifiers=stats)
    write(OUT/f"{phase}_{method}_predictions.json",dict(values=values,classes=preds))
    write(OUT/f"{phase}_{method}_scores.json",summary)
    print(phase,method,summary["all"]["packet_exact_pct"],summary["all"]["mean_accuracy"],flush=True)
    return summary["all"]


def baseline(phase):
    rows=read(f"local_{phase}.json")
    if phase=="A":
        with np.load(PE/"tfi_imputed_30.npz") as z:
            lookup=dict(zip(z["ids"],z["values"]))
        seqs=[lookup[r["flow"]][:len(r["truth"])].tolist() for r in rows]
    else:
        import tarfile,tempfile
        import torch
        sys.path.extend([str(NUWA/".deps"),str(NUWA/"shared")])
        from transformers import AutoModelForMaskedLM,PreTrainedTokenizerFast
        from paired_evaluation_v3 import mlm_fill
        torch.set_num_threads(4)
        seqs=[None]*len(rows)
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(OUT.parent/"d4_constrained_u1000_20261112/tfi_model.tar") as t:
                for name in ("config.json","pytorch_model.bin","tokenizer.json","special_tokens_map.json","tokenizer_config.json"):
                    (Path(tmp)/name).write_bytes(t.extractfile("./"+name).read())
            expected=json.loads((OUT.parent/"d4_full_aligned_20261113/cloud_code/identity.json").read_bytes())["tfi"]
            assert digest((Path(tmp)/"pytorch_model.bin").read_bytes())==expected
            net=AutoModelForMaskedLM.from_pretrained(tmp,local_files_only=True).eval()
            tok=PreTrainedTokenizerFast.from_pretrained(tmp,local_files_only=True)
            order=sorted(range(len(rows)),key=lambda i:len(rows[i]["truth"]))
            mask=np.zeros((len(rows),512),bool)
            for j,i in enumerate(order): mask[j,rows[i]["pos"]]=True
            filled,_=mlm_fill(net,tok,[rows[i]["truth"] for i in order],mask,"cpu",16)
            for i,s in zip(order,filled):seqs[i]=s
    for r,s in zip(rows,seqs,strict=True):
        hidden=set(r["pos"])
        assert len(s)==len(r["truth"]) and all(a==b for j,(a,b) in enumerate(zip(s,r["truth"])) if j not in hidden)
    return score(phase,"local_tfi",[[int(s[p]) for p in r["pos"]] for r,s in zip(rows,seqs)])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","baseline-A","baseline-B"])
    action=parser.parse_args().action
    if action=="prepare": prepare()
    else: baseline(action[-1])
