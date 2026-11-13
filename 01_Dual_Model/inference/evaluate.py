"""Run a registered dual-encoder checkpoint, validating format before downstream inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from constrained import generate
from eval_frontend import TrafficFrontend, collate_features, featurize
from eval_packing import pack_single


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def features(signed, times, hidden):
    observed = [i for i in range(len(signed)) if i not in hidden]
    clocks = [times[i] for i in observed]
    timed = bool(clocks) and all(t is not None for t in clocks)
    timed = timed and all(a <= b for a, b in zip(clocks, clocks[1:]))
    records = [dict(slot=i, signed_len=int(signed[i]), time_us=int(times[i]-clocks[0]) if timed else 0)
               for i in observed]
    feat = featurize(records, len(signed))
    if not timed:
        for valid in feat["stat_valid"]:
            valid[6:9] = [0., 0., 0.]
    return feat


@torch.inference_mode()
def complete(model, tok, front, signed, times, hidden, canonical):
    # Two balanced, disjoint windows if a flow exceeds the native 256 slots.
    n = len(signed)
    cuts = [0, n] if n <= 256 else [0, n//2, n]
    completed = list(signed)
    for a, b in zip(cuts, cuts[1:]):
        query = [i-a for i in hidden if a <= i < b]
        if not query:
            continue
        feat = features(signed[a:b], times[a:b], set(query))
        z = front(**collate_features([feat], device=model.device))
        pack = pack_single(tok, model.get_input_embeddings(), z["packet_embeds"][0],
                           z["behavior_embeds"][0], b-a, query, canonical=canonical)
        _, values = generate(model, tok, pack, len(query))
        for slot, value in zip(query, values, strict=True):
            completed[a+slot] = value
    return completed


def run(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ck["stage"] in ("a", "b")
    for key, path in [("data_windows_sha256", args.windows),
                      ("model_config_sha256", args.model/"config.json"),
                      ("tokenizer_sha256", args.model/"tokenizer.json")]:
        assert digest(path) == ck["meta"][key], key
    rows = read(args.windows)
    dev_captures = {Path(w["capture_id"]).name for w in rows if w["split"] in ("train","val")}
    val = {w["sample_id"]:w for w in rows if w["split"] == "val"}
    manifest = json.loads(args.validation_manifest.read_text())["rows"]
    assert len(manifest) == len({m["sample_id"] for m in manifest}) == 256
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype)
    assert model.config.hidden_size == 2560 and model.config.num_hidden_layers == 36
    model = get_peft_model(model, LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias="none",
                           target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    assert set(ck["lora"]) == {k for k in model.state_dict() if "lora_" in k}
    model.load_state_dict(ck["lora"],strict=False)
    front = TrafficFrontend(model.config.hidden_size)
    front.load_state_dict(ck["frontend"])
    model.to(args.device).eval()
    front.to(args.device).eval()
    identity = {"checkpoint": digest(args.checkpoint), "stage":ck["stage"], "update":ck["update"],
                "windows":digest(args.windows), "validation_manifest":digest(args.validation_manifest),
                "flows":digest(args.cohort/"flows.jsonl"), "torch":str(torch.__version__), "device":args.device}
    for filename in ("constrained.py","eval_frontend.py","eval_packing.py","evaluate.py"):
        identity[filename] = digest(Path(__file__).with_name(filename))
    for rate in (10,30,50):
        identity[f"mask{rate}"] = digest(args.cohort/f"mask{rate}.npz")
    del ck
    args.out.mkdir(exist_ok=True, parents=True)
    config = args.out/"identity.json"
    if config.exists():
        assert json.loads(config.read_text()) == identity, "Result identity mismatch"
    else:
        config.write_text(json.dumps(identity,indent=2))
    # Use the original validation prompts and fixed masks before processing the test cohort.
    errors = []
    validation = args.out/"validation.json"
    if not validation.exists():
        for m in manifest:
            w = val[m["sample_id"]]
            pred = complete(model,tok,front,w["signed_lengths"],w["times_us"],m["hidden"],True)
            errors.append(np.mean([abs(pred[i]-w["signed_lengths"][i]) for i in m["hidden"]]))
        tmp = validation.with_suffix(".tmp")
        tmp.write_text(json.dumps({"valid":len(errors),"n":256,"signed_mae_window_macro":np.mean(errors)}))
        os.replace(tmp, validation)
    assert json.loads(validation.read_text())["valid"] == 256
    flows = read(args.cohort/"flows.jsonl")
    for rate in (10,30,50):
        with np.load(args.cohort/f"mask{rate}.npz") as z:
            assert list(z["ids"].astype(str)) == [f["id"] for f in flows]
            masks = z["mask"]
        path = args.out/f"dual_{rate}.jsonl"
        done = read(path) if path.exists() else []
        assert [d["id"] for d in done] == [f["id"] for f in flows[:len(done)]]
        with path.open("a", encoding="utf-8") as output:
            for f, mask in zip(flows[len(done):],masks[len(done):],strict=True):
                q = np.flatnonzero(mask).tolist()
                values = complete(model,tok,front,f["signed"],f["us"],q,False)
                rec = {"id":f["id"], "signed":values,
                       "shared_development_capture":f["id"].split("#")[0] in dev_captures}
                print(json.dumps(rec,separators=(",",":")),file=output,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("model","checkpoint","windows","validation-manifest","cohort","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    run(p.parse_args())
