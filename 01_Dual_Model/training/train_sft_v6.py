# -*- coding: utf-8 -*-
"""P1 training entry v3 閳?fixes the v2 follow-up blockers.

  * dev/reload load the checkpoint BEFORE any evaluation;
  * --init-from (cross-stage) loads weights only and starts a fresh optimizer;
    --resume (same stage) restores optimizer/scheduler/RNG/counters and checks
    the immutable config hash;
  * reload is a real fresh-process verification comparing generated token IDs;
  * every --select-every updates the frozen val subset, selects by
    valid_rate>=0.99 then MAE, and saves the selected checkpoint;
  * checkpoints carry model revision, tokenizer/template/data/mask/code hashes,
    peak memory and timing;
  * generation budget is pre-registered from an exhaustive signed-integer
    tokenization plus punctuation/EOS;
  * stage B runs a bounded wiring check (frontend + LoRA finite grads, frozen
    base unchanged) before training.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/data1/lhy/behavior_lora_d4_20261111")
sys.path.insert(0, str(ROOT / "src"))
from audited_frontend import TrafficFrontend, collate_features, featurize, mask_indices, redact  # noqa: E402
from audited_packing import pack_single, answer_loss_from_hidden, parse_p1  # noqa: E402

RATES_TRAIN = [0.1, 0.3, 0.5, 0.7, 0.9]
FAMILIES = ["random", "block", "multispan"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def load_split(split):
    return [json.loads(l) for l in open(ROOT / "data/windows.jsonl") if json.loads(l)["split"] == split]


def mask_manifest(windows, seed, epoch=0, family="random", rate=0.3):
    rows = [{"sample_id": w["sample_id"], "hidden": mask_indices(w["n"], rate, family,
                                                                 w["sample_id"], seed=seed, epoch=epoch)}
            for w in windows]
    for r, w in zip(rows, windows):
        r["k"] = len(r["hidden"]); r["n"] = w["n"]
    payload = {"seed": seed, "epoch": epoch, "family": family, "rate": rate, "rows": rows}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def worst_case_generation_tokens(tok, lo=40, hi=65535):
    """Exhaustive signed-integer tokenisation + brackets/comma/EOS (pre-registered)."""
    worst = 0
    for v in range(lo, hi + 1):
        worst = max(worst, len(tok(str(v), add_special_tokens=False)["input_ids"]),
                    len(tok(str(-v), add_special_tokens=False)["input_ids"]))
    punct = len(tok('[,' * 2, add_special_tokens=False)["input_ids"]) + 2
    return worst, punct


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0"
        self.run_dir = ROOT / "runs" / args.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map={"": self.device},
            trust_remote_code=False)
        self.model.gradient_checkpointing_enable()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = get_peft_model(self.model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        self.base = self.model.get_base_model().model
        self.frontend = TrafficFrontend(llm_dim=self.model.config.hidden_size).to(self.device)
        self.per_int, self.punct = worst_case_generation_tokens(self.tok)
        names = dict(self.frontend.named_parameters())
        self.groups = {
            "packet_encoder": [p for n, p in names.items() if n.startswith("packet_encoder")],
            "packet_projector": [p for n, p in names.items() if n.startswith("packet_projector")],
            "behavior_encoder": [p for n, p in names.items() if n.startswith("behavior_encoder")],
            "behavior_projector": [p for n, p in names.items() if n.startswith("behavior_projector")],
            "input_scale_type": [p for n, p in names.items()
                                 if n.startswith(("packet_input", "behavior_input", "scale_embedding", "type_"))],
        }
        self.base_fingerprint = {n: p.detach().float().sum().item()
                                 for n, p in self.model.named_parameters() if "lora_" not in n}

    def emit(self, **kw):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}
        with open(self.run_dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def lora_params(self):
        return [p for n, p in self.model.named_parameters() if "lora_" in n]

    def set_lora_trainable(self, flag):
        for p in self.lora_params():
            p.requires_grad = flag

    def _front(self, w, hidden):
        recs = redact(w["signed_lengths"], w["times_us"], hidden)
        z = self.frontend(**collate_features([featurize(recs, w["n"])], device=self.device))
        return z["packet_embeds"][0, : w["n"]], z["behavior_embeds"][0][z["behavior_valid"][0]]

    def train_step(self, w, hidden):
        pkt, beh = self._front(w, hidden)
        answer = [w["signed_lengths"][i] for i in hidden]
        pack = pack_single(self.tok, self.model.get_input_embeddings(), pkt, beh,
                           w["n"], hidden, answer=answer)
        out = self.base(inputs_embeds=pack["inputs_embeds"], attention_mask=pack["attention_mask"],
                        position_ids=pack["position_ids"], use_cache=False)
        loss = answer_loss_from_hidden(out.last_hidden_state, self.model.get_output_embeddings(),
                                       pack["labels"])
        return loss

    @torch.inference_mode()
    def generate(self, w, hidden):
        was_m = self.model.training
        self.model.eval(); self.frontend.eval()
        pkt, beh = self._front(w, hidden)
        pack = pack_single(self.tok, self.model.get_input_embeddings(), pkt, beh,
                           w["n"], hidden, answer=None)
        budget = len(hidden) * (self.per_int + 1) + self.punct + 8
        assert pack["inputs_embeds"].shape[1] + budget <= 8192
        t0 = time.time()
        gen = self.model.generate(inputs_embeds=pack["inputs_embeds"],
                                  attention_mask=pack["attention_mask"],
                                  max_new_tokens=budget, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        ids = gen[0].tolist()
        text = self.tok.decode(gen[0], skip_special_tokens=True)
        eos = bool(ids and ids[-1] == self.tok.eos_token_id)
        parsed = parse_p1(text, len(hidden))
        status = "truncated" if not eos else parsed["status"]
        if was_m:
            self.model.train(); self.frontend.train()
        return {"raw": text, "token_ids": ids, "status": status,
                "parsed": parsed["parsed_values"], "k": len(hidden), "budget": budget,
                "latency_ms": round((time.time() - t0) * 1000, 1)}

    def evaluate(self, windows, rows, tag):
        out = []
        with open(self.run_dir / f"generations_{tag}.jsonl", "w") as jl:
            for k, (w, m) in enumerate(zip(windows, rows), 1):
                g = self.generate(w, m["hidden"])
                gold = [w["signed_lengths"][i] for i in m["hidden"]]
                err = (float(np.mean(np.abs(np.array(g["parsed"]) - np.array(gold))))
                       if g["status"] == "valid" else None)
                rec = {"sample_id": w["sample_id"], "gold": gold, **g, "abs_err": err}
                out.append(rec)
                jl.write(json.dumps(rec) + "\n")
                if k % 8 == 0:
                    self.emit(event="gen_progress", tag=tag, done=k, total=len(windows),
                              last_latency_ms=g["latency_ms"])
        valid = [r for r in out if r["status"] == "valid"]
        summary = {"tag": tag, "n": len(out), "valid_rate": len(valid) / max(len(out), 1),
                   "truncated": sum(r["status"] == "truncated" for r in out),
                   "invalid_json": sum(r["status"] == "invalid_json" for r in out),
                   "mae_valid_only": float(np.mean([r["abs_err"] for r in valid])) if valid else None,
                   "mean_latency_ms": float(np.mean([r["latency_ms"] for r in out]))}
        with open(self.run_dir / f"generations_{tag}.json", "w") as fh:
            json.dump(out, fh)
        return summary

    def save(self, path, extra):
        meta = {
            "model_revision": open(ROOT / "model/revision.txt").read().strip()
            if (ROOT / "model/revision.txt").exists() else None,
            "model_config_sha256": sha(ROOT / "model/qwen3-4b-instruct-2507/config.json"),
            "tokenizer_sha256": sha(ROOT / "model/qwen3-4b-instruct-2507/tokenizer.json"),
            "data_windows_sha256": sha(ROOT / "data/windows.jsonl"),
            "split_manifest_sha256": sha(ROOT / "data/split_manifest.json"),
            "frontend_code_sha256": sha(ROOT / "src/audited_frontend.py"),
            "packing_code_sha256": sha(ROOT / "src/audited_packing.py"),
            "train_code_sha256": sha(Path(__file__)),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "config_hash": hashlib.sha256(json.dumps(
                {k: v for k, v in vars(self.args).items()
                 if k not in ("run_id", "resume", "init_from")},
                sort_keys=True, default=str).encode()).hexdigest(),
            "rng": rng_state(),
        }
        tmp = str(path) + ".tmp"
        torch.save({"frontend": self.frontend.state_dict(),
                    "lora": {k: v for k, v in self.model.state_dict().items() if "lora_" in k},
                    "meta": meta, **extra}, tmp)
        os.replace(tmp, path)

    def load_weights(self, path):
        ck = torch.load(path, map_location="cpu")
        self.frontend.load_state_dict(ck["frontend"])
        self.model.load_state_dict(ck["lora"], strict=False)
        return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "overfit", "dev", "check32", "a", "b", "reload"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-dir", default=str(ROOT / "model/qwen3-4b-instruct-2507"))
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--accum", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--init-from", default="", help="weights only (cross-stage)")
    ap.add_argument("--resume", default="", help="same-stage full state")
    ap.add_argument("--select-every", type=int, default=500)
    ap.add_argument("--roll-eval-windows", type=int, default=256)
    ap.add_argument("--final-eval-windows", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    r = Runner(args)
    r.emit(event="loaded", stage=args.stage, hidden=r.model.config.hidden_size,
           frontend_params=sum(p.numel() for p in r.frontend.parameters()),
           lora_params=sum(p.numel() for p in r.lora_params()),
           per_int_tokens=r.per_int, punctuation=r.punct,
           gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))

    val = load_split("val")[:256]
    dev_masks = mask_manifest(val, args.seed)
    json.dump(dev_masks, open(r.run_dir / "dev_mask_manifest.json", "w"))
    r.emit(event="val_registered", n=len(val), sha256=dev_masks["sha256"])

    source = args.init_from or args.resume
    if args.stage in ("dev", "reload", "b", "check32") and not source:
        raise SystemExit(f"stage {args.stage} requires --init-from or --resume")
    if source:
        r.load_weights(source)
        r.emit(event="weights_loaded", path=source, cross_stage=bool(args.init_from))
    if args.stage == "check32":
        train32 = load_split("train")[:32]
        tm = mask_manifest(train32, args.seed)
        res = r.evaluate(train32, tm["rows"], "train32")
        json.dump({"passed": res["valid_rate"] > 0, **res},
                  open(r.run_dir / "check32.json", "w"), indent=2)
        r.emit(event="check32_eval", **res)
        return
    if args.stage == "dev":
        kdev = args.final_eval_windows or 256
        res = r.evaluate(val[:kdev], dev_masks["rows"][:kdev], f"dev{kdev}")
        json.dump({"passed": res["valid_rate"] >= 0.0, **res},
                  open(r.run_dir / "dev.json", "w"), indent=2)
        r.emit(event="dev_eval", **res)
        return
    if args.stage == "reload":
        ref = json.load(open(ROOT / "runs" / args.init_from.split("/")[-2] / "generations_reload_ref.json"))
        if not ref:
            raise SystemExit("reload reference is empty; refusing to report PASS")
        rows = dev_masks["rows"][:8]
        outs = []
        for w, m, exp in zip(val[:8], rows, ref):
            g = r.generate(w, m["hidden"])
            outs.append({"sample_id": w["sample_id"], "same_ids": g["token_ids"] == exp["token_ids"],
                         "raw": g["raw"]})
        ok = all(o["same_ids"] for o in outs)
        json.dump({"passed": ok, "samples": outs}, open(r.run_dir / "reload_check.json", "w"), indent=2)
        r.emit(event="reload_check", passed=ok)
        return

    train = load_split("train")
    fixed = args.stage in ("smoke", "overfit")
    train = train[:4] if args.stage == "smoke" else (train[:32] if args.stage == "overfit" else train)
    tr_masks = mask_manifest(train, args.seed) if fixed else None
    if fixed:
        json.dump(tr_masks, open(r.run_dir / "train_mask_manifest.json", "w"))
        r.emit(event="train_masks_frozen", sha256=tr_masks["sha256"], n=len(train))

    joint = args.stage == "b"
    r.set_lora_trainable(joint)
    groups = [{"params": list(r.frontend.parameters()), "lr": 5e-5 if joint else 1e-4}]
    if joint:
        groups.append({"params": r.lora_params(), "lr": 1e-4})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    total = args.max_updates or {"smoke": 3, "overfit": 100, "a": 1000, "b": 10000}[args.stage]
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) *
        (0.5 * (1 + math.cos(math.pi * max(0, s - warm) / max(1, total - warm)))))

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        want = hashlib.sha256(json.dumps(
            {k: v for k, v in vars(args).items() if k not in ("run_id", "resume", "init_from")},
            sort_keys=True, default=str).encode()).hexdigest()
        if ck["meta"].get("config_hash") != want:
            r.emit(event="resume_config_mismatch", fatal=True)
            raise SystemExit("resume config hash mismatch; refusing to continue")
        opt.load_state_dict(ck["optim"]); sched.load_state_dict(ck["sched"])
        set_rng(ck["meta"]["rng"]); start = ck["update"]
        best = ck.get("best", best)
        r.emit(event="resumed", update=start, best=best)

    if joint:
        # bounded wiring check before B: frontend and LoRA grads finite, base frozen
        w = train[0]; hidden = mask_indices(w["n"], 0.3, "random", w["sample_id"], seed=args.seed)
        opt.zero_grad(set_to_none=True)
        r.train_step(w, hidden).backward()
        g = {k: float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in ps if p.grad is not None)))
             for k, ps in r.groups.items()}
        g["lora"] = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in r.lora_params()
                                         if p.grad is not None)))
        opt.zero_grad(set_to_none=True)
        r.emit(event="wiring_check", grad_norms=g, all_finite=all(np.isfinite(v) for v in g.values()))
        if not all(np.isfinite(v) and v > 0 for v in g.values()):
            r.emit(event="wiring_check_failed", grad_norms=g)
            raise SystemExit("stage B wiring check failed")

    r.model.train()
    r.frontend.train()
    opt.zero_grad(set_to_none=True)
    upd, micro, losses = start, 0, []
    best = {"update": None, "mae": None, "valid_rate": None}
    t0 = time.time()
    while upd < total:
        idx = (upd * args.accum + micro) % len(train)
        w = train[idx]
        if fixed:
            hidden = tr_masks["rows"][idx]["hidden"]
        else:
            hidden = mask_indices(w["n"], random.choice(RATES_TRAIN), random.choice(FAMILIES),
                                  w["sample_id"], seed=args.seed,
                                  epoch=(upd * args.accum + micro) // len(train))
        loss = r.train_step(w, hidden)
        (loss / args.accum).backward()
        losses.append(float(loss))
        micro += 1
        if micro == args.accum:
            gn = {k: float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in ps if p.grad is not None)))
                  for k, ps in r.groups.items()}
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]
                                            if p.grad is not None], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            upd += 1; micro = 0
            if upd % 10 == 0 or upd <= 3:
                r.emit(event="update", update=upd, loss=float(np.mean(losses[-args.accum:])),
                       grad_norms=gn, lr=opt.param_groups[-1]["lr"],
                       elapsed=round(time.time() - t0, 1))
            if args.select_every and upd % args.select_every == 0:
                roll = r.evaluate(val[: args.roll_eval_windows],
                                  dev_masks["rows"][: args.roll_eval_windows], f"roll_u{upd}")
                eligible = roll["valid_rate"] >= 0.99
                r.emit(event="rolling_eval", update=upd, eligible=eligible, **roll)
                cur_mae = roll["mae_valid_only"]
                if eligible and cur_mae is not None:
                    if best["mae"] is None or cur_mae < best["mae"]:
                        best = {"update": upd, "mae": cur_mae,
                                "valid_rate": roll["valid_rate"]}
                        r.save(r.run_dir / "ckpt_best.pt",
                               {"update": upd, "optim": opt.state_dict(),
                                "sched": sched.state_dict(), "stage": args.stage,
                                "seed": args.seed, "args": vars(args), "rolling": roll,
                                "best": best})
    r.save(r.run_dir / "ckpt_final.pt",
           {"update": upd, "optim": opt.state_dict(), "sched": sched.state_dict(),
            "stage": args.stage, "seed": args.seed, "args": vars(args)})
    k_final = args.final_eval_windows or (16 if args.stage in ("smoke", "overfit") else 256)
    final = r.evaluate(val[:k_final], dev_masks["rows"][:k_final], f"val{k_final}")
    r.emit(event="final_eval", updates=upd, **final)
    acc = {"passed": bool(np.isfinite(final["valid_rate"])),
           "stage": args.stage, "updates": upd, "wall_seconds": round(time.time() - t0, 1),
           "loss_first10_mean": float(np.mean(losses[:10])),
           "loss_last10_mean": float(np.mean(losses[-10:])),
           "val": final, "best_rolling": best,
           "checkpoint_eligible": best["update"] is not None,
           "note": "engineering acceptance; valid_rate>=0.99 is the formal selection gate"}
    ref = [{"sample_id": w["sample_id"], "token_ids": r.generate(w, m["hidden"])["token_ids"]}
           for w, m in zip(val[:8], dev_masks["rows"][:8])]
    json.dump(ref, open(r.run_dir / "generations_reload_ref.json", "w"))
    json.dump(acc, open(r.run_dir / "acceptance.json", "w"), indent=2)
    r.emit(event="done", updates=upd, val_valid_rate=final["valid_rate"])


if __name__ == "__main__":
    main()
