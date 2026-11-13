"""Local interface checks using public tokenizer and random weights; no test metrics."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from constrained import ArrayGrammar
from evaluate import complete, features
from eval_frontend import TrafficFrontend
from eval_packing import parse_p1

here = Path(__file__).resolve().parent
cohort = here.parent
tok = AutoTokenizer.from_pretrained(cohort.parent/"behavior_lora_d4_20261111/official_model_small",
                                    local_files_only=True)
torch.set_num_threads(2)
torch.manual_seed(17)
rng = random.Random(17)
counts = set()
mask_stats = {}
for rate in (10, 30, 50):
    with np.load(cohort/f"mask{rate}.npz") as z:
        counts.update(map(int, z["mask"].sum(axis=1)))
        mask_stats[rate] = [len(z["ids"]), int(z["mask"].sum())]
for count in sorted(counts):
    grammar = ArrayGrammar(tok, count)
    ids = []
    for step in range(7*count+2):
        allowed = grammar.allowed(0,torch.tensor(ids,dtype=torch.long))
        chosen = rng.choice(allowed)
        if chosen == tok.eos_token_id:
            break
        ids.append(chosen)
    else:
        raise AssertionError("Budget exhausted")
    assert parse_p1(tok.decode(ids),count)["status"] == "valid"
cfg = Qwen3Config(vocab_size=len(tok),hidden_size=32,num_hidden_layers=1,
                  intermediate_size=64,num_attention_heads=2,num_key_value_heads=1,
                  head_dim=16,max_position_embeddings=4096,eos_token_id=tok.eos_token_id)
m = get_peft_model(Qwen3ForCausalLM(cfg),LoraConfig(r=2,target_modules=["q_proj","k_proj","v_proj","o_proj"])).eval()
front = TrafficFrontend(32).eval()
s = [40, -54, 70, 121, -1400, 100, -64, 65]
t = [30, 40, 80, 94, 123, 140, 190, 200]
for q, times in [([0,3,7], t), ([2], [None]*len(t)), ([0,1,2,3,4,5,6,7], t)]:
    pred = complete(m,tok,front,s,times,q,False)
    assert parse_p1(json.dumps([pred[i] for i in q]),len(q))["status"] == "valid"
    assert all(pred[i]==s[i] for i in range(len(s)) if i not in q)
    changed = [0 if i in q else v for i,v in enumerate(s)]
    changed_times = [None if i in q else v for i,v in enumerate(times)]
    assert features(changed,changed_times,set(q)) == features(s,times,set(q))
result = {"kind":"interface check only; random tiny decoder, no performance measurement",
          "distinct_query_counts_checked":len(counts),"ranges":[min(counts),max(counts)],
          "cohort_and_hidden":mask_stats,"soft_prefix_peft_generation_cases":3}
(here/"format_check.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
