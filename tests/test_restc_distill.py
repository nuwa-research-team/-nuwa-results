import json
import pytest
from restc_distill.common import canonical, parse_answer, target_input, student_messages, restore, digest
from restc_distill.data import split_for, retrieve
from restc_distill.student import encode_example, CompletionCollator
from restc_distill.teacher import extract_answer
from restc_distill.evaluate import packet_metrics, classifier_metrics


@pytest.mark.parametrize("text", [
    '{"values":{"1":0}}', '{"values":{"1":true}}', '{"values":{"1":1.0}}',
    '{"values":{"1":1601}}', '{"values":{"1":-1601}}', '{"values":{"2":10}}',
    '{"values":{"1":2},"extra":1}', '{"values":{"1":2,"1":3}}',
    '```json\n{"values":{"1":2}}\n```', '{"values":[2]}'])
def test_bad_answers_are_not_repaired(text):
    with pytest.raises(ValueError):
        parse_answer(text, [1])


def test_reconstruction_and_hidden_independence():
    observed, positions = [52, None, -60, None], [1, 3]
    messages = student_messages(observed, positions, "system")
    assert messages == student_messages(observed.copy(), positions, "system")
    row = {"observed": observed, "positions": positions, "truth": [52, 1000, -60, -1000]}
    assert restore(row, {"values": {"1":1500,"3":-52}}) == [52,1500,-60,-52]
    assert observed == [52,None,-60,None]
    assert "truth" not in messages[1]["content"]


def test_groups_and_retrieval():
    groups = [digest([i, i+1]) for i in range(1000)]
    sets = {s: {g for g in groups if split_for(g, 42, [.8,.1,.1]) == s} for s in ("train","validation","test")}
    assert len(set.union(*sets.values())) == 1000
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"])
    bank = [{"group_id":"self", "truth":[50,999,-60],"indices":[0]},
            {"group_id":"a", "truth":[50,100,-60],"indices":[1]},
            {"group_id":"b", "truth":[400,900,-300],"indices":[2]}]
    examples, ids = retrieve([50,None,-60], [1], bank, "self", 1)
    assert ids == ["a"]
    assert examples[0]["answer"] == {"values":{"1":100}}
    bank[1]["truth"][1] = 1400
    assert retrieve([50,None,-60], [1], bank, "self", 1)[1] == ids
    assert retrieve([50,None,-60], [1], [], "self", 4) == ([], [])


def test_rle_slots():
    obj = target_input([1500,1500,None,None,-52], [2,3])
    assert obj["intervals_start_end_direction_bytes"] == [[0,1,"P",1500],[2,3,None,None],[4,4,"N",52]]
    with pytest.raises(ValueError):
        target_input([1,None], [0])


def test_teacher_requires_complete_single_candidate():
    response = {"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":'{"values":{"1":-52}}'}]}}]}
    assert extract_answer(response,[1])["values"]["1"] == -52
    response["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(ValueError): extract_answer(response,[1])


class ToyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1,2,3] if add_generation_prompt else [1,2,3,4,5]


def test_only_answer_and_eos_supervised():
    messages = [{"role":r,"content":"x"} for r in ("system","user","assistant")]
    row = encode_example(ToyTokenizer(), messages, 5)
    assert row["labels"] == [-100,-100,-100,4,5]
    with pytest.raises(ValueError): encode_example(ToyTokenizer(), messages, 4)
    short = {k: v[:-1] for k,v in row.items()}
    batch = CompletionCollator(0)([row,short])
    assert batch["labels"][1,-1].item() == -100
    assert batch["attention_mask"][1,-1].item() == 0


def test_metrics_include_abstentions():
    row = {"observed":[52,None],"positions":[1],"truth":[52,-100]}
    metrics = packet_metrics([row,row], [{"values":{"1":-80}},None])
    assert metrics["valid_flow_rate"] == .5
    assert metrics["length_mae_valid_slots"] == 20
    assert metrics["direction_accuracy_all_slots"] == .5
    assert classifier_metrics([0,1],[0,-1],2) == {"accuracy":50.0,"macro_f1":50.0}
    assert packet_metrics([row],[None])["length_mae_valid_slots"] is None


def test_lora_gradient_and_adapter_roundtrip(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    base = LlamaForCausalLM(config)
    original = {k:v.detach().clone() for k,v in base.state_dict().items()}
    model = get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",r=2,lora_alpha=4,target_modules="all-linear",lora_dropout=0))
    ids = torch.tensor([[1,2,3,4,5]])
    labels = torch.tensor([[-100,-100,-100,4,5]])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    model(input_ids=ids,labels=labels).loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if "lora_" not in n)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if "lora_" in n)
    optimizer.step();model.eval()
    expected = model(input_ids=ids).logits.detach()
    model.save_pretrained(tmp_path)
    restored_base=LlamaForCausalLM(config);restored_base.load_state_dict(original)
    restored=PeftModel.from_pretrained(restored_base,tmp_path).eval()
    torch.testing.assert_close(restored(input_ids=ids).logits,expected)


def test_prepared_pipeline_provenance_and_rejection(tmp_path):
    import pickle
    from restc_distill.common import ROOT, read, save, file_hash, checked_rows, read_lines, write_lines
    from restc_distill.data import prepare, build_sft
    from restc_distill.evaluate import evaluate
    seqs = [[i+1,100,-52,200] for i in range(100)]
    seqs.append(seqs[0].copy())
    raw = tmp_path/"raw.pkl"
    raw.write_bytes(pickle.dumps({"seqs":seqs,"labels":["A","B"],"y":[0]*100+[1]}))
    config=read(ROOT/"environment/distillation.json")
    config.update(raw_data=str(raw),raw_sha256=file_hash(raw),flow_limits={"train":2,"validation":1,"test":1})
    save(tmp_path/"config.json",config)
    prepared=tmp_path/"prepared"
    prepare(tmp_path/"config.json",prepared)
    groups=read(prepared/"groups.json")
    group_sets={s:{g["group_id"] for g in v} for s,v in groups.items()}
    assert not group_sets["train"] & group_sets["test"]
    assert sum(len(v) for v in group_sets.values()) == 100
    assert read(prepared/"manifest.json")["conflicting_label_groups"] == 1
    rows,manifest=checked_rows(prepared,"train")
    assert len(rows) == 6
    for row in rows:
        assert set(row["retrieval_group_ids"]) <= group_sets["train"]
        assert row["group_id"] not in row["retrieval_group_ids"]
        answer={"values":{str(i):row["truth"][i] for i in row["positions"]}}
        response={"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":canonical(answer)}]}}]}
        save(tmp_path/"teacher"/(row["id"]+".json"),{"request_sha256":row["request_sha256"],"teacher":config["teacher"],"error":None,"response":response})
    build_sft(prepared,tmp_path/"teacher",tmp_path/"sft")
    assert len(read_lines(tmp_path/"sft/train.jsonl")) == 6
    assert all(r["target_source"] == "ground_truth" for r in read_lines(tmp_path/"sft/validation.jsonl"))
    path=tmp_path/"teacher"/(rows[0]["id"]+".json")
    bad=read(path);bad["request_sha256"]="wrong";save(path,bad)
    with pytest.raises(ValueError,match="provenance"):
        build_sft(prepared,tmp_path/"teacher",tmp_path/"bad-sft")
    test,_=checked_rows(prepared,"test")
    predictions=[{"id":r["id"],"input_sha256":digest(r["messages"]),"answer":{"values":{str(i):r["truth"][i] for i in r["positions"]}}} for r in test]
    write_lines(tmp_path/"predictions.jsonl",predictions)
    evaluate(prepared,[f'Fixture={tmp_path/"predictions.jsonl"}'],tmp_path/"metrics.json")
    assert all(v["Fixture"]["exact_accuracy_all_slots"] == 1 for v in read(tmp_path/"metrics.json")["packet_metrics"].values())
    write_lines(tmp_path/"partial.jsonl",predictions[:-1])
    with pytest.raises(ValueError,match="cover exactly"):
        evaluate(prepared,[f'Fixture={tmp_path/"partial.jsonl"}'],tmp_path/"partial-metrics.json")
    with pytest.raises(ValueError,match="overlapping"):
        evaluate(prepared,[f'Fixture={tmp_path/"predictions.jsonl"}'],tmp_path/"legacy-metrics.json",classifiers="unused.json")
    (prepared/"test.jsonl").write_text((prepared/"test.jsonl").read_text()+" ")
    with pytest.raises(ValueError,match="hash mismatch"):
        checked_rows(prepared,"test")


@pytest.mark.parametrize("text,accepted", [
    ('{"values":{"1":-1600,"3":1600}}',True),
    ('{"values":{"3":-1,"1":1}}',True),
    ('{"values":{"1":0,"3":4}}',False),
    ('{"values":{"1":1601,"3":4}}',False),
    ('{"values":{"1":-1601,"3":4}}',False),
    ('{"values":{"1":2.5,"3":4}}',False),
    ('{"values":{"1":true,"3":4}}',False),
    ('{"1":10,"3":4}',False),
    ('{"values":{"1":10,"4":4}}',False)])
def test_json_grammar_semantics(text,accepted):
    import xgrammar as xgr
    from restc_distill.common import schema
    info=xgr.TokenizerInfo([chr(i) for i in range(128)]+["<eos>"],stop_token_ids=[128])
    compiled=xgr.GrammarCompiler(info).compile_json_schema(schema([1,3]),any_order=True)
    matcher=xgr.GrammarMatcher(compiled)
    assert matcher.accept_string(text) == accepted
    if accepted:
        assert matcher.accept_token(128)
        assert matcher.is_terminated()
