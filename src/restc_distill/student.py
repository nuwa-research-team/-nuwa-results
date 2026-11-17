"""Assistant-only causal-LM SFT with standard PEFT LoRA/QLoRA."""
from pathlib import Path
from .common import read, read_lines, save, file_hash, digest, canonical, checked_rows, parse_answer, write_lines, schema


def encode_example(tokenizer, messages, max_length):
    if len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system/user/assistant training conversation")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("chat template does not give an exact assistant prefix; refusing ambiguous labels")
    if len(full) > max_length:
        raise ValueError(f"{len(full)} tokens exceed max_length={max_length}; no truncation is allowed")
    if len(full) <= len(prefix):
        raise ValueError("empty assistant supervision")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": [-100] * len(prefix) + full[len(prefix):]}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, rows):
        import torch
        n = max(len(r["input_ids"]) for r in rows)
        return {key: torch.tensor([r[key] + [padding] * (n - len(r[key])) for r in rows], dtype=torch.long)
                for key, padding in (("input_ids", self.pad), ("attention_mask", 0), ("labels", -100))}


def load_base(config, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    student = config["student"]
    tokenizer = AutoTokenizer.from_pretrained(student["model"], revision=student["revision"],
                                              local_files_only=student["local_files_only"], trust_remote_code=False)
    if tokenizer.pad_token_id is None or not tokenizer.chat_template:
        raise ValueError("model tokenizer must define its padding token and chat template")
    quantization = student["quantization"]
    kwargs = {"revision": student["revision"], "local_files_only": student["local_files_only"], "trust_remote_code": False}
    if quantization == "nf4":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured NF4/BF16 QLoRA requires a compatible CUDA device")
        kwargs.update(device_map={"": torch.cuda.current_device()}, dtype=torch.bfloat16,
                      quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    elif quantization == "none":
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    else:
        raise ValueError("quantization must be nf4 or none")
    model = AutoModelForCausalLM.from_pretrained(student["model"], **kwargs)
    if quantization == "none":
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    if training and quantization == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def train(config_path, data, out, max_steps=-1, validation_limit=None, eval_steps=None, resume=None):
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    config = read(config_path)
    data, out = Path(data), Path(out)
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError("training output is nonempty; supply an explicit checkpoint to resume")
    manifest = read(data / "manifest.json")
    for filename, checksum in manifest["files"].items():
        if file_hash(data / filename) != checksum:
            raise ValueError("SFT data hash mismatch")
    cfg = config["training"]
    set_seed(cfg["seed"])
    model, tokenizer = load_base(config, training=True)
    datasets, stats, group_ids = {}, {}, {}
    for split in ("train", "validation"):
        rows = read_lines(data / (split + ".jsonl"))
        if split == "validation" and validation_limit is not None:
            if not 0 < validation_limit <= len(rows):
                raise ValueError("invalid validation subset")
            rows = rows[:validation_limit]
        if not rows:
            raise ValueError("empty " + split)
        group_ids[split] = sorted({r["group_id"] for r in rows})
        datasets[split] = [encode_example(tokenizer, r["messages"], config["student"]["max_length"]) for r in rows]
        stats[split] = {"records": len(rows), "max_tokens": max(len(r["input_ids"]) for r in datasets[split]),
                        "assistant_tokens": sum(sum(x != -100 for x in r["labels"]) for r in datasets[split])}
    if set(group_ids["train"]) & set(group_ids["validation"]):
        raise ValueError("training and validation groups overlap")
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    model.config.use_cache = False
    trainable = {name: param.numel() for name, param in model.named_parameters() if param.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("expected only LoRA parameters to be trainable")
    steps = eval_steps if eval_steps is not None else cfg["eval_steps"]
    identity = {"config": config, "sft_manifest_sha256": file_hash(data / "manifest.json"), "max_steps": max_steps,
                "validation_limit": validation_limit, "eval_steps": steps}
    if resume is not None:
        previous = read(out / "run.json")
        if previous["identity"] != identity:
            raise ValueError("resume configuration or dataset differs from original run")
        if Path(resume).resolve().parent != out.resolve():
            raise ValueError("resume checkpoint must belong to this run directory")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "run.json", {"identity": identity, "data": stats, "trainable_parameters": sum(trainable.values()),
                            "total_parameters": sum(p.numel() for p in model.parameters()),
                            "student_system": manifest["student_system"], "objective": manifest["objective"]})
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=cfg["epochs"], max_steps=max_steps,
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"], max_grad_norm=cfg["max_grad_norm"],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", lr_scheduler_type="cosine", eval_strategy="steps", eval_steps=steps,
        save_strategy="steps", save_steps=steps if eval_steps is not None else cfg["save_steps"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=1, report_to=[], seed=cfg["seed"], data_seed=cfg["seed"],
        remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
                      data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=resume)
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    save(adapter / "restc.json", {"config": config, "student_system": manifest["student_system"],
                                  "sft_manifest_sha256": file_hash(data / "manifest.json"),
                                  "training_group_ids": group_ids["train"], "validation_group_ids": group_ids["validation"],
                                  "best_checkpoint": trainer.state.best_model_checkpoint,
                                  "adapter_sha256": file_hash(adapter / "adapter_model.safetensors")})
    save(out / "training_result.json", {"metrics": result.metrics, "best_metric": trainer.state.best_metric,
                                        "global_step": trainer.state.global_step, "log_history": trainer.state.log_history})
    print(canonical({"adapter": str(adapter), "steps": trainer.state.global_step, "metrics": result.metrics}))


def infer(prepared, adapter, out, split="test", limit=None):
    import torch
    from peft import PeftModel
    rows, manifest = checked_rows(prepared, split)
    if limit is not None:
        if not 0 < limit <= len(rows):
            raise ValueError("invalid inference subset")
        rows = rows[:limit]
    adapter, out = Path(adapter), Path(out)
    if out.exists():
        raise FileExistsError("prediction file already exists")
    metadata = read(adapter / "restc.json")
    if file_hash(adapter / "adapter_model.safetensors") != metadata["adapter_sha256"]:
        raise ValueError("adapter hash mismatch")
    if metadata["student_system"] != manifest["student_system"]:
        raise ValueError("student prompt differs from training prompt")
    if set(metadata["training_group_ids"]) & {r["group_id"] for r in rows}:
        raise ValueError("inference split contains groups used for student training")
    if split == "test" and set(metadata["validation_group_ids"]) & {r["group_id"] for r in rows}:
        raise ValueError("test split contains validation groups")
    model, tokenizer = load_base(metadata["config"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    model.config.use_cache = True
    student = metadata["config"]["student"]
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=model.config.vocab_size, stop_token_ids=[tokenizer.eos_token_id]))
    records = []
    for row in rows:
        inputs = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True).to(model.device)
        if inputs["input_ids"].shape[-1] + student["max_new_tokens"] > student["max_length"]:
            raise ValueError("input plus output budget exceeds configured context; no truncation")
        # Standard grammar-constrained decoding: restrict token choices before generation.
        # This does not alter generated answers or insert values after generation.
        grammar = compiler.compile_json_schema(schema(row["positions"]), any_order=True)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=student["max_new_tokens"],
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                       temperature=None, top_p=None, top_k=None, logits_processor=[LogitsProcessor(grammar)])
        suffix = generated[0, inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(suffix, skip_special_tokens=True)
        answer, error = None, None
        try:
            if len(suffix) == student["max_new_tokens"] and int(suffix[-1]) != tokenizer.eos_token_id:
                raise ValueError("generation reached token limit")
            answer = parse_answer(text, row["positions"])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        records.append({"id": row["id"], "input_sha256": digest(row["messages"]), "answer": answer,
                        "raw_text": text, "error": error, "adapter_sha256": metadata["adapter_sha256"]})
        print(f"student {len(records)}/{len(rows)} {error or 'valid'}", flush=True)
    write_lines(out, records)
    save(out.with_suffix(".manifest.json"), {"prepared_manifest_sha256": file_hash(Path(prepared) / "manifest.json"),
                                           "adapter_sha256": metadata["adapter_sha256"], "split": split,
                                           "records": len(records), "predictions_sha256": file_hash(out),
                                           "decoding": "greedy + XGrammar 0.2.9 per-input JSON Schema; one generation; invalid=abstention"})
