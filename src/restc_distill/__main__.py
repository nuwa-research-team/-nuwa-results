"""Run with PYTHONPATH=src python -m restc_distill --help."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ReSTC teacher distillation and LoRA")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", default="environment/distillation.json")
    p.add_argument("--out", required=True)
    p = sub.add_parser("teacher")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--predictions")
    p = sub.add_parser("build-sft")
    p.add_argument("--data", required=True)
    p.add_argument("--responses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--target", choices=["teacher", "truth"], default="teacher")
    p = sub.add_parser("train")
    p.add_argument("--config", default="environment/distillation.json")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--validation-limit", type=int)
    p.add_argument("--eval-steps", type=int)
    p.add_argument("--resume")
    p = sub.add_parser("infer")
    p.add_argument("--data", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", action="append", required=True, help="MethodName=predictions.jsonl; repeat for TFI/teacher/student")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--classifiers")
    p.add_argument("--allow-development-classifiers", action="store_true")
    a = parser.parse_args()
    if a.command == "prepare":
        from .data import prepare
        prepare(a.config, a.out)
    elif a.command == "teacher":
        from .teacher import generate
        generate(a.data, a.out, a.limit, a.split, a.predictions)
    elif a.command == "build-sft":
        from .data import build_sft
        build_sft(a.data, a.responses, a.out, a.limit, a.target)
    elif a.command == "train":
        from .student import train
        train(a.config, a.data, a.out, a.max_steps, a.validation_limit, a.eval_steps, a.resume)
    elif a.command == "infer":
        from .student import infer
        infer(a.data, a.adapter, a.out, a.split, a.limit)
    else:
        from .evaluate import evaluate
        evaluate(a.data, a.predictions, a.out, a.split, a.classifiers, a.allow_development_classifiers)


if __name__ == "__main__":
    main()
