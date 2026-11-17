"""ReSTC experiment entry point; completion model: Gemini 3.8 Flash (WSL)."""
import argparse, json, runpy, sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description="ReSTC; model: Gemini 3.8 Flash; 10%, 20%, 30% loss")
    parser.add_argument("action",choices=["prepare","run"])
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    config=json.loads((repo/"environment/active_method.json").read_text())
    assert config["model"]=="gemini-3.8-flash"
    target=Path(config["experiment_directory"])/"run.py"
    assert target.is_file(),target
    sys.path.insert(0,str(target.parent))
    sys.argv=[str(target),args.action]
    runpy.run_path(str(target),run_name="__main__")

if __name__=="__main__":main()
