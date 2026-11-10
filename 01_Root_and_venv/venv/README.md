# Miniforge + dependency inventory

This folder stores configuration and lists, not a copied virtual environment. The real experiments used a Linux Python venv; converting that setup into a Miniforge definition does not mean Miniforge was the original execution environment.

1. Install Miniforge from its official source: https://github.com/conda-forge/miniforge/releases
2. From the repository root, run:

```sh
conda env create -f venv/environment.yml
conda activate nuwa-reproduction
# NVIDIA GPU / CUDA 11.8 (Linux or supported Windows host):
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# For a CPU-only machine, use the following instead of the GPU command:
# python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r venv/requirements-common.txt
python -m pip check
```

`pip-list-server.txt` is the measured inventory from the original training venv (42 installed packages), not a newly verified installation lock.
`server_environment.json` records the actual Python/Torch/platform.
`environment.yml` pins the historical Python version; these Miniforge reconstruction instructions have not been installation-tested on this computer.
PCAP extraction also requires an external tshark/Wireshark installation.
Original Nuwa source has known code/configuration issues; creating the environment alone does not fix those issues.

Official PyTorch legacy installation reference: https://pytorch.org/get-started/previous-versions/
