# Code provenance

This directory contains the upstream Nuwa Python files, preserved byte-for-byte. The upstream `code/` directory has been flattened into this `src/` directory; preprocessing scripts remain in `data_process/`.
Source: https://github.com/Timeless-zfqi/Nuwa
Pinned commit: 2d4d5bf3bfdf1f7c3b9a8bd310ace5baa49bcf65

Dataset names and download links are listed in [data/README.md](../data/README.md).
Known upstream issues: `TFM.py` has invalid indentation; `pre-training.py` uses the standard collator; `impute_missing_features.py` assigns the result of `list.append()` to `n_pad`.
This source tree is not the complete code that generated our engineering results. Actual evaluation / corrected classifier snapshots are preserved in [results/evidence/execution_code.zip](../results/evidence/execution_code.zip).
