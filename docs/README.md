# 仓库说明

## 按内容查找

| 内容 | 位置 |
| --- | --- |
| 论文 PDF、补充材料及提取文本 | [paper/](../paper/) |
| 训练数据、作者配置和数据集链接 | [data/](../data/) |
| Nuwa 上游 Python 源码 | [src/](../src/) |
| 首轮与修订实验指标、运行与核验记录 | [results/](../results/) |
| 离线结果页面 | [results/viewer/index.html](../results/viewer/index.html) |
| 完整预测、检查点和执行代码 ZIP | [results/evidence/](../results/evidence/) |
| 环境配置和原训练环境记录 | [environment/](../environment/) |

## 目录迁移

此前的一级目录按网页上传批次拆分。本次改为用途目录，源码中的 Python 文件、论文 PDF、数据和实验指标均保留原始字节。

| 原位置 | 现位置 |
| --- | --- |
| `01_Root_and_venv/` 中的入口和依赖文件 | 仓库根目录 |
| `01_Root_and_venv/venv/` | `environment/` |
| `02_src/src/code/` | `src/` |
| `02_src/src/` 中的说明 | `src/` |
| `03_Paper_main/Paper/`、`04_Paper_supplement/Paper/`、`05_Paper_notes/Paper/` | `paper/` |
| `06_data/data/` | `data/` |
| `07_results_reports/results/` | `results/` |
| `08_results_viewer/results/viewer/` | `results/viewer/` |
| `09_results_small_evidence/results/evidence/` | `results/evidence/`；原包清单归入 `docs/provenance/` |
| `10_results_predictions_parts/`、`11_results_predictions_parts/` | 合并为 `results/evidence/first_round_predictions_and_masks.zip` |
| `12_results_checkpoints_parts/`、`13_results_checkpoints_parts/` | 合并为 `results/evidence/classifier_checkpoints.zip` |
| `Nuwa_Experiment_Results_20261110/`、`Nuwa_Pages_Upload/` | 与 `results/viewer/` 完全相同的文件已去重 |

两个拆分归档按原清单顺序拼接，逐片及整体 SHA256 均核对通过；ZIP 成员完整性也已检查。完整 ZIP 保留后，删除分片和已完成使命的恢复脚本。今后直接使用完整归档。

## 校验与来源

[manifest.json](manifest.json) 记录本次整理后的文件路径和 SHA256。路径相对于仓库根目录，清单自身及 `.git/` 不计入。

`provenance/` 保留整理前的来源记录，供追溯原始材料：

- [source_map.json](provenance/source_map.json)：原文件来源和原始哈希。
- [original_package_manifest.json](provenance/original_package_manifest.json)：拆片前的原包清单。
- [split_package_manifest.json](provenance/split_package_manifest.json)：分批上传版本清单。
- [split_archives_manifest.json](provenance/split_archives_manifest.json)：原分片顺序、长度与哈希。
- [package_readme.md](provenance/package_readme.md)、[upload_guide.md](provenance/upload_guide.md)：原包说明与上传过程记录。

这些历史记录保留当时的路径和环境描述，并非当前操作说明。其中 `Paper/` 对应现在的 `paper/`，`venv/` 对应 `environment/`，`src/code/` 对应 `src/`。历史清单记录的文件范围也不等同于实际上传到 Git 的文件范围；当前文件以本目录的 `manifest.json` 为准。
