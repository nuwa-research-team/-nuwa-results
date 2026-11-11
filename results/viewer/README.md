# 结果页面

双击 [index.html](index.html)，使用 DNS2027 三线表页面。页面沿用提供的 Experiment Viewer 模板，“查看内容”按数据集全名区分实测、论文参考和历史记录。当前本地实测只有 d4（USTC-TFC2016 八类版本）；d1–d3 暂无本地实测。

| 文件／目录 | 内容 |
| --- | --- |
| [index.html](index.html) | 内嵌数据的交互页面 |
| [experiment_data.json](experiment_data.json) | 原始表格、数据集说明及按 d1–d4 区分的论文参考数据 |
| [source_data/](source_data/) | 页面数据对应的原始指标、协议和核验记录 |
| [paper_reference_extraction.csv](paper_reference_extraction.csv) | 论文参考指标 |
| [validation.json](validation.json) | 数据一致性及页面交互校验记录 |
| [manifest.json](manifest.json) | 页面及来源文件校验清单 |

页面源码与构建命令见 [src/viewer/](../../src/viewer/)。
