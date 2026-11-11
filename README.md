# Nuwa 实验代码与材料

论文、源码、数据和实验结果按用途集中存放。

```text
.
├── paper/          论文正文、补充材料及提取文本
├── results/        实验指标、展示页面和证据归档
│   ├── first_round/  首轮结果
│   ├── revised/      修订结果与合并指标
│   ├── viewer/       可直接打开的实验结果页面
│   └── evidence/     预测、检查点和执行代码 ZIP
├── src/            Nuwa 上游源码与数据预处理脚本
├── data/           USTC 数据、作者配置和数据集链接
├── environment/    环境配置、依赖和原训练环境记录
├── docs/           目录说明、文件校验清单和来源记录
├── index.html      实验结果页面入口
└── requirements.txt
```

## 常用入口

- 看论文：[正文](paper/Nuwa_TON_2025.pdf)、[补充材料](paper/Nuwa_Supplementary_Material.pdf)、[文件说明](paper/README.md)。
- 看结果：直接打开根目录 [index.html](index.html)，或查看 [结果目录说明](results/README.md)。
- 看代码：[源码说明](src/SOURCE.md)、[预训练](src/pre-training.py)、[缺失特征补全](src/impute_missing_features.py)、[数据预处理](src/data_process/)。
- 找数据：[d1–d4 数据集与链接](data/README.md)。
- 配置环境：[environment/README.md](environment/README.md)。

结果页面内嵌数据，可以离线打开。预测和检查点已保存为完整 ZIP，直接解压即可。

现有实验是 USTC 八类数据上的工程适配，包含首轮 132 个评估单元和修订的 66 个单元；当前合并表为 132 个单元。具体口径见 [results/README.md](results/README.md)。

源码来源、原始证据和目录迁移记录见 [docs/README.md](docs/README.md)。
