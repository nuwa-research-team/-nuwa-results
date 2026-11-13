# D4 optimization code

D4 行为感知流量补全与下游分类的近期优化代码。当前下游方案保留逐包与收发段联合输入。回退规则仅用于复核。

## 上传与阅读

把本目录里的六个子目录及三个根文件拖入仓库根目录，保留子目录结构。也可以每次拖入一个完整子目录，最后单独上传三个根文件。

| Folder | 内容 | 文件数 |
|---|---|---:|
| 01_Dual_Model | 双路补全模型、阶段 A/B 训练入口、约束解码和直接数值读出。 | 18 |
| 02_Flow_Preparation | D4 切流、原缺包掩码、按流时间戳及包方向对齐、开发集构建。 | 6 |
| 03_Packet_Refinement | 补全后的短包头、方向／幅值对照、可见上下文候选与冻结规则。 | 19 |
| 04_Classifier_Baselines | 六分类器的本地实现与适配、六类标签映射、可见间隔消融和逐类诊断。 | 19 |
| 05_Packet_Segment_Models | 时间摘要、逐包、收发段和逐包＋收发段联合分类；当前保留联合版。 | 14 |
| 06_Fold_Diagnostics | 固定 50% 缺包下的 A/B 折回退审计、按 PCAP 归因及 Weibo 分歧流对照。 | 5 |

阅读联合分类从 `05_Packet_Segment_Models/joint/evaluate.py` 开始。它包含原实验拟合入口。最近按文件复核使用 `06_Fold_Diagnostics/paired_audits/weibo_disagreements_B50.py`，读取缓存的 B 折预测。模型训练代码保存在第 01 组，此次整理只复制源码。

## 原项目中的位置

分类目录适合浏览和上传。脚本之间有相对路径依赖。要按原项目树放回源码，在本目录运行 `python restore_layout.py ../D4_Source_Tree`，目标应为新的空闲路径。source_manifest.json 记录每个文件的分类位置、项目位置、字节数和 SHA-256。

实验复核另需原 D4 开发面板、固定掩码、保存的预测或模型权重。Weibo-4 诊断用到 d4_packet_segment 下的 paired.npz、B50_fallback_predictions.npz，以及 dev_readout 下的 panels.json.gz 和 identity.json。原训练和云端评估脚本沿用当时的本地或服务器布局，训练环境要求与下游分析环境可能不同。这里只收录源码、小型配置和实验协议。

六分类器为本地复现/适配版本。TST 上游源码的许可放在 transformer_reference/LICENSE。
