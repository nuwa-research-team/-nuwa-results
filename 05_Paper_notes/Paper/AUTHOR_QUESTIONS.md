# 需核实的问题（草稿，未向作者发送）

1. 四场景精确类别列表、PCAP 文件列表及会话 ID/标签映射；特别是 d1 三类/五类和 d4 六类/十类/二十类差异。
2. 六种下游分类器的完整层结构、训练轮数、优化器、学习率、早停、输入归一化及 checkpoints。
3. F1 的 averaging、类别支持数、原始预测；部分表格的平均行与提升标注是否有勘误。
4. 能否提供实际接入 TFM 的 pretraining 入口；互斥比例、span 采样、deletion 对齐和 MLM target 如何定义。
5. tokenization vocab/config/checkpoint 对应哪个数据场景、split 和训练角色，是否使用测试无标签流量。
6. 真实丢包位置如何检测和插入 <mask>；RMSE 是否只针对缺失位置、采用何种归一化。
7. 模拟 random/Poisson 丢包的参数、共同 mask 与 seeds；DPDK 重放拓扑及 10/20/30% 实际丢包率控制方式。
8. SFDA 是否对目标域微调；正文定义与实验描述如何对齐。
9. 补充材料 XXXI–XXXIV 最后一列 Deberta 与 TFI 对应关系，以及重复或相冲突的数值。

取得答复前，以上内容均保留为未知，不用高分作为反向选择实现的依据。
