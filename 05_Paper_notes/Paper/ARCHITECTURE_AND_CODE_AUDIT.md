# 架构与公开代码核查

## 论文主链

TCP 会话 → 有方向的 IP 包长序列（另讨论时间侧信道）→ S2E 字级 token 表示 → TFM 自监督噪声 → TFI Transformer MLM 预训练 → 补全缺失 token → 还原数值序列 → 固定下游分类器预测。

下游分类器六种：CNN、LSTM、GRU、TST、SDAE、FS-Net。对比补全方法：ZeroPad、一阶 Markov、BERT、RoBERTa、DeBERTa。不能把这五种补全器与六种分类器混为一组“六个 Nüwa 主模型”。

上游使用 DebertaForMaskedLM / DebertaConfig，6 层、12 个注意力头、max_position_embeddings=514、type_vocab_size=0；WordLevel 词表搭配 RobertaTokenizerFast。hidden_size=768 和 intermediate_size=3072 是对应库版本的默认值推断，必须读取实际 config 验证。论文报告预训练 batch=256、lr=1e-4；脚本未显式设置该 batch，不能直接声称配置一致。

序列至少 20 个包，过滤超过 1600 字节的 IP 包长所在会话，长度截断/填充到 512；signed length +1600 进入 token 化。具体方向定义、重传处理、流切分和标准化必须记录。

## 已确认的公开代码问题（未运行训练）

| 文件 | 问题 | 复现影响 |
|---|---|---|
| code/TFM.py | 缩进语法错误，缺依赖导入，未形成完整可用 collator | 不能直接运行 |
| code/pre-training.py | 导入 Hugging Face 标准 collator，未接入自定义 TFM | 执行结果不等于论文 TFM |
| code/TFM.py | 顺序 Bernoulli 与论文互斥类别比例不同；span 索引和维度处理待修正 | 噪声机制不一致 |
| code/pre-training.py | 启动删除 ./model；词表遍历 ./ 下 txt；block_size=1024 与 514 位置配置冲突 | 存在覆盖、测试泄漏或长度异常风险 |
| code/impute_missing_features.py | n_pad 被 list.append 返回值覆盖为 None | 补全输出可能失败 |
| code/data_process/train_test_data.py | clean /1500，masked /1600，其他入口又 /1600 | 干净、缺失和补全输入标准化不一致 |
| code/data_process/session2features.py | deprecated DataFrame.append，空流长度变量处理不完整 | 预处理不可直接照搬 |
| code/test.py | 占位文件；六种完整分类器、训练协议和 SOTA 入口未公开齐全 | 不具备全链一键复现条件 |

所有修复应在独立目录进行，保留上游原件、补丁和修复理由。区分“原代码执行”和“按论文重实现”；软件检查通过不等于得到论文指标。
