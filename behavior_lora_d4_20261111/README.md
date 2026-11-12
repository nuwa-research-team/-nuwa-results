# 行为感知流量补全模型

本仓库保存 D4（USTC）实验的 Qwen3-4B 版本。给定一个包序列、可见包记录和已知的缺失位置，模型按位置顺序生成带符号的 IP 包长 JSON 数组。绝对值表示字节数，正负号表示方向。

## 从哪里读

| 文件（均在 src 下） | 用途 |
|---|---|
| audited_frontend.py | 整包遮蔽、可见包特征和多尺度行为统计；包编码器、行为编码器与两个投影器 |
| audited_packing.py | 将两路连续向量嵌入 Qwen 对话前缀；计算答案部分的交叉熵；校验 JSON、项数、整数和包长范围 |
| train_sft_v6.py | 加载 Qwen、注入注意力 LoRA、组装训练样本、生成和保存模型状态 |
| formal_train_v1.py | **正式训练入口**。控制阶段、优化器、固定验证掩码、生成评估和检查点选择 |
| audited_control.py | 检查点选择与断点续训身份校验 |
| parse_pcap.py、build_windows.py | 用 tshark 提取 TCP/IPv4 包记录，再分窗和按采集组划分数据 |
| check_audited_frontend.py、check_audited_control.py | 不依赖真实流量的前端与训练控制检查；使用同目录的合成 fixture |

## 模型与训练

遮蔽一个包时同步隐藏长度、方向和时间。包分支保留局部上下文，行为分支用可见记录计算 16 包、64 包及全窗口统计。两个投影器将两路表示映射到 Qwen 的 2560 维输入空间。监督目标来自被遮蔽包的真实带符号长度。有效预测必须能解析，数组长度须与查询数相同，每一项还须为绝对值在 40 至 65535 的整数。

- Stage A 为本次已运行的设置：1882 个训练窗口，seed 17，每 32 个窗口更新一次，共 1000 次。训练前端，冻结 Qwen 基座和 LoRA。
- Stage B 是入口支持的联合训练配置：前端和注意力 LoRA 可训练，基座冻结。LoRA 配置为 r=16、alpha=32、dropout=0.05，作用在 q/k/v/o 四个投影。
- 正式验证在固定的 256 个窗口上使用 30% 随机遮蔽。输出有效率按全部窗口计算，MAE 按有效输出计算。本版本采用真实值监督，无教师蒸馏。

## 环境和自检

正式运行记录为 Linux、Python 3.12、PyTorch 2.5.1+cu121。以下为 Linux shell 命令，在仓库根目录执行。仅从 PCAP 重建数据时才需要系统命令 tshark。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
sha256sum -c SHA256SUMS
python src/check_audited_control.py
python src/check_audited_frontend.py
```

## 复用正式训练入口

1. 克隆或解压后，将本地项目目录命名为 `behavior_lora_d4_20261111`。原入口会校验此目录名。
2. 将已有 D4 窗口和划分文件放到 `data/windows.jsonl` 和 `data/split_manifest.json`。正式运行用的 SHA-256 分别为 `3ed258f582825d80c4e4619bc46ed6f20a3d05deae048bd7505f1971c740dd64` 和 `d063efbc914a7b101cd5926d0d56140076c7443dd125e007fea7e9e9ecd183e6`。如从原始 PCAP 开始，原始采集目录和文件名会影响样本及划分身份；按下方命令转换后应核对哈希。
3. 将 `Qwen/Qwen3-4B-Instruct-2507` 在 revision `cdbee75f17c01a7cc42f958dc650907174af0554` 的完整权重、分词器和配置放在 `model/qwen3-4b-instruct-2507/`。同级的 `model/revision.txt` 随包提供。
4. 在支持 BF16 的 CUDA 环境中，从本项目根目录运行。每次新实验使用独立的 run ID。

```bash
# 可选：从原始 D4 采集重建；已有窗口文件时跳过
python src/parse_pcap.py --root /path/to/D4 --out data/flows.jsonl
python src/build_windows.py --flows data/flows.jsonl --out data --seed 17

# 正式 Stage A 示例
python src/formal_train_v1.py --root "$PWD" --stage a --run-id stage_a_reproduction --seed 17
```

输出在 `runs/<run-id>/`。先读 `protocol.json` 核对数据和参数，再看 `validation_u*.json` 和 `completion.json`。共享包仅包含代码、依赖、版本标识和合成检查用例。正式训练的五份源码按运行协议逐文件核对，两个数据处理脚本按服务器文件核对；七份文件的校验值见 `SHA256SUMS`。
