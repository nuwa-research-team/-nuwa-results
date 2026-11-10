# 数据身份与范围

| 场景 | 论文 / 仓库名称 | 论文规模与待确认项 |
|---|---|---|
| d1 | 论文描述匹配CIC-AAGM2017；仓库链接CIC-AndMal2017 | 论文1900应用=250Adware+150GeneralMalware+1500Benign，与AAGM官方吻合；确切作者拆分尚未闭合，不能把AndMal包当AAGM |
| d2 | CIRA-CIC-DoHBrw2020 | 193,883 sessions；论文三大类、代码 Chrome/Firefox/dns2tcp/dnscat2/iodine 五标签 |
| d3 | CICIoT2023 | 197,888 sessions；六类 TCP 相关流量 |
| d4 | USTC-TFC2016 | 138,901 sessions，论文六应用；示例代码十标签、公开原始数据二十类，六类名单待确认 |

d3 六类：DoS-HTTP_Flood、DDoS-HTTP_Flood、DoS-TCP_Flood、MITM-ArpSpoofing、DDoS-TCP_Flood、DNS_Spoofing。官方浏览目录对应共 35 个 PCAP 入口。原始全数据规模可能远大于这些类别，先确认空间再完整下载。

USTC 来源：https://github.com/davidyslu/USTC-TFC2016，固定 commit `4bc9683b996f582c3843815b68da8e4dce9c7e1e`。下载的是 PCAP/7z 流量数据；不执行恶意软件，不重放到外部网络。

官方数据入口由上游 README 给出。CIC 下载需要登记；请使用官方页面，保留下载文件大小与本地 SHA-256，不保存或共享会话 cookie。

2026-11-10核对的AAGM官方说明：https://www.unb.ca/cic/datasets/android-adware.html 。其登记入口为 https://cicresearch.ca/CICDataset/CICAndAdGMal2017/ ，当前返回服务器错误。已完整下载的`Adware-PCAPs.zip`来自`CICMalAnal2017/PCAPs`（AndMal），不是此AAGM。两者分开保存、分开报告。

上游随附 IoT mask0.1/0.2/0.3 各 44,457 行，pretraining 5,000 行；缺少可核对的行 ID、标签及配对干净真值，不能作为四个完整数据集，也不能单独计算可信的分类复现分数。

必须为每条会话记录 source_file_sha256、流标识、方向、时间、长度、标签、过滤理由和 split。禁止通过匹配论文分数反推或挑选类别。
