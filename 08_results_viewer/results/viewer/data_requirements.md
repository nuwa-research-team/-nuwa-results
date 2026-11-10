# Nuwa 两项复现：仍需补齐的数据和口径

仅做缺失包鲁棒性和插补方法 SOTA。USTC 原始包已齐，已完成首轮工程评估；其他场景缺文件，不能用不相干数据替代。

1. **CIC-AAGM2017**：需要完整 `AndroidAdware2017.zip`，不是56字节的 `AndroidAdware2017-md5sum.txt`。论文描述1900应用、250 adware / 150 general malware / 1500 benign，与该数据集吻合；仓库却链接AndMal2017，因此还需作者确认实际d1来源和样本划分。
   官方说明：https://www.unb.ca/cic/datasets/android-adware.html
   登记入口：https://cicresearch.ca/CICDataset/CICAndAdGMal2017/
2. **CIRA-CIC-DoHBrw-2020**：需要论文采用的原始PCAP或作者预处理后的有ID序列及划分。当前尚未下载完整数据。
   官方说明：https://www.unb.ca/cic/datasets/dohbrw-2020.html
   登记入口：https://cicresearch.ca/CICDataset/DoHBrw-2020/
3. **CIC-IoT-2023**：目标清单35个PCAP，目前只有DNS_Spoofing完整，其余34个待补齐；逐文件清单见已有“手动下载清单_Nuwa两项实验_20261110.md”。
   官方说明：https://www.unb.ca/cic/datasets/iotdataset-2023.html
   登记入口：https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/
4. **CIC-AndMal2017 的 Benign-PCAPs.zip**：在本机检查路径仅找到51字节MD5文件。已有8,143,605,444字节Adware包完整，但它不是AAGM，不能据此声称论文d1已齐。

优先向师兄或作者索取：论文d1-d4的类别名单、精确流ID与train/val/test划分、原始包到特征的版本与规则、F1 averaging、随机/Poisson评价丢包定义、六分类器及TFI实际训练配置。若能提供作者处理后的完整特征和划分，可减少重复下载与重建差异。

已取得的四份作者权重本地副本不完整，尚不能用于推理；不能据此认定作者原文件损坏。共享仓库不上传凭据，不把多GB原始数据当普通网页文件上传。
