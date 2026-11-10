# Nuwa 两项实验：手动下载清单

范围：补全方法SOTA、缺包鲁棒性。只下载原始PCAP及校验文件，不下载APK程序或用聚合CSV替代包序列。

## 已有及正在下载

- [x] USTC-TFC2016：完整原始数据已到服务器。
- [x] CIC-AndMal2017 Adware-PCAPs.zip：8,143,605,444字节，两端MD5均与官方一致。
- [x] CIC-IoT-2023 DNS_Spoofing.pcap：971,824,063字节，已上传。
- [ ] CIC-AndMal2017 Benign-PCAPs.zip：用户已手动启动；尚未确认完整。其.md5已有。

当前AndMal页上的Ransomware/SMSmalware/Scareware先不下载；它们不能充当AAGM的General Malware。

## 1. CIC-AAGM2017（优先）

官方说明：https://www.unb.ca/cic/datasets/android-adware.html

登记入口：https://cicresearch.ca/CICDataset/CICAndAdGMal2017/

下载这套的原始PCAP，覆盖Adware、General Malware、Benign三类，并下载其校验文件（若提供）。实际文件名以官方目录为准；本清单不猜文件名。

版本说明：论文1900应用/250 Adware/150 General/1500 Benign的描述与官方AAGM一致；仓库却链接到AndMal2017。已有AndMal包不能直接标成AAGM，也不能证明作者的精确划分。

## 2. CIRA-CIC-DoHBrw2020

官方说明：https://www.unb.ca/cic/datasets/dohbrw-2020.html

登记入口：https://cicresearch.ca/CICDataset/DoHBrw-2020/

下载原始PCAP及校验文件，覆盖Chrome、Firefox、dns2tcp、DNSCat2、Iodine流量，以构建Non-DoH、Benign-DoH、Malicious-DoH三大类。CSV特征表不足以复现包序列补全实验。

## 3. CIC-IoT-2023：仅论文所需六类

官方说明：https://www.unb.ca/cic/datasets/iotdataset-2023.html

登记入口：https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/

以下链接来自先前已观察的官方目录；需先完成官方登记。总计35文件，DNS_Spoofing已完成，剩34。不要下载整个IoT总压缩包。

### DoS-HTTP_Flood（2文件）

- [ ] [DoS-HTTP_Flood.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-HTTP_Flood%2FDoS-HTTP_Flood.pcap)
- [ ] [DoS-HTTP_Flood1.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-HTTP_Flood%2FDoS-HTTP_Flood1.pcap)

### DDoS-HTTP_Flood（1文件）

- [ ] [DDoS-HTTP_Flood-.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-HTTP_Flood%2FDDoS-HTTP_Flood-.pcap)

### DoS-TCP_Flood（11文件）

- [ ] [DoS-TCP_Flood.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood.pcap)
- [ ] [DoS-TCP_Flood1.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood1.pcap)
- [ ] [DoS-TCP_Flood10.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood10.pcap)
- [ ] [DoS-TCP_Flood2.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood2.pcap)
- [ ] [DoS-TCP_Flood3.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood3.pcap)
- [ ] [DoS-TCP_Flood4.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood4.pcap)
- [ ] [DoS-TCP_Flood5.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood5.pcap)
- [ ] [DoS-TCP_Flood6.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood6.pcap)
- [ ] [DoS-TCP_Flood7.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood7.pcap)
- [ ] [DoS-TCP_Flood8.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood8.pcap)
- [ ] [DoS-TCP_Flood9.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDoS-TCP_Flood%2FDoS-TCP_Flood9.pcap)

### MITM-ArpSpoofing（2文件）

- [ ] [MITM-ArpSpoofing.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FMITM-ArpSpoofing%2FMITM-ArpSpoofing.pcap)
- [ ] [MITM-ArpSpoofing1.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FMITM-ArpSpoofing%2FMITM-ArpSpoofing1.pcap)

### DDoS-TCP_Flood（18文件）

- [ ] [DDoS-TCP_Flood.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood.pcap)
- [ ] [DDoS-TCP_Flood1.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood1.pcap)
- [ ] [DDoS-TCP_Flood10.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood10.pcap)
- [ ] [DDoS-TCP_Flood11.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood11.pcap)
- [ ] [DDoS-TCP_Flood12.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood12.pcap)
- [ ] [DDoS-TCP_Flood13.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood13.pcap)
- [ ] [DDoS-TCP_Flood14.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood14.pcap)
- [ ] [DDoS-TCP_Flood15.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood15.pcap)
- [ ] [DDoS-TCP_Flood16.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood16.pcap)
- [ ] [DDoS-TCP_Flood17.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood17.pcap)
- [ ] [DDoS-TCP_Flood2.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood2.pcap)
- [ ] [DDoS-TCP_Flood3.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood3.pcap)
- [ ] [DDoS-TCP_Flood4.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood4.pcap)
- [ ] [DDoS-TCP_Flood5.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood5.pcap)
- [ ] [DDoS-TCP_Flood6.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood6.pcap)
- [ ] [DDoS-TCP_Flood7.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood7.pcap)
- [ ] [DDoS-TCP_Flood8.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood8.pcap)
- [ ] [DDoS-TCP_Flood9.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDDoS-TCP_Flood%2FDDoS-TCP_Flood9.pcap)

### DNS_Spoofing（1文件）

- [x] [DNS_Spoofing.pcap](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file=PCAP%2FDNS_Spoofing%2FDNS_Spoofing.pcap)（已有，不重复下）

## 下载与交接

- 每次下载1–2个大文件，完成后核对大小、PCAP/ZIP结构和官方校验值，再上传服务器。
- 当前默认Downloads目录的DSH上传监视器已启动；不要改名或把未完成的.crdownload当完整包。
- 当前C盘约74GB、D盘约313GB可用，不能假设所有原始PCAP都能同时放进C盘。分批下载，已完成文件可由我们整理到D盘。
- 官网若仍显示Server error，保留错误状态，不把注册失败记为数据已获得。Chrome安全提示由用户本人处理，不共享Cookie或Token。
