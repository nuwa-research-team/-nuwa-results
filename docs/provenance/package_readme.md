# Nuwa 复现材料配置包

按用户指定布局整理，原始目录未删除、未改动；本包未自动上传到GitHub。

```text
data/       实际USTC训练序列、划分和下载清单
Paper/      论文PDF、补充材料、复现说明
src/        第四批官方Nuwa源码（原样复制）
results/    实测结果、展示页面、预测/检查点/执行代码证据
venv/       Miniforge配置、依赖清单和创建说明
```

没有methods或outputs目录。之前第三批辅助代码和第五批基线代码保留为results/evidence内的参考归档，未混称作者源码。

打开根目录index.html会进入results/viewer/index.html。
已完成的是USTC八类数据上的缺包鲁棒性与插补SOTA工程对照；并未完成论文d1-d4四场景同口径复现。

## 上传

建议在GitHub仓库根目录依次上传Paper、src、data、results、venv，再上传根目录README.md、index.html和配置文件。不要把外层“配置”目录一起套进去，才能保留该目录结构及Pages首页入口。
完整预测/代码以ZIP保存，便于网页上传；单个ZIP仍是可解压逐文件核验的真实产物。
大PCAP和四份完整MLM模型权重不在本包，下载位置和范围已说明。

## 来源与复现边界

官方源码来自Timeless-zfqi/Nuwa固定commit；实际工程评估代码另存results/evidence/execution_code.zip。
历史源码修订、协议初始哈希偏离与CPU核验范围均保留，不宣称全链独立重训或逐位官方复现。
Miniforge仅提供环境配置和清单；原训练使用Linux Python venv，新配置未安装测试。


## 分批上传版本

两个大型证据ZIP已无损拆片；查看UPLOAD_GUIDE.md。安装环境、论文指标和真实实验结果没有因此改变。原ZIP可通过restore_large_files.py恢复。
