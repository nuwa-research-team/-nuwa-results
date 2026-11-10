# 分批上传说明

五个一级目录名称保持data、Paper、src、results、venv。

桌面“GitHub_分批上传”按数字顺序组织，每批总量不超过10MB、文件数不超过35。
进入每一批，把里面的data/Paper/src/results/venv目录或根文件拖到GitHub仓库根目录的上传区域，再Commit changes。不要拖外层批次目录，也不要上传46MB的旧整包配置.zip。

只有两个大型证据ZIP拆成4MiB小片，位于原路径加.parts的子目录；论文PDF、训练raw.pkl、网页和数值结果都保持原文件。
下载全部仓库材料后，可执行`python restore_large_files.py`检查完整性，再执行`python restore_large_files.py --restore`恢复原ZIP。工具先核对每片和合并SHA256，已有不同文件不会被覆盖。网页展示无需恢复ZIP。

原始桌面page目录及配置.zip保留未改。这一版只是无损分批上传组织。
