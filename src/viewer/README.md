# 结果页面源码

沿用 `Experiment-Viewer-2026-10-21.zip` 的版式、三线表、数据校验和交互。左上角名称为 DNS2027；`navigation.js` 按数据文件中的明确分组组织表格。

修改 `results/viewer/experiment_data.json` 后，在仓库根目录运行 `python3 src/viewer/build.py`，生成可离线打开的 `results/viewer/index.html`。需要 Python 3 和 Node.js 20+。
