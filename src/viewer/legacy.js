/* Explicit adapter for experiment_review/build_review.py's schema 1.
   Project-specific mappings live here, never in the general renderer. */
'use strict';
(function (root) {
  function fromLegacy(d) {
    if (d.schema !== 1 || !Array.isArray(d.rows) || !d.sources || !Array.isArray(d.files) || !Array.isArray(d.matrices) || !Array.isArray(d.differences)) throw new Error('不是受支持的本地实验快照 schema 1');
    const text = (key, label) => ({key, label, type: 'text'});
    const num = (key, label, direction = 'none', decimals = 4, extra = {}) => ({key, label, type: 'number', direction, decimals, ...extra});
    const names = {recall: 'Recall', f1: 'F1', roc_auc: 'AUROC', average_precision: 'AP', accuracy: 'Accuracy', precision: 'Precision', fpr: 'FPR', specificity: 'Specificity', mcc: 'MCC', sensitivity: 'Sensitivity', false_negative_rate: 'FNR', balanced_accuracy: 'Balanced accuracy'};
    const primary = ['recall', 'f1', 'roc_auc', 'average_precision', 'accuracy'];
    const kinds = k => d.rows.filter(r => r.kind === k);
    const tables = [], represented = new Set();
    function row(r, values) {
      represented.add(r.id);
      return {id: r.id, values, source: r.source, pointer: r.pointer, details: r};
    }
    function metricTable(id, title, rs, {rank = true, group = [], seed = false, dataset = false, config = false, fields = primary, notes = '', caption = ''} = {}) {
      const columns = [text('label', 'Method / Condition'), ...(dataset ? [text('dataset', '数据集 / 协议')] : []), ...(seed ? [num('seed', 'Seed', 'none', 0)] : []), ...(config ? [text('configuration', '实际配置')] : []), ...fields.map(k => num(k, names[k], rank ? (['fpr', 'false_negative_rate'].includes(k) ? 'min' : 'max') : 'none'))];
      tables.push({id, title, caption, notes, columns, compareBy: group, rows: rs.map(r => row(r, {label: r.label, ...(dataset ? {dataset: r.dataset ?? null} : {}), ...(seed ? {seed: r.seed ?? null} : {}), ...(config ? {configuration: r.configuration ? JSON.stringify(r.configuration) : null} : {}), ...Object.fromEntries(fields.map(k => [k, r.metrics[k] ?? null]))}))});
    }
    const summaries = kinds('formal_summary'), seeds = kinds('formal_seed');
    const full = summaries.filter(r => r.group === 'E1' && r.matrix.includes('d1-formal'));
    metricTable('paper-sota', '论文当前值 · SOTA', kinds('paper_sota'), {caption: 'Table III · Detection Performance on the Full CESNET-MINER22 Test Set', notes: '作者确认的论文展示值；与归档的差异单独列出。不会自动用归档覆盖论文。'});
    metricTable('archive-sota', '归档结果 · SOTA', [...kinds('baseline_summary'), ...full.map(r => ({...r, label: 'BCRNet (BCT-only)'}))], {notes: 'HGB、DeCrypto 为指定文件的均值，BCRNet 为正式 D1 report，其他为单种子。'});
    tables.push({id: 'differences', title: '论文与归档差异', notes: 'Δ = 论文值 − 归档值；仅展示原快照声明的不一致项。', columns: [text('method', 'Method'), text('metric', '指标'), num('paper', '论文'), num('archive', '归档', 'none', 7), num('delta', 'Δ', 'none', 7)], rows: d.differences.filter(x => !x.same).map((x, i) => ({id: `diff-${i}`, values: {method: x.method, metric: names[x.metric] ?? x.metric, paper: x.paper, archive: x.archive, delta: x.delta}, details: x}))});
    tables.push({id: 'cost', title: '模型效率与部署成本', caption: 'Table IV · Model Efficiency and Deployment Cost', notes: 'Parameters 原值单位为 M，显示时 × 1,000 转为 K。CryptoCatch 时间为分类器拟合；ERNN 为训练时间上界。计时不跨协议排名。', columns: [text('method', 'Method'), num('parameters_M', 'Parameters', 'min', 3, {scale: 1000, unit: 'K'}), num('checkpoint_MB', 'Checkpoint', 'min', 4, {unit: 'MB'}), num('training_seconds', 'Training time', 'none', 2, {unit: 's'})], rows: kinds('paper_cost').map(r => row(r, {method: r.method, parameters_M: r.values.parameters_M, checkpoint_MB: r.values.checkpoint_MB, training_seconds: r.values.training_seconds}))});
    const mean = v => typeof v === 'number' ? v : v?.mean ?? v?.value ?? null;
    tables.push({id: 'profiles', title: '推理 Profile', notes: '按源数据的设备、batch 和线程展示，不跨计时条件排名。', columns: [text('profile', 'Profile'), text('device', 'Device'), num('batch', 'Batch', 'none', 0), num('threads', 'CPU threads', 'none', 0), num('latency', 'Latency', 'none', 4, {unit: 'ms/Flow'}), num('throughput', 'Throughput', 'none', 2, {unit: 'Flows/s'})], rows: kinds('efficiency_profile').map(r => row(r, {profile: r.label, device: r.values.device, batch: r.values.protocol?.batch_size ?? null, threads: r.values.protocol?.cpu_threads ?? null, latency: mean(r.values.timing?.latency_ms_per_flow), throughput: mean(r.values.timing?.throughput_flows_per_second)}))});
    metricTable('generalization', '泛化 · 汇总', summaries.filter(r => ['E2', 'E3'].includes(r.group)), {dataset: true, group: ['dataset']});
    metricTable('generalization-seeds', '泛化 · 逐种子', seeds.filter(r => ['E2', 'E3'].includes(r.group)), {dataset: true, seed: true, group: ['dataset', 'seed']});
    metricTable('ablation', '消融 · 汇总', summaries.filter(r => r.group === 'E6' || full.some(x => x.id === r.id)), {notes: '旧完整模型和旧消融变体属于同一归档；不能与修正后的论文基准直接配对。'});
    metricTable('ablation-seeds', '消融 · 逐种子', seeds.filter(r => r.group === 'E6'), {seed: true, group: ['seed']});
    tables.push({id: 'ablation-pairs', title: '消融 · 配对增益', columns: [text('condition', 'Variant'), num('full_mean_f1', '旧完整模型 F1'), num('variant_mean_f1', '变体 F1', 'max'), num('paired_delta_mean', '配对 ΔF1', 'max'), num('n', 'n', 'none', 0)], rows: kinds('ablation_summary').map(r => row(r, Object.fromEntries(['condition', 'full_mean_f1', 'variant_mean_f1', 'paired_delta_mean', 'n'].map(k => [k, r.values[k]]))))});
    metricTable('hyperparameters', '超参数 · 汇总', summaries.filter(r => r.group === 'E7'), {config: true});
    metricTable('hyperparameters-seeds', '超参数 · 逐种子', seeds.filter(r => r.group === 'E7'), {seed: true, group: ['seed']});
    metricTable('all-metrics', '全部数值记录', kinds('inventory'), {rank: false, dataset: true, fields: Object.keys(names), notes: '保留全部 Train / Validation / Test / 子组 / 诊断数值块。行数不等于独立实验次数，混合记录不统一排名。来源路径和完整上下文可在行详情中检索。'});
    // Every original row remains accessible, including types without a paper table.
    tables.push({id: 'other-evidence', title: '补充证据', columns: [text('kind', '记录类型'), text('label', '记录'), text('values', '归档内容')], rows: d.rows.filter(r => !represented.has(r.id)).map(r => row(r, {kind: r.kind, label: r.label, values: JSON.stringify(r.values ?? r.metrics)}))});
    tables.push({id: 'files', title: '完整文件清单', columns: [text('path', '文件路径'), text('status', '状态'), num('nodes', '数值块', 'none', 0), num('bytes', '大小', 'none', 0, {unit: 'bytes'})], rows: d.files.map((f, i) => ({id: `file-${i}`, values: {path: f.path, status: f.status, nodes: f.metric_nodes, bytes: f.bytes}, source: f.source, details: f}))});
    const metadata = Object.fromEntries(Object.entries(d).filter(([k]) => !['rows', 'sources', 'files'].includes(k)));
    metadata.adapter = 'bcrnet-local-review-schema-1';
    metadata.originalRowCount = d.rows.length;
    return {format: 'experiment-review', version: 1, project: {id: 'bcrnet-local-review', title: 'BCRNet 实验三线表 · 数据核对', description: '原始实验快照已转换为通用表格；原值、来源、标准差和审计元数据完整保留。', updatedAt: d.created_at}, tables, sources: d.sources, metadata};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = fromLegacy;
  else root.fromLegacyReview = fromLegacy;
})(globalThis);
