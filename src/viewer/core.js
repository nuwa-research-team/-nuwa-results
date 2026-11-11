/* Shared data contract and algorithms; independent of the DOM. */
'use strict';
(function (root) {
  const fail = (path, message) => { throw new Error(`${path}: ${message}`); };
  const object = (x, path) => { if (!x || typeof x !== 'object' || Array.isArray(x)) fail(path, '必须是对象'); };
  const string = (x, path) => { if (typeof x !== 'string' || !x.trim()) fail(path, '必须是非空字符串'); };
  function fields(x, allowed, path) {
    object(x, path);
    for (const key of Object.keys(x)) if (!allowed.includes(key)) fail(`${path}.${key}`, '未知字段');
  }
  function validate(doc) {
    fields(doc, ['format', 'version', 'project', 'tables', 'sources', 'metadata'], 'document');
    if (doc.format !== 'experiment-review' || doc.version !== 1) fail('document', '需要 experiment-review / version 1');
    fields(doc.project, ['id', 'title', 'description', 'updatedAt'], 'project');
    for (const key of ['id', 'title']) string(doc.project[key], `project.${key}`);
    for (const key of ['description', 'updatedAt']) if (doc.project[key] !== undefined && typeof doc.project[key] !== 'string') fail(`project.${key}`, '必须是字符串');
    if (doc.sources !== undefined) object(doc.sources, 'sources');
    if (!Array.isArray(doc.tables) || !doc.tables.length) fail('tables', '至少需要一张表');
    const tableIds = new Set();
    doc.tables.forEach((table, ti) => {
      const p = `tables[${ti}]`;
      fields(table, ['id', 'title', 'caption', 'description', 'notes', 'columns', 'rows', 'compareBy'], p);
      for (const key of ['id', 'title']) string(table[key], `${p}.${key}`);
      for (const key of ['caption', 'description', 'notes']) if (table[key] !== undefined && typeof table[key] !== 'string') fail(`${p}.${key}`, '必须是字符串');
      if (tableIds.has(table.id)) fail(p, '表格 ID 重复');
      tableIds.add(table.id);
      if (!Array.isArray(table.columns) || !table.columns.length) fail(`${p}.columns`, '至少需要一列');
      const columns = new Map();
      table.columns.forEach((c, ci) => {
        const cp = `${p}.columns[${ci}]`;
        fields(c, ['key', 'label', 'type', 'unit', 'decimals', 'direction', 'scale'], cp);
        for (const key of ['key', 'label']) string(c[key], `${cp}.${key}`);
        if (columns.has(c.key)) fail(cp, '列 key 重复');
        if (!['text', 'number', 'boolean'].includes(c.type)) fail(cp, 'type 必须为 text、number 或 boolean');
        if (c.unit !== undefined && typeof c.unit !== 'string') fail(cp, 'unit 必须是字符串');
        if (c.direction !== undefined && !['none', 'max', 'min'].includes(c.direction)) fail(cp, 'direction 必须为 none、max 或 min');
        if (c.type !== 'number' && (c.decimals !== undefined || c.scale !== undefined || (c.direction && c.direction !== 'none'))) fail(cp, '精度、倍率和最优方向仅适用于数值列');
        if (c.decimals !== undefined && (!Number.isInteger(c.decimals) || c.decimals < 0 || c.decimals > 12)) fail(cp, 'decimals 必须为 0–12 的整数');
        if (c.scale !== undefined && (typeof c.scale !== 'number' || !Number.isFinite(c.scale) || c.scale <= 0)) fail(cp, 'scale 必须为有限正数');
        columns.set(c.key, c);
      });
      if (table.compareBy !== undefined && (!Array.isArray(table.compareBy) || table.compareBy.some(key => !columns.has(key)) || new Set(table.compareBy).size !== table.compareBy.length)) fail(p, 'compareBy 必须是不重复的列 key 数组');
      if (!Array.isArray(table.rows)) fail(`${p}.rows`, '必须是数组');
      const ids = new Set();
      table.rows.forEach((row, ri) => {
        const rp = `${p}.rows[${ri}]`;
        fields(row, ['id', 'values', 'source', 'pointer', 'details'], rp);
        string(row.id, `${rp}.id`);
        if (ids.has(row.id)) fail(rp, '行 ID 重复');
        ids.add(row.id);
        object(row.values, `${rp}.values`);
        for (const key of Object.keys(row.values)) if (!columns.has(key)) fail(`${rp}.values.${key}`, '没有对应的列声明');
        for (const [key, c] of columns) {
          if (!Object.hasOwn(row.values, key)) fail(`${rp}.values.${key}`, '缺失值请显式填写 null');
          const v = row.values[key];
          if (v === null) continue;
          if (typeof v !== ({text: 'string', number: 'number', boolean: 'boolean'})[c.type] || (c.type === 'number' && (!Number.isFinite(v) || !Number.isFinite(v * (c.scale ?? 1))))) fail(`${rp}.values.${key}`, `需要 ${c.type} 或 null`);
        }
        if (row.source !== undefined && (typeof row.source !== 'string' || !doc.sources || !Object.hasOwn(doc.sources, row.source))) fail(`${rp}.source`, '来源 ID 不存在');
        if (row.pointer !== undefined && typeof row.pointer !== 'string') fail(`${rp}.pointer`, '必须是字符串');
      });
    });
    return doc;
  }

  // Exact numeric comparison on the full table, before search, sorting or paging.
  function winners(table) {
    const groups = new Map(), result = new Map();
    for (const row of table.rows) {
      const group = (table.compareBy ?? []).map(key => row.values[key]);
      if (group.some(v => v === null)) continue;
      const id = JSON.stringify(group);
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(row);
    }
    for (const group of groups.values()) for (const col of table.columns) {
      if (!['min', 'max'].includes(col.direction)) continue;
      let best, count = 0;
      for (const row of group) {
        const v = row.values[col.key];
        if (v === null) continue;
        count++;
        if (best === undefined || (col.direction === 'max' ? v > best : v < best)) best = v;
      }
      if (count < 2) continue;
      for (const row of group) if (row.values[col.key] === best) {
        if (!result.has(row.id)) result.set(row.id, new Set());
        result.get(row.id).add(col.key);
      }
    }
    return result;
  }
  function display(value, col) {
    if (value === null) return '—';
    if (col.type === 'number') return (value * (col.scale ?? 1)).toLocaleString('en-US', {minimumFractionDigits: col.decimals ?? 4, maximumFractionDigits: col.decimals ?? 4});
    if (col.type === 'boolean') return value ? '是' : '否';
    return value;
  }
  function sorted(rows, column, descending) {
    return rows.toSorted((a, b) => {
      const av = a.values[column.key], bv = b.values[column.key];
      if (av === null || bv === null) return av === bv ? 0 : av === null ? 1 : -1;
      const cmp = column.type === 'text' ? av.localeCompare(bv) : (av < bv ? -1 : av > bv ? 1 : 0);
      return descending ? -cmp : cmp;
    });
  }

  // Deterministic JSON identity: object-key order is irrelevant; array order is significant.
  function canonical(value) {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
    return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + canonical(value[k])).join(',') + '}';
  }
  async function fingerprint(doc) {
    const hash = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical(doc)));
    return Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
  }
  const noteKey = (tableId, rowId) => JSON.stringify([tableId, rowId]);
  function validateNotes(bundle, doc, hash) {
    fields(bundle, ['format', 'version', 'projectId', 'fingerprint', 'exportedAt', 'notes'], 'notes');
    if (bundle.format !== 'experiment-review-notes' || bundle.version !== 1 || bundle.projectId !== doc.project.id || bundle.fingerprint !== hash) fail('notes', '核对单与当前数据快照不匹配');
    object(bundle.notes, 'notes.notes');
    const ids = new Set(doc.tables.flatMap(t => t.rows.map(r => noteKey(t.id, r.id))));
    for (const [id, n] of Object.entries(bundle.notes)) {
      if (!ids.has(id)) fail('notes', '包含当前快照中不存在的行');
      fields(n, ['status', 'text', 'updatedAt'], `notes.${id}`);
      if (!['pending', 'checked', 'question'].includes(n.status) || typeof n.text !== 'string' || typeof n.updatedAt !== 'string') fail(`notes.${id}`, '无效的核对记录');
    }
    return bundle.notes;
  }

  // RFC 4180-style parser with explicit delimiter; no guessed numeric types.
  function parseDelimited(input, delimiter = ',') {
    if (![',', '\t', ';'].includes(delimiter)) fail('CSV', '不支持的分隔符');
    const text = input.replace(/^\uFEFF/, '');
    const rows = []; let row = [], field = '', state = 'start';
    const endField = () => { row.push(field); field = ''; state = 'start'; };
    const endRow = () => { endField(); rows.push(row); row = []; };
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (state === 'quoted') {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else state = 'closed'; }
        else field += c;
      } else if (c === delimiter) endField();
      else if (c === '\r' || c === '\n') { if (c === '\r' && text[i + 1] === '\n') i++; endRow(); }
      else if (state === 'closed') fail('CSV', `字符 ${i + 1}：结束引号后必须是分隔符或换行`);
      else if (c === '"') { if (state !== 'start') fail('CSV', `字符 ${i + 1}：非转义引号`); state = 'quoted'; }
      else { field += c; state = 'plain'; }
    }
    if (state === 'quoted') fail('CSV', '引号未闭合');
    if (row.length || state !== 'start' || field.length) endRow();
    if (!rows.length) fail('CSV', '文件为空');
    const headers = rows.shift();
    if (headers.some(h => !h.trim()) || new Set(headers).size !== headers.length) fail('CSV', '表头必须非空且不重复');
    rows.forEach((r, i) => { if (r.length !== headers.length) fail('CSV', `数据行 ${i + 1} 有 ${r.length} 列，表头有 ${headers.length} 列`); });
    return {headers, rows};
  }
  function csvDocument(parsed, config) {
    const {columns, projectId, title, tableId, compareBy, idColumn} = config;
    if (columns.length !== parsed.headers.length || columns.some((c, i) => c.key !== parsed.headers[i])) fail('CSV', '列配置与表头不匹配');
    const rows = parsed.rows.map((cells, i) => ({
      id: idColumn === null ? `row-${i + 1}` : cells[parsed.headers.indexOf(idColumn)],
      values: Object.fromEntries(columns.map((c, j) => {
        const cell = cells[j];
        if (cell === '') return [c.key, null];
        if (c.type === 'text') return [c.key, cell];
        if (c.type === 'number') {
          if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(cell.trim()) || !Number.isFinite(Number(cell))) fail('CSV', `数据行 ${i + 1}，列 ${c.key} 不是有限十进制数`);
          return [c.key, Number(cell)];
        }
        if (!['true', 'false'].includes(cell)) fail('CSV', `数据行 ${i + 1}，列 ${c.key} 需要 true 或 false`);
        return [c.key, cell === 'true'];
      }))
    }));
    return validate({format: 'experiment-review', version: 1, project: {id: projectId, title}, tables: [{id: tableId, title, columns, compareBy, rows}]});
  }
  const api = {validate, winners, display, sorted, canonical, fingerprint, noteKey, validateNotes, parseDelimited, csvDocument};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewCore = api;
})(globalThis);
