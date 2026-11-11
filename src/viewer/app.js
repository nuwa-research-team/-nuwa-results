'use strict';
(function () {
  const C = globalThis.ReviewCore, N = globalThis.ReviewNavigation, $ = id => document.getElementById(id);
  const cleanHTML = '<!doctype html>\n' + document.documentElement.outerHTML;
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const states = {pending: '未核对', checked: '已核对', question: '有疑问'};
  let navigation = null;
  let doc = null, hash = '', storageKey = '', notes = {}, active = 0, page = 0, sort = null, selection = null, busy = false, csvInput = null, printing = false;
  const bestCache = new Map(), searchCache = new Map();
  const table = () => doc.tables[active];
  const status = (t, r) => notes[C.noteKey(t.id, r.id)]?.status ?? 'pending';
  function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); $('message').hidden = false; }
  async function task(fn, dialog = null) {
    if (busy) { message('正在处理数据，请稍候。'); return; }
    busy = true; document.body.setAttribute('aria-busy', 'true');
    if (dialog) dialog.querySelector('.dialog-error').textContent = '';
    try { await fn(); } catch (err) { if (dialog?.open) dialog.querySelector('.dialog-error').textContent = err.message; else message(err.message, true); }
    finally { busy = false; document.body.removeAttribute('aria-busy'); }
  }
  function download(name, content, type = 'application/json') {
    const blob = new Blob([typeof content === 'string' ? content : JSON.stringify(content, null, 2)], {type: `${type};charset=utf-8`});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function noteBundle(next = notes) { return {format: 'experiment-review-notes', version: 1, projectId: doc.project.id, fingerprint: hash, exportedAt: new Date().toISOString(), notes: next}; }
  function persist(next) {
    // A failed durable write is a failed save. Do not claim an in-memory save succeeded.
    localStorage.setItem(storageKey, JSON.stringify(noteBundle(next)));
    notes = next;
  }
  function updateURL() {
    const url = new URL(location.href); url.hash = encodeURIComponent(table().id);
    history.replaceState(null, '', url);
  }
  async function activate(input, name, {fromURL = false} = {}) {
    const converted = input?.schema === 1 ? globalThis.fromLegacyReview(input) : input;
    const next = C.validate(converted), nextNavigation = N.validate(next), nextHash = await C.fingerprint(next);
    const key = `experiment-viewer/v1/${encodeURIComponent(next.project.id)}/${nextHash}`;
    const saved = localStorage.getItem(key);
    const nextNotes = saved === null ? {} : C.validateNotes(JSON.parse(saved), next, nextHash);
    const requested = decodeURIComponent(location.hash.slice(1));
    // Commit only after the complete input and saved annotations have been validated.
    doc = next; navigation = nextNavigation; hash = nextHash; storageKey = key; notes = nextNotes;
    bestCache.clear(); searchCache.clear();
    const requestedIndex = doc.tables.findIndex(t => t.id === requested);
    active = requestedIndex >= 0 ? requestedIndex : navigation ? doc.tables.findIndex(t => t.id === navigation.defaultTable) : 0;
    page = 0; sort = null; selection = null;
    $('query').value = ''; $('status-filter').value = 'all';
    $('project-title').textContent = doc.project.title;
    $('project-description').textContent = doc.project.description ?? '';
    $('project-meta').textContent = `${doc.tables.length} 张表 · ${doc.tables.reduce((n, t) => n + t.rows.length, 0).toLocaleString()} 行 · ${name}` + (doc.project.updatedAt ? ` · ${doc.project.updatedAt}` : '');
    document.title = doc.project.title + ' · DNS2027';
    $('welcome').hidden = true; $('workspace').hidden = false;
    for (const id of ['details-dialog', 'info-dialog']) if ($(id).open) $(id).close();
    if (!fromURL) { const url = new URL(location.href); url.searchParams.delete('data'); history.replaceState(null, '', url); }
    renderNavigation(); render(); updateURL();
    message(input?.schema === 1 ? '已导入原实验快照。来源、原始数值及审计元数据均已保留；旧版核对单需在旧页面查看。' : '数据已载入。点击任意行的第一列，可查看原始数据并记录核对备注。');
  }
  function renderNavigation() {
    const group = navigation ? N.groupForTable(navigation, table().id) : null;
    $('group-label').hidden = !group; $('group-description').hidden = !group;
    $('table-group').innerHTML = group ? navigation.groups.map(g => `<option value="${esc(g.id)}"${g.id === group.id ? ' selected' : ''}>${esc(g.label)}</option>`).join('') : '';
    $('group-description').textContent = group?.description ?? '';
    const entries = group ? group.tables : doc.tables.map(t => ({id: t.id, label: t.title}));
    $('navigation').innerHTML = entries.map(entry => {
      const i = doc.tables.findIndex(t => t.id === entry.id), t = doc.tables[i];
      return `<button data-table="${i}"${i === active ? ' aria-current="page"' : ''}>${esc(entry.label)} <span class="small">${t.rows.length}</span></button>`;
    }).join('');
  }
  function selectTable(index) {
    active = index; page = 0; sort = null; $('query').value = ''; $('status-filter').value = 'all';
    renderNavigation(); render(); updateURL();
  }
  function searchText(t) {
    if (!searchCache.has(t.id)) searchCache.set(t.id, new Map(t.rows.map(r => [r.id, JSON.stringify([r.id, r.values, r.details, doc.sources?.[r.source]?.paths ?? []]).toLocaleLowerCase()])));
    return searchCache.get(t.id);
  }
  function filteredRows(t) {
    const query = $('query').value.toLocaleLowerCase().trim(), filter = $('status-filter').value;
    const index = query ? searchText(t) : null;
    let rows = t.rows.filter(r => (!query || index.get(r.id).includes(query)) && (filter === 'all' || status(t, r) === filter));
    if (sort) rows = C.sorted(rows, t.columns[sort.index], sort.descending);
    return rows;
  }
  function render() {
    if (!doc) return;
    const t = table(), rows = filteredRows(t), size = Number($('page-size').value), pages = Math.max(1, Math.ceil(rows.length / size));
    page = Math.min(page, pages - 1);
    const shown = printing ? rows : rows.slice(page * size, (page + 1) * size);
    if (!bestCache.has(t.id)) bestCache.set(t.id, C.winners(t));
    const best = bestCache.get(t.id), audit = $('audit').checked && !printing;
    $('table-title').textContent = t.title; $('table-caption').textContent = t.caption ?? '';
    $('table-description').textContent = t.description ?? ''; $('table-notes').textContent = t.notes ?? '';
    const ranked = t.columns.some(c => ['max', 'min'].includes(c.direction));
    const groups = (t.compareBy ?? []).map(k => t.columns.find(c => c.key === k).label);
    $('legend').textContent = ranked ? `黄色标记原始数值最优项，并列均标记。比较范围：${groups.length ? groups.join(' × ') + ' 相同的记录' : '整张表'}；按完整表计算，不随搜索或翻页变化，不代表统计显著性。缺少分组值的行不排名。` : '此表未声明最优值比较。数值按声明精度显示；点击第一列查看完整原值。';
    $('table-container').innerHTML = `<table><thead><tr>${t.columns.map((c, i) => `<th scope="col" aria-sort="${sort?.index === i ? sort.descending ? 'descending' : 'ascending' : 'none'}"><button class="sort" data-sort="${i}">${esc(c.label)}${c.unit ? ` (${esc(c.unit)})` : ''}<span class="arrow">${c.direction === 'max' ? '↑' : c.direction === 'min' ? '↓' : ''}${sort?.index === i ? sort.descending ? ' ▾' : ' ▴' : ''}</span></button></th>`).join('')}${audit ? '<th class="audit-col">核对</th>' : ''}</tr></thead><tbody>${shown.map((r, ri) => `<tr>${t.columns.map((c, ci) => { const value = esc(C.display(r.values[c.key], c)); const content = best.get(r.id)?.has(c.key) ? `<mark title="同一声明分组内原始数值最优">${value}</mark>` : value; return `<td${c.type === 'text' ? ' class="text-cell"' : ''}>${ci === 0 ? `<button class="cell-link" data-detail="${printing ? ri : page * size + ri}">${content}</button>` : content}</td>`; }).join('')}${audit ? `<td class="audit-col"><span class="badge ${status(t, r)}">${states[status(t, r)]}</span></td>` : ''}</tr>`).join('') || `<tr><td colspan="${t.columns.length + Number(audit)}" class="empty">没有匹配的记录</td></tr>`}</tbody></table>`;
    $('page-info').textContent = `${rows.length.toLocaleString()} / ${t.rows.length.toLocaleString()} 行 · 第 ${page + 1} / ${pages} 页`;
    $('prev').disabled = page === 0; $('next').disabled = page >= pages - 1;
    $('review-info').textContent = `此数据快照已记录 ${Object.keys(notes).length} 条核对备注 · 数据 SHA-256 ${hash.slice(0, 12)}`;
  }
  function detail(index) {
    const t = table(), r = filteredRows(t)[index];
    selection = {table: t, row: r};
    $('detail-title').textContent = C.display(r.values[t.columns[0].key], t.columns[0]);
    $('detail-values').textContent = JSON.stringify(r, null, 2);
    $('detail-source').textContent = r.source === undefined ? '此记录未附来源。' : JSON.stringify(doc.sources[r.source], null, 2);
    const note = notes[C.noteKey(t.id, r.id)];
    $('review-status').value = note?.status ?? 'pending'; $('review-text').value = note?.text ?? ''; $('save-state').textContent = '';
    $('details-dialog').showModal();
  }
  function info(title, html) { $('info-title').textContent = title; $('info-body').innerHTML = html; $('info-dialog').showModal(); }
  async function loadURL(value) {
    const url = new URL(value, location.href);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) throw new Error('请输入不含登录凭据的 HTTP / HTTPS 数据地址。');
    const response = await fetch(url, {credentials: 'omit', referrerPolicy: 'no-referrer'});
    if (!response.ok) throw new Error(`数据链接返回 HTTP ${response.status}`);
    const input = await response.json();
    await activate(input, url.pathname.split('/').at(-1) || '远程 JSON', {fromURL: true});
    const address = new URL(location.href); address.searchParams.set('data', url.href); history.replaceState(null, '', address);
  }
  function csvMapping() {
    const delimiter = $('csv-delimiter').value === 'tab' ? '\t' : $('csv-delimiter').value;
    csvInput.parsed = C.parseDelimited(csvInput.text, delimiter);
    const {headers, rows} = csvInput.parsed;
    $('csv-id').innerHTML = '<option value="">按行号生成</option>' + headers.map((h, i) => `<option value="${i}">${esc(h)}</option>`).join('');
    $('csv-columns').innerHTML = `<table><thead><tr><th>原始表头</th><th>类型</th><th>小数位</th><th>单位</th><th>最优方向</th><th>比较分组</th></tr></thead><tbody>${headers.map((h, i) => `<tr data-column="${i}"><td>${esc(h)}</td><td><select data-type aria-label="${esc(h)} 类型"><option value="text">文本</option><option value="number">数值</option><option value="boolean">布尔</option></select></td><td><input data-decimals type="number" value="4" min="0" max="12" step="1" disabled aria-label="${esc(h)} 小数位"></td><td><input data-unit type="text" aria-label="${esc(h)} 单位"></td><td><select data-direction disabled aria-label="${esc(h)} 最优方向"><option value="none">不比较</option><option value="max">越大越好 ↑</option><option value="min">越小越好 ↓</option></select></td><td><input data-group type="checkbox" aria-label="按 ${esc(h)} 分组"></td></tr>`).join('')}</tbody></table>`;
    $('csv-preview').textContent = `${rows.length} 行 · ${headers.length} 列。第一行数据：${rows.length ? JSON.stringify(rows[0]) : '无数据'}`;
    $('csv-submit').disabled = false;
  }
  async function readFile(file) {
    const extension = file.name.split('.').at(-1).toLowerCase(), text = await file.text();
    if (extension === 'json') return activate(JSON.parse(text), file.name);
    if (!['csv', 'tsv'].includes(extension)) throw new Error('请选择 JSON、CSV 或 TSV 文件。Excel 可另存为 UTF-8 CSV 后导入。');
    csvInput = {text, name: file.name}; $('csv-title').value = file.name.replace(/\.[^.]+$/, '');
    $('csv-delimiter').value = extension === 'tsv' ? 'tab' : ',';
    $('csv-dialog').querySelector('.dialog-error').textContent = '';
    $('csv-dialog').showModal();
    try { csvMapping(); } catch (err) { $('csv-submit').disabled = true; $('csv-dialog').querySelector('.dialog-error').textContent = err.message; }
  }
  async function importNotes(file) {
    const incoming = C.validateNotes(JSON.parse(await file.text()), doc, hash);
    const conflicts = Object.keys(incoming).filter(k => Object.hasOwn(notes, k) && C.canonical(notes[k]) !== C.canonical(incoming[k]));
    let mode = 'replace';
    if (conflicts.length) {
      $('merge-info').textContent = `有 ${conflicts.length} 条核对记录与当前版本不同。请选择冲突项的处理方式，其余记录会合并。`;
      mode = await new Promise(resolve => { $('merge-dialog').addEventListener('close', () => resolve($('merge-dialog').returnValue), {once: true}); $('merge-dialog').returnValue = 'cancel'; $('merge-dialog').showModal(); });
      if (!['keep', 'replace'].includes(mode)) return;
    }
    persist(mode === 'keep' ? {...incoming, ...notes} : {...notes, ...incoming});
    render(); message('核对单已导入并保存。');
  }
  $('import-data').onclick = $('welcome-import').onclick = () => $('data-file').click();
  $('data-file').onchange = e => { const file = e.target.files[0]; e.target.value = ''; if (file) task(() => readFile(file)); };
  $('sample').onclick = () => task(() => activate(JSON.parse($('sample-data').textContent), '演示数据（非真实实验）'));
  $('load-url').onclick = () => { $('url-dialog').querySelector('.dialog-error').textContent = ''; $('url-dialog').showModal(); };
  $('url-form').onsubmit = e => { e.preventDefault(); task(async () => { await loadURL($('data-url').value); $('url-dialog').close(); }, $('url-dialog')); };
  $('navigation').onclick = e => { const button = e.target.closest('[data-table]'); if (button) selectTable(Number(button.dataset.table)); };
  $('table-group').onchange = () => { const group = N.groupById(navigation, $('table-group').value); selectTable(doc.tables.findIndex(t => t.id === group.tables[0].id)); };
  $('table-container').onclick = e => { const button = e.target.closest('button'); if (!button) return; if (button.dataset.detail !== undefined) detail(Number(button.dataset.detail)); if (button.dataset.sort !== undefined) { const index = Number(button.dataset.sort); sort = {index, descending: sort?.index === index && !sort.descending}; page = 0; render(); } };
  for (const id of ['query', 'status-filter', 'page-size', 'audit']) $(id).addEventListener(id === 'query' ? 'input' : 'change', () => { page = 0; render(); });
  $('prev').onclick = () => { page--; render(); }; $('next').onclick = () => { page++; render(); };
  $('review-form').onsubmit = e => { e.preventDefault(); try { const key = C.noteKey(selection.table.id, selection.row.id); persist({...notes, [key]: {status: $('review-status').value, text: $('review-text').value, updatedAt: new Date().toISOString()}}); render(); $('save-state').textContent = '已保存到此浏览器。'; } catch (err) { $('save-state').textContent = '保存失败：' + err.message; } };
  $('export-data').onclick = () => download('experiment-data.json', doc);
  $('export-notes').onclick = () => download('experiment-notes.json', noteBundle());
  $('import-notes').onclick = () => $('notes-file').click();
  $('notes-file').onchange = e => { const file = e.target.files[0]; e.target.value = ''; if (file) task(() => importNotes(file)); };
  $('export-html').onclick = () => {
    const standalone = new DOMParser().parseFromString(cleanHTML, 'text/html');
    standalone.getElementById('embedded-data').textContent = JSON.stringify(doc).replace(/</g, '\\u003c').replace(/>/g, '\\u003e').replace(/&/g, '\\u0026');
    download('experiment-report.html', '<!doctype html>\n' + standalone.documentElement.outerHTML, 'text/html');
    message('已导出自包含 HTML。核对备注请另外导出核对单。');
  };
  $('print').onclick = () => window.print();
  window.addEventListener('beforeprint', () => { printing = true; render(); });
  window.addEventListener('afterprint', () => { printing = false; render(); });
  $('metadata').onclick = () => info('数据说明与审计元数据', `<p class="small">数据 SHA-256：${hash}</p><pre>${esc(JSON.stringify({project: doc.project, metadata: doc.metadata ?? {}}, null, 2))}</pre>`);
  $('help').onclick = () => info('一次保存，持续复用', '<ol><li>直接打开这个 HTML，选择 JSON、CSV 或 TSV 数据。也支持原实验页导出的 <code>data.json</code>。</li><li>CSV 首行为表头；请配置数值列、单位、小数位和比较分组。Excel 请另存为 UTF-8 CSV。</li><li>JSON 使用 <code>experiment-review / version 1</code> 格式。点“查看示例”后导出 JSON，即可作为模板。</li><li>点击第一列查看完整原值、来源并填写备注。核对单与完整数据快照的 SHA-256 绑定，数据变更后不会误用旧备注。</li><li>打印会输出当前表筛选后的全部行；“导出当前 HTML”可生成包含这份数据的独立报告。</li><li>页面可放在 GitHub Pages 等静态托管中。用 <code>?data=reports/run.json</code> 指定要读取的 JSON；从链接读取需服务端允许 CORS。</li></ol><p>文件导入、表格渲染和备注保存在浏览器内。页面没有后端，不会自动将本地数据或备注上传到 Git。</p>');
  $('home').onclick = e => { e.preventDefault(); $('welcome').hidden = false; $('workspace').hidden = true; $('project-title').textContent = '一份页面，每一组实验。'; $('project-description').textContent = '导入数据，生成三线表。表格、指标与来源，随你的实验而变。'; $('project-meta').textContent = ''; };
  document.addEventListener('click', e => { const button = e.target.closest('[data-close]'); if (button) $(button.dataset.close).close(); });
  $('csv-delimiter').onchange = () => { $('csv-dialog').querySelector('.dialog-error').textContent = ''; try { csvMapping(); } catch (err) { $('csv-submit').disabled = true; $('csv-dialog').querySelector('.dialog-error').textContent = err.message; } };
  $('csv-columns').onchange = e => { if (!e.target.matches('[data-type]')) return; const row = e.target.closest('tr'), number = e.target.value === 'number'; row.querySelector('[data-decimals]').disabled = !number; row.querySelector('[data-direction]').disabled = !number; if (!number) row.querySelector('[data-direction]').value = 'none'; };
  $('csv-form').onsubmit = e => { e.preventDefault(); task(async () => {
    const mapped = [...$('csv-columns').querySelectorAll('[data-column]')];
    const columns = mapped.map((tr, i) => { const type = tr.querySelector('[data-type]').value; return {key: csvInput.parsed.headers[i], label: csvInput.parsed.headers[i], type, unit: tr.querySelector('[data-unit]').value, ...(type === 'number' ? {decimals: Number(tr.querySelector('[data-decimals]').value), direction: tr.querySelector('[data-direction]').value} : {})}; });
    const data = C.csvDocument(csvInput.parsed, {columns, projectId: $('csv-project').value, title: $('csv-title').value, tableId: 'results', compareBy: mapped.flatMap((tr, i) => tr.querySelector('[data-group]').checked ? [columns[i].key] : []), idColumn: $('csv-id').value === '' ? null : csvInput.parsed.headers[Number($('csv-id').value)]});
    await activate(data, csvInput.name); $('csv-dialog').close();
  }, $('csv-dialog')); };
  let dragDepth = 0;
  document.addEventListener('dragenter', e => { if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); } });
  document.addEventListener('dragleave', () => { dragDepth--; if (dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
  document.addEventListener('dragover', e => { if (e.dataTransfer.types.includes('Files')) e.preventDefault(); });
  document.addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; document.body.classList.remove('dragging'); const files = [...e.dataTransfer.files]; if (files.length !== 1) { message('每次请导入一个数据文件。', true); return; } task(() => readFile(files[0])); });
  const embedded = JSON.parse($('embedded-data').textContent), dataURL = new URL(location.href).searchParams.get('data');
  if (dataURL) task(() => loadURL(dataURL));
  else if (embedded) task(() => activate(embedded, '内嵌数据'));
})();
