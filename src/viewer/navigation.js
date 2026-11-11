/* Explicit table groups for the original Experiment Viewer template. */
'use strict';
(function (root) {
  const fail = message => { throw new Error(`metadata.navigation: ${message}`); };
  const text = value => typeof value === 'string' && value.trim().length > 0;
  function validate(doc) {
    if (doc.metadata?.navigation === undefined) return null;
    const config = doc.metadata.navigation;
    if (!config || !Array.isArray(config.groups) || !config.groups.length) fail('需要非空 groups');
    const known = new Set(doc.tables.map(table => table.id)), used = new Set(), groups = new Set();
    for (const group of config.groups) {
      if (!group || !text(group.id) || !text(group.label) || !text(group.description)) fail('每组需要 id、label 和 description');
      if (groups.has(group.id)) fail(`分组 ID 重复：${group.id}`);
      groups.add(group.id);
      if (!Array.isArray(group.tables) || !group.tables.length) fail(`分组没有表格：${group.id}`);
      for (const entry of group.tables) {
        if (!entry || !text(entry.id) || !text(entry.label) || !known.has(entry.id)) fail(`无效表格：${entry?.id}`);
        if (used.has(entry.id)) fail(`表格重复分组：${entry.id}`);
        used.add(entry.id);
      }
    }
    if (used.size !== known.size) fail('每张表必须明确归属一个分组');
    if (!known.has(config.defaultTable)) fail('defaultTable 不存在');
    return config;
  }
  function groupById(config, id) {
    const group = config.groups.find(item => item.id === id);
    if (!group) fail(`未知分组：${id}`);
    return group;
  }
  function groupForTable(config, id) {
    const group = config.groups.find(item => item.tables.some(table => table.id === id));
    if (!group) fail(`表格未分组：${id}`);
    return group;
  }
  const api = {validate, groupById, groupForTable};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewNavigation = api;
})(globalThis);
