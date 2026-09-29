/*
 * "What model should I pick?" — the benchmark guide modal in /chat.
 *
 * Data: #model-guide-data (json_script from llm.model_benchmarks.build_model_guide),
 * already restricted to the organisation's enabled models. One page per benchmark;
 * the list shows each model at its best recorded effort, a model's detail page
 * shows every recorded effort. Paging from a detail page keeps the model.
 * Bars are coloured by the model's $ price group (tokens --mg-tier-N in chat.html).
 */
(function () {
  'use strict';

  var TOP_N = 5;
  var EFFORT_ORDER = ['none', 'off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
  var EFFORT_LABELS = {
    none: 'Reasoning off', off: 'Reasoning off', minimal: 'Minimal', low: 'Low',
    medium: 'Medium', high: 'High', xhigh: 'Extra high', max: 'Max'
  };
  var SVG_NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      var value = attrs[key];
      if (value === null || value === undefined || value === false) return;
      if (key === 'text') node.textContent = value;
      else if (key === 'style') node.style.cssText = value;
      else if (key === 'onclick') node.addEventListener('click', value);
      else node.setAttribute(key, value === true ? '' : value);
    });
    (children || []).forEach(function (child) {
      if (child) node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    });
    return node;
  }

  function backArrow() {
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '14');
    svg.setAttribute('height', '14');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    ['m12 19-7-7 7-7', 'M19 12H5'].forEach(function (d) {
      var path = document.createElementNS(SVG_NS, 'path');
      path.setAttribute('d', d);
      svg.appendChild(path);
    });
    return svg;
  }

  function dollars(level) { return new Array(level + 1).join('$'); }

  function tierColor(level) { return 'var(--mg-tier-' + Math.max(1, Math.min(5, level || 1)) + ')'; }

  function effortLabel(effort) {
    return effort ? (EFFORT_LABELS[effort] || effort) : 'Default';
  }

  function effortRank(effort) {
    var i = EFFORT_ORDER.indexOf(effort);
    return i === -1 ? EFFORT_ORDER.length : i;
  }

  function init() {
    var dataNode = document.getElementById('model-guide-data');
    var modal = document.getElementById('model-guide-modal');
    var trigger = document.getElementById('model-guide-btn');
    if (!dataNode || !modal || !trigger) return;

    var guide;
    try { guide = JSON.parse(dataNode.textContent); } catch (e) { return; }
    var benchmarks = (guide && guide.benchmarks) || [];
    if (!benchmarks.length || !(guide.models || []).length) {
      trigger.classList.add('hidden');
      return;
    }

    var titleEl = document.getElementById('mg-title');
    var questionEl = document.getElementById('mg-question');
    var countEl = document.getElementById('mg-count');
    var filtersEl = document.getElementById('mg-filters');
    var body = document.getElementById('model-guide-body');
    var footer = document.getElementById('mg-footer');
    var models = {};
    guide.models.forEach(function (m) { models[m.id] = m; });

    var priceLevels = [];
    guide.models.forEach(function (m) {
      if (m.price_level > 0 && priceLevels.indexOf(m.price_level) === -1) priceLevels.push(m.price_level);
    });
    priceLevels.sort(function (a, b) { return a - b; });

    var state = {
      bench: 0,
      model: null,        // detail view when set
      notice: '',         // one-line message above the list
      showAll: false,
      // [] = all price groups ("All" chip), the default.
      prices: []
    };
    var lastFocus = null;

    function isPercent(bench) { return bench.metric.indexOf('%') !== -1; }

    function formatScore(bench, score) {
      return isPercent(bench) ? score.toFixed(1) + '%' : String(Math.round(score));
    }

    // Longer bar = better, always. For a lower-is-better % metric the bar
    // length is 100 − value while the label shows the value itself.
    function goodness(bench, score) {
      if (bench.higher_is_better) return score;
      return isPercent(bench) ? 100 - score : -score;
    }

    function inPriceGroups(modelId) {
      var m = models[modelId];
      return !!m && (!state.prices.length || state.prices.indexOf(m.price_level) !== -1);
    }

    function visibleRows(bench) {
      return bench.rows.filter(function (r) { return inPriceGroups(r.model_id); });
    }

    // One scale per page (list + detail share it): percent metrics from 0,
    // Elo from a round number below the lowest visible score (Elo has no zero).
    function makeScale(bench, rows) {
      var values = rows.map(function (r) { return goodness(bench, r.score); });
      if (!values.length) return function () { return 0; };
      var hi = Math.max.apply(null, values);
      var lo = 0;
      if (!isPercent(bench)) {
        var min = Math.min.apply(null, values);
        lo = bench.higher_is_better ? Math.floor((min - 100) / 100) * 100 : min - Math.max((hi - min) * 0.15, 5);
      }
      return function (score) {
        var f = (goodness(bench, score) - lo) / (hi - lo || 1);
        return Math.max(0.015, Math.min(1, f));
      };
    }

    function bestPerModel(bench, rows) {
      var best = {};
      rows.forEach(function (r) {
        var cur = best[r.model_id];
        if (!cur || goodness(bench, r.score) > goodness(bench, cur.score)) best[r.model_id] = r;
      });
      return Object.keys(best).map(function (k) { return best[k]; }).sort(function (a, b) {
        return goodness(bench, b.score) - goodness(bench, a.score);
      });
    }

    function barRow(bench, scale, opts) {
      var row = opts.row;
      var ci = row.ci ? ' ±' + (isPercent(bench) ? row.ci.toFixed(1) : Math.round(row.ci)) : '';
      var head = el('div', { style: 'display:flex;align-items:' + (opts.best !== undefined ? 'center' : 'baseline') + ';gap:8px;margin-bottom:6px;' }, [
        el('span', { style: 'font-size:14px;font-weight:' + (opts.subtitle !== undefined ? '500' : '400') + ';color:var(--mg-heading);' }, [
          opts.title,
          opts.subtitle ? el('span', { text: ' · ' + opts.subtitle, style: 'font-weight:400;color:var(--mg-muted);' }) : null
        ]),
        opts.best ? el('span', { 'class': 'mg-best', text: 'Best' }) : null,
        el('span', { style: 'margin-left:auto;font-family:var(--font-mono);font-size:13px;color:var(--mg-heading);white-space:nowrap;' }, [
          formatScore(bench, row.score),
          ci ? el('span', { text: ci, style: 'color:var(--mg-muted);font-size:11px;' }) : null
        ])
      ]);
      var bar = el('div', { 'class': 'mg-bar', style: 'width:' + (scale(row.score) * 100).toFixed(1) + '%;background:' + tierColor(opts.priceLevel) + ';' });
      var children = [head, el('div', { 'class': 'mg-track', 'aria-hidden': 'true' }, [bar])];
      if (row.note) children.push(el('div', { text: row.note, style: 'margin-top:4px;font-size:12px;color:var(--mg-muted);' }));
      if (opts.onclick) {
        return el('button', { type: 'button', 'class': 'mg-row', title: opts.tooltip || null, onclick: opts.onclick }, children);
      }
      return el('div', { 'class': 'mg-row is-static', title: opts.tooltip || null }, children);
    }

    function renderFilters() {
      filtersEl.textContent = '';
      var all = !state.prices.length;
      filtersEl.appendChild(el('button', {
        type: 'button', 'class': 'mg-chip mg-chip-all', 'aria-pressed': all ? 'true' : 'false', text: 'All',
        onclick: function () { state.prices = []; state.model = null; state.notice = ''; render(); }
      }));
      priceLevels.forEach(function (level) {
        var on = state.prices.indexOf(level) !== -1;
        filtersEl.appendChild(el('button', {
          type: 'button', 'class': 'mg-chip', 'aria-pressed': on ? 'true' : 'false',
          'aria-label': dollars(level) + ' price group',
          onclick: function () {
            state.prices = on
              ? state.prices.filter(function (p) { return p !== level; })
              : state.prices.concat([level]);
            state.model = null;
            state.notice = '';
            render();
          }
        }, [el('span', { 'class': 'mg-swatch', style: 'background:' + tierColor(level) + ';' }), dollars(level)]));
      });
    }

    function renderFooter(bench) {
      footer.textContent = '';
      footer.appendChild(el('h4', {
        text: 'Benchmark: ' + bench.name,
        style: 'margin:0;font-family:var(--font-serif);font-size:16px;font-weight:600;color:var(--mg-heading);'
      }));
      footer.appendChild(el('p', { text: bench.description, style: 'margin:0;' }));
      if (bench.caveat) footer.appendChild(el('p', { text: bench.caveat, style: 'margin:0;' }));
      if (!bench.higher_is_better) footer.appendChild(el('p', { text: 'Lower is better.', style: 'margin:0;' }));
      var host = bench.source_url.replace(/^https?:\/\//, '').split('/')[0];
      var asOf = new Date(bench.as_of + 'T00:00:00').toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' });
      footer.appendChild(el('p', { style: 'margin:0;font-size:12px;' }, [
        'Source: ',
        el('a', { href: bench.source_url, target: '_blank', rel: 'noopener', text: host }),
        ' · ',
        el('span', { text: 'as of ' + asOf, style: 'font-family:var(--font-mono);' })
      ]));
    }

    function renderList(bench) {
      var rows = visibleRows(bench);
      var scale = makeScale(bench, rows);
      var ranked = bestPerModel(bench, rows);
      var shown = state.showAll ? ranked : ranked.slice(0, TOP_N);
      if (state.notice) {
        body.appendChild(el('p', { text: state.notice, style: 'margin:8px 24px 2px;font-size:13px;color:var(--mg-muted);' }));
      }
      if (!ranked.length) {
        body.appendChild(el('p', {
          text: 'No scores yet for models in the selected price groups.',
          style: 'margin:12px 24px;font-size:13px;color:var(--mg-body);'
        }));
      } else {
        var list = el('div', { style: 'padding:4px 0;' });
        shown.forEach(function (row) {
          var m = models[row.model_id];
          list.appendChild(barRow(bench, scale, {
            row: row, title: m.display_name, subtitle: row.effort ? effortLabel(row.effort) : '',
            priceLevel: m.price_level, tooltip: 'Show every reasoning level for ' + m.display_name,
            onclick: function () { state.model = row.model_id; render(); }
          }));
        });
        body.appendChild(list);
        if (ranked.length > TOP_N) {
          body.appendChild(el('div', { style: 'padding:0 24px;' }, [el('button', {
            type: 'button', 'class': 'mg-link',
            text: state.showAll ? 'Show only top ' + TOP_N : 'Show all (' + ranked.length + ')',
            onclick: function () { state.showAll = !state.showAll; render(); }
          })]));
        }
      }
      var missing = bench.missing.filter(inPriceGroups);
      if (missing.length) {
        body.appendChild(el('p', { style: 'margin:6px 24px 14px;font-size:13px;line-height:1.5;color:var(--mg-muted);text-wrap:pretty;' }, [
          el('span', { text: 'Not tested on ' + bench.name + ':', style: 'color:var(--mg-body);font-weight:500;' }),
          ' ' + missing.map(function (id) { return models[id].display_name; }).join(', ')
        ]));
      }
    }

    function renderDetail(bench) {
      var m = models[state.model];
      var rows = bench.rows.filter(function (r) { return r.model_id === state.model; })
        .sort(function (a, b) { return effortRank(a.effort) - effortRank(b.effort); });
      var scale = makeScale(bench, visibleRows(bench).concat(rows));
      var best = bestPerModel(bench, rows)[0];
      var head = el('div', { style: 'padding:4px 24px 6px;' }, [
        el('button', {
          type: 'button', 'class': 'mg-link', onclick: function () { state.model = null; render(); }
        }, [backArrow(), 'All models']),
        el('div', { style: 'display:flex;align-items:baseline;gap:10px;margin-top:8px;' }, [
          el('h3', { text: m.display_name, style: 'margin:0;font-family:var(--font-serif);font-size:18px;font-weight:600;color:var(--mg-heading);' }),
          m.price_level ? el('span', { text: dollars(m.price_level), style: 'font-family:var(--font-mono);font-size:12px;color:var(--mg-muted);' }) : null
        ])
      ]);
      if (rows.length === 1) {
        head.appendChild(el('p', { text: 'One setting only — no reasoning levels to compare.', style: 'margin:4px 0 0;font-size:13px;color:var(--mg-muted);' }));
      }
      body.appendChild(head);
      var list = el('div', { style: 'padding:0 0 8px;' });
      rows.forEach(function (row) {
        list.appendChild(barRow(bench, scale, {
          row: row, title: effortLabel(row.effort), best: rows.length > 1 && row === best,
          priceLevel: m.price_level, tooltip: 'Listed by the source as: ' + row.source_label
        }));
      });
      body.appendChild(list);
    }

    function render() {
      var bench = benchmarks[state.bench];
      titleEl.textContent = bench.title;
      questionEl.textContent = bench.question;
      countEl.textContent = (state.bench + 1) + '/' + benchmarks.length;
      renderFilters();
      body.textContent = '';
      if (state.model) renderDetail(bench);
      else renderList(bench);
      renderFooter(bench);
    }

    function go(index) {
      var n = benchmarks.length;
      state.bench = ((index % n) + n) % n;
      state.notice = '';
      if (state.model) {
        var bench = benchmarks[state.bench];
        var has = bench.rows.some(function (r) { return r.model_id === state.model; });
        if (!has) {
          state.notice = 'No score for ' + models[state.model].display_name + ' on ' + bench.name + '.';
          state.model = null;
        }
      }
      render();
    }

    function open() {
      lastFocus = document.activeElement;
      modal.classList.remove('hidden');
      modal.setAttribute('aria-hidden', 'false');
      render();
      modal.focus();
    }

    function close() {
      modal.classList.add('hidden');
      modal.setAttribute('aria-hidden', 'true');
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }

    trigger.addEventListener('click', open);
    document.getElementById('close-model-guide').addEventListener('click', close);
    document.getElementById('model-guide-prev').addEventListener('click', function () { go(state.bench - 1); });
    document.getElementById('model-guide-next').addEventListener('click', function () { go(state.bench + 1); });
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    modal.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { e.stopPropagation(); close(); }
      else if (e.key === 'ArrowLeft') go(state.bench - 1);
      else if (e.key === 'ArrowRight') go(state.bench + 1);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
