/*
 * "What model should I pick?" — the benchmark guide modal in /chat.
 *
 * Data: #model-guide-data (json_script from llm.model_benchmarks.build_model_guide),
 * already restricted to the organisation's enabled models. One page per benchmark;
 * the list shows each model at its best recorded effort, a model's detail page
 * shows every recorded effort. Paging from a detail page keeps the model.
 */
(function () {
  'use strict';

  var TOP_N = 5;
  var DEFAULT_PRICE_LEVEL = 4;
  var EFFORT_ORDER = ['none', 'off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
  var EFFORT_LABELS = {
    none: 'Reasoning off', off: 'Reasoning off', minimal: 'Minimal', low: 'Low',
    medium: 'Medium', high: 'High', xhigh: 'Extra high', max: 'Max'
  };

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

  function effortLabel(effort) {
    return effort ? (EFFORT_LABELS[effort] || effort) : 'Effort not stated';
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

    var body = document.getElementById('model-guide-body');
    var benchName = document.getElementById('model-guide-bench-name');
    var dots = document.getElementById('model-guide-dots');
    var models = {};
    guide.models.forEach(function (m) { models[m.id] = m; });

    var priceLevels = [];
    guide.models.forEach(function (m) {
      if (m.price_level > 0 && priceLevels.indexOf(m.price_level) === -1) priceLevels.push(m.price_level);
    });
    priceLevels.sort();

    var state = {
      bench: 0,
      model: null,        // detail view when set
      notice: '',         // one-line message above the list
      showAll: false,
      prices: [priceLevels.indexOf(DEFAULT_PRICE_LEVEL) !== -1
        ? DEFAULT_PRICE_LEVEL : priceLevels[priceLevels.length - 1]]
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
      return m && state.prices.indexOf(m.price_level) !== -1;
    }

    function visibleRows(bench) {
      return bench.rows.filter(function (r) { return inPriceGroups(r.model_id); });
    }

    // One scale per page (list + detail share it): percent metrics from 0,
    // Elo from just below the lowest visible score, since Elo has no zero.
    function makeScale(bench, rows) {
      var values = rows.map(function (r) { return goodness(bench, r.score); });
      if (!values.length) return function () { return 0; };
      var hi = Math.max.apply(null, values);
      var lo;
      if (isPercent(bench)) {
        lo = 0;
      } else {
        var min = Math.min.apply(null, values);
        lo = min - Math.max((hi - min) * 0.15, 25);
      }
      return function (score) {
        var f = (goodness(bench, score) - lo) / (hi - lo || 1);
        return Math.max(0.03, Math.min(1, f));
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

    function barRow(bench, scale, title, subtitle, row, onclick, tooltip) {
      var ci = row.ci ? ' ±' + (isPercent(bench) ? row.ci.toFixed(1) : Math.round(row.ci)) : '';
      var head = el('div', { style: 'display:flex;align-items:baseline;justify-content:space-between;gap:12px;' }, [
        el('span', { style: 'min-width:0;font-size:14px;color:var(--color-heading);' }, [
          el('span', { text: title, style: 'font-weight:500;' }),
          subtitle ? el('span', { text: ' · ' + subtitle, style: 'color:var(--color-body-subtle);' }) : null
        ]),
        el('span', { style: 'flex-shrink:0;font-family:var(--font-mono);font-size:13px;color:var(--color-heading);' }, [
          formatScore(bench, row.score),
          ci ? el('span', { text: ci, style: 'color:var(--color-fg-disabled);font-size:11px;' }) : null
        ])
      ]);
      var bar = el('div', { 'class': 'mg-bar', style: 'width:' + (scale(row.score) * 100).toFixed(1) + '%;' });
      var children = [head, el('div', { 'class': 'mg-track', 'aria-hidden': 'true' }, [bar])];
      if (row.note) children.push(el('div', { text: row.note, style: 'margin-top:4px;font-size:12px;color:var(--color-body-subtle);' }));
      var attrs = { 'class': 'mg-row' + (onclick ? '' : ' is-static'), title: tooltip || null };
      if (onclick) {
        attrs.type = 'button';
        attrs.onclick = onclick;
        return el('button', attrs, children);
      }
      return el('div', attrs, children);
    }

    function priceChips() {
      var wrap = el('div', { role: 'group', 'aria-label': 'Price groups', style: 'display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:12px;' }, [
        el('span', { text: 'Price', style: 'font-size:12px;color:var(--color-body-subtle);margin-right:2px;' })
      ]);
      priceLevels.forEach(function (level) {
        var on = state.prices.indexOf(level) !== -1;
        wrap.appendChild(el('button', {
          type: 'button', 'class': 'mg-chip', 'aria-pressed': on ? 'true' : 'false',
          text: new Array(level + 1).join('$'),
          onclick: function () {
            if (on) state.prices = state.prices.filter(function (p) { return p !== level; });
            else state.prices = state.prices.concat([level]);
            state.notice = '';
            render();
          }
        }));
      });
      return wrap;
    }

    function explanation(bench, missingIds, extra) {
      var parts = [el('p', { text: bench.description, style: 'margin:0;' })];
      if (bench.caveat) parts.push(el('p', { text: bench.caveat, style: 'margin:8px 0 0;' }));
      if (extra) parts.push(el('p', { text: extra, style: 'margin:8px 0 0;' }));
      if (missingIds && missingIds.length) {
        parts.push(el('p', {
          text: 'Not yet scored: ' + missingIds.map(function (id) { return models[id].display_name; }).join(', ') + '.',
          style: 'margin:8px 0 0;'
        }));
      }
      var host = bench.source_url.replace(/^https?:\/\//, '').split('/')[0];
      var asOf = new Date(bench.as_of + 'T00:00:00').toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
      parts.push(el('p', { style: 'margin:8px 0 0;' }, [
        'Source: ',
        el('a', { href: bench.source_url, target: '_blank', rel: 'noopener', 'class': 'mg-link', text: host }),
        ' · as of ' + asOf
      ]));
      return el('div', {
        style: 'margin-top:16px;padding-top:14px;border-top:1px solid var(--color-default-subtle);font-size:12.5px;line-height:1.55;color:var(--color-body);'
      }, parts);
    }

    function renderList(bench) {
      var rows = visibleRows(bench);
      var scale = makeScale(bench, rows);
      var ranked = bestPerModel(bench, rows);
      var shown = state.showAll ? ranked : ranked.slice(0, TOP_N);
      body.appendChild(priceChips());
      if (state.notice) {
        body.appendChild(el('p', { text: state.notice, style: 'margin:0 0 10px;font-size:12.5px;color:var(--color-body-subtle);' }));
      }
      if (!ranked.length) {
        body.appendChild(el('p', {
          text: state.prices.length ? 'No scores yet for models in the selected price groups.' : 'Select a price group to see models.',
          style: 'margin:8px 0;font-size:13px;color:var(--color-body);'
        }));
      } else {
        var list = el('div', { style: 'display:flex;flex-direction:column;gap:2px;margin:0 -10px;' });
        shown.forEach(function (row, i) {
          var m = models[row.model_id];
          var node = barRow(bench, scale, m.display_name, row.effort ? effortLabel(row.effort) : '', row, function () {
            state.model = row.model_id;
            render();
          }, 'Show every reasoning level for ' + m.display_name);
          if (i === 0) node.classList.add('is-top');
          list.appendChild(node);
        });
        body.appendChild(list);
        if (ranked.length > TOP_N) {
          body.appendChild(el('button', {
            type: 'button', 'class': 'mg-link', style: 'margin-top:8px;font-size:13px;',
            text: state.showAll ? 'Show top ' + TOP_N : 'Show all (' + ranked.length + ')',
            onclick: function () { state.showAll = !state.showAll; render(); }
          }));
        }
      }
      var missing = bench.missing.filter(inPriceGroups);
      body.appendChild(explanation(bench, missing,
        (bench.higher_is_better ? 'Higher is better. ' : 'Lower is better. ') +
        'Each model is shown at its best recorded reasoning level; select a model to see every level.'));
    }

    function renderDetail(bench) {
      var m = models[state.model];
      var rows = bench.rows.filter(function (r) { return r.model_id === state.model; })
        .sort(function (a, b) { return effortRank(a.effort) - effortRank(b.effort); });
      var scale = makeScale(bench, visibleRows(bench).concat(rows));
      body.appendChild(el('button', {
        type: 'button', 'class': 'mg-link', style: 'font-size:13px;margin-bottom:10px;', text: '← All models',
        onclick: function () { state.model = null; render(); }
      }));
      body.appendChild(el('div', { style: 'display:flex;align-items:baseline;gap:8px;margin-bottom:8px;' }, [
        el('h3', { text: m.display_name, style: 'margin:0;font-size:16px;font-weight:600;color:var(--color-heading);' }),
        m.price_level ? el('span', { text: new Array(m.price_level + 1).join('$'), style: 'font-family:var(--font-mono);font-size:12px;color:var(--color-body-subtle);' }) : null
      ]));
      var list = el('div', { style: 'display:flex;flex-direction:column;gap:2px;margin:0 -10px;' });
      rows.forEach(function (row) {
        list.appendChild(barRow(bench, scale, effortLabel(row.effort), '', row, null, 'Listed by the source as: ' + row.source_label));
      });
      body.appendChild(list);
      body.appendChild(explanation(bench, null,
        (bench.higher_is_better ? 'Higher is better. ' : 'Lower is better. ') +
        'Only the reasoning levels the source has tested are shown.'));
    }

    function render() {
      var bench = benchmarks[state.bench];
      benchName.textContent = bench.name + ' — ' + bench.metric;
      dots.textContent = '';
      benchmarks.forEach(function (b, i) {
        dots.appendChild(el('button', {
          type: 'button', 'class': 'mg-dot' + (i === state.bench ? ' is-active' : ''),
          'aria-label': b.name, 'aria-current': i === state.bench ? 'true' : null,
          onclick: function () { go(i); }
        }));
      });
      body.textContent = '';
      body.scrollTop = 0;
      if (state.model) renderDetail(bench);
      else renderList(bench);
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
