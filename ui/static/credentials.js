// DIX VISION — Dashboard-2026 wave-01.5 — credentials matrix.
//
// Reads /api/credentials/status and renders one row per
// `auth: required` registry source. Pure read-only: no secrets are
// ever sent from the browser to the server.

(function () {
  'use strict';

  const STATE_LABEL = {
    present: 'present',
    partial: 'partial',
    missing: 'missing',
  };

  function escapeHtml(s) {
    if (s == null) {
      return '';
    }
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderEnvVars(envVars, present) {
    // Render each env var with a present/missing dot so the operator
    // can see exactly which name is missing on a partial row.
    const parts = envVars.map(function (name, idx) {
      const ok = !!present[idx];
      const cls = ok ? 'env-ok' : 'env-miss';
      const dot = ok ? '●' : '○';
      return (
        '<span class="' + cls + '">' +
        '<span class="env-dot">' + dot + '</span>' +
        '<code>' + escapeHtml(name) + '</code>' +
        '</span>'
      );
    });
    return parts.join(' ');
  }

  function renderRow(item) {
    const stateClass = 'cred-state cred-state-' + item.state;
    const stateLabel = STATE_LABEL[item.state] || item.state;
    const freeTier = item.free_tier
      ? '<span class="badge badge-free">free tier</span>'
      : '<span class="badge badge-paid">paid</span>';
    const signup = item.signup_url
      ? '<a href="' + escapeHtml(item.signup_url) +
        '" target="_blank" rel="noopener noreferrer">sign up</a>'
      : '<span class="muted">no public signup</span>';
    return (
      '<tr data-state="' + escapeHtml(item.state) + '">' +
      '<td><div class="src-name">' + escapeHtml(item.source_name) +
      '</div><div class="src-id muted">' + escapeHtml(item.source_id) +
      '</div></td>' +
      '<td><code>' + escapeHtml(item.category) + '</code></td>' +
      '<td><code>' + escapeHtml(item.provider) + '</code></td>' +
      '<td>' + renderEnvVars(item.env_vars, item.env_vars_present) +
      '</td>' +
      '<td><span class="' + stateClass + '">' + stateLabel +
      '</span></td>' +
      '<td>' + freeTier + '</td>' +
      '<td>' + signup + '</td>' +
      '<td class="muted">' + escapeHtml(item.notes || '') + '</td>' +
      '</tr>'
    );
  }

  function renderSummary(summary) {
    const el = document.getElementById('summary');
    if (!el) {
      return;
    }
    if (summary.total === 0) {
      el.innerHTML =
        '<span class="muted">No <code>auth: required</code>' +
        ' rows in the registry.</span>';
      return;
    }
    el.innerHTML =
      '<span class="cred-state cred-state-present">' +
      summary.present + ' present</span>' +
      '<span class="cred-state cred-state-partial">' +
      summary.partial + ' partial</span>' +
      '<span class="cred-state cred-state-missing">' +
      summary.missing + ' missing</span>' +
      '<span class="muted">of ' + summary.total + ' required.</span>';
  }

  function renderError(message) {
    const tbody = document.getElementById('cred-rows');
    if (tbody) {
      tbody.innerHTML =
        '<tr><td colspan="8" class="cred-error">' +
        escapeHtml(message) + '</td></tr>';
    }
    const summary = document.getElementById('summary');
    if (summary) {
      summary.innerHTML =
        '<span class="cred-error">' + escapeHtml(message) + '</span>';
    }
  }

  async function load() {
    let resp;
    try {
      resp = await fetch('/api/credentials/status');
    } catch (e) {
      renderError('Failed to reach /api/credentials/status.');
      // eslint-disable-next-line no-console
      console.error(e);
      return;
    }
    if (!resp.ok) {
      renderError(
        '/api/credentials/status returned HTTP ' + resp.status + '.',
      );
      return;
    }
    const data = await resp.json();
    renderSummary(data.summary || { total: 0, present: 0, partial: 0, missing: 0 });
    const tbody = document.getElementById('cred-rows');
    if (!tbody) {
      return;
    }
    if (!data.items || data.items.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="8" class="muted">' +
        'No <code>auth: required</code> rows.' +
        '</td></tr>';
      return;
    }
    tbody.innerHTML = data.items.map(renderRow).join('');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load);
  } else {
    load();
  }
})();
