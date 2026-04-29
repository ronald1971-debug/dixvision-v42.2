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

  // Captured from /api/credentials/status; controls whether the
  // "set key" inline form is rendered (false inside Devin sessions).
  let writable = true;

  // Outcome → label/CSS class for the verify button result badge.
  // Mirrors VerifyOutcome in system_engine/credentials/verifiers.py.
  const VERIFY_OUTCOME = {
    ok: { label: 'ok', cls: 'verify-ok' },
    unauthorized: { label: 'unauthorized', cls: 'verify-bad' },
    rate_limited: { label: 'rate-limited', cls: 'verify-warn' },
    not_found: { label: 'not found', cls: 'verify-bad' },
    server_error: { label: 'server error', cls: 'verify-warn' },
    timeout: { label: 'timeout', cls: 'verify-warn' },
    network_error: { label: 'network', cls: 'verify-warn' },
    no_verifier: { label: 'no verifier', cls: 'verify-skip' },
    missing_key: { label: 'no key set', cls: 'verify-skip' },
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

  function renderEnvVars(envVars, present, sourceId) {
    // Render each env var with a present/missing dot so the operator
    // can see exactly which name is missing on a partial row.
    const safeSid = escapeHtml(sourceId);
    const parts = envVars.map(function (name, idx) {
      const ok = !!present[idx];
      const cls = ok ? 'env-ok' : 'env-miss';
      const dot = ok ? '●' : '○';
      const safeName = escapeHtml(name);
      const setBtn = (!ok && writable)
        ? ' <button class="set-btn" data-source-id="' + safeSid +
          '" data-env-var="' + safeName + '">set…</button>'
        : '';
      return (
        '<span class="env-var-row ' + cls + '">' +
        '<span class="env-dot">' + dot + '</span>' +
        '<code>' + safeName + '</code>' +
        setBtn +
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
    const sid = escapeHtml(item.source_id);
    const verifyCell =
      '<button class="verify-btn" data-source-id="' + sid +
      '">verify</button>' +
      '<span class="verify-result" data-source-id="' + sid +
      '"></span>';
    return (
      '<tr data-state="' + escapeHtml(item.state) +
      '" data-source-id="' + sid + '">' +
      '<td><div class="src-name">' + escapeHtml(item.source_name) +
      '</div><div class="src-id muted">' + sid + '</div></td>' +
      '<td><code>' + escapeHtml(item.category) + '</code></td>' +
      '<td><code>' + escapeHtml(item.provider) + '</code></td>' +
      '<td>' +
      renderEnvVars(
        item.env_vars, item.env_vars_present, item.source_id,
      ) +
      '</td>' +
      '<td><span class="' + stateClass + '">' + stateLabel +
      '</span></td>' +
      '<td>' + freeTier + '</td>' +
      '<td>' + signup + '</td>' +
      '<td>' + verifyCell + '</td>' +
      '<td class="muted">' + escapeHtml(item.notes || '') + '</td>' +
      '</tr>'
    );
  }

  function setVerifyResult(sourceId, outcome, httpStatus, detail) {
    const node = document.querySelector(
      '.verify-result[data-source-id="' + sourceId + '"]',
    );
    if (!node) {
      return;
    }
    const meta = VERIFY_OUTCOME[outcome] ||
      { label: outcome, cls: 'verify-warn' };
    const status = httpStatus ? ' (' + httpStatus + ')' : '';
    node.className = 'verify-result ' + meta.cls;
    node.title = detail || '';
    node.textContent = meta.label + status;
  }

  async function verifyOne(button) {
    const sourceId = button.getAttribute('data-source-id');
    if (!sourceId) {
      return;
    }
    button.disabled = true;
    setVerifyResult(sourceId, 'pending', null, 'verifying…');
    const node = document.querySelector(
      '.verify-result[data-source-id="' + sourceId + '"]',
    );
    if (node) {
      node.className = 'verify-result verify-pending';
      node.textContent = 'verifying…';
    }
    try {
      const resp = await fetch('/api/credentials/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: sourceId }),
      });
      if (!resp.ok) {
        setVerifyResult(
          sourceId,
          'network_error',
          resp.status,
          'HTTP ' + resp.status,
        );
        return;
      }
      const data = await resp.json();
      setVerifyResult(
        sourceId,
        data.outcome,
        data.http_status,
        data.detail,
      );
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      setVerifyResult(
        sourceId,
        'network_error',
        null,
        'fetch failed',
      );
    } finally {
      button.disabled = false;
    }
  }

  function attachVerifyHandlers() {
    document.querySelectorAll('.verify-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        verifyOne(btn);
      });
    });
  }

  function attachSetHandlers() {
    document.querySelectorAll('.set-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        promptAndSet(btn);
      });
    });
  }

  async function promptAndSet(button) {
    const sourceId = button.getAttribute('data-source-id');
    const envVar = button.getAttribute('data-env-var');
    if (!sourceId || !envVar) {
      return;
    }
    // eslint-disable-next-line no-alert
    const value = window.prompt(
      'Set ' + envVar + ' (will be written to .env, gitignored):',
    );
    if (value == null || value === '') {
      return;
    }
    button.disabled = true;
    try {
      const resp = await fetch('/api/credentials/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: sourceId,
          env_var: envVar,
          value: value,
        }),
      });
      if (!resp.ok) {
        let msg = 'HTTP ' + resp.status;
        try {
          const data = await resp.json();
          if (data && data.detail) {
            msg = data.detail;
          }
        } catch (e) {
          /* fall through */
        }
        // eslint-disable-next-line no-alert
        window.alert('Could not set ' + envVar + ': ' + msg);
        return;
      }
      await load();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      // eslint-disable-next-line no-alert
      window.alert('Could not set ' + envVar + ': fetch failed');
    } finally {
      button.disabled = false;
    }
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
        '<tr><td colspan="9" class="cred-error">' +
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
    writable = data.writable !== false;
    renderWritabilityBanner(writable);
    renderSummary(data.summary || { total: 0, present: 0, partial: 0, missing: 0 });
    const tbody = document.getElementById('cred-rows');
    if (!tbody) {
      return;
    }
    if (!data.items || data.items.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="9" class="muted">' +
        'No <code>auth: required</code> rows.' +
        '</td></tr>';
      return;
    }
    tbody.innerHTML = data.items.map(renderRow).join('');
    attachVerifyHandlers();
    attachSetHandlers();
  }

  function renderWritabilityBanner(canWrite) {
    const el = document.getElementById('writability');
    if (!el) {
      return;
    }
    if (canWrite) {
      el.innerHTML = '';
      el.style.display = 'none';
      return;
    }
    el.style.display = '';
    el.innerHTML =
      '<strong>Read-only on this host.</strong> ' +
      'Detected a Devin session, so credentials cannot be written ' +
      'from the dashboard. Add keys via the <code>secrets</code> ' +
      'tool instead; they will appear here once injected into the ' +
      'environment.';
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load);
  } else {
    load();
  }
})();
