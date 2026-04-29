/* Shared chat widget runtime — Dashboard-2026 wave-01.
 *
 * One JS module is reused by both Indira Chat and Dyon Chat. The
 * widget is INTENTIONALLY registry-driven: it only knows about a
 * task class (passed in from the page) and a `/api/ai/providers`
 * endpoint. The widget never names an AI vendor.
 *
 * Authority lint rule B23 enforces this at the source level — this
 * file must contain no string literal naming any specific AI vendor.
 * The provider list is whatever the SCVS registry decides, full stop.
 *
 * Wave-01 scope: dropdown + transcript + send button. Streaming
 * responses, tool-call rendering, and Governance routing land in
 * wave-02 (React port).
 */

'use strict';

const PROVIDER_LIST_ENDPOINT = '/api/ai/providers';

async function fetchProviders(taskClass) {
  const url = taskClass
    ? `${PROVIDER_LIST_ENDPOINT}?task=${encodeURIComponent(taskClass)}`
    : PROVIDER_LIST_ENDPOINT;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`provider fetch failed: HTTP ${res.status}`);
  }
  return res.json();
}

function setProviderOptions(selectEl, providers) {
  selectEl.innerHTML = '';
  if (providers.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '— no enabled providers in registry —';
    opt.disabled = true;
    selectEl.appendChild(opt);
    selectEl.disabled = true;
    return;
  }
  selectEl.disabled = false;
  // First option is the auto / fallback chain (registry order).
  const auto = document.createElement('option');
  auto.value = '';
  auto.textContent = '(registry order — fallback chain)';
  selectEl.appendChild(auto);
  for (const p of providers) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.name} — ${p.capabilities.join(', ')}`;
    selectEl.appendChild(opt);
  }
}

function appendChatRow(logEl, who, body, meta) {
  const row = document.createElement('div');
  row.className = 'chat-row';
  const whoEl = document.createElement('div');
  whoEl.className = 'who';
  whoEl.textContent = who;
  row.appendChild(whoEl);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    row.appendChild(m);
  }
  const bodyEl = document.createElement('div');
  bodyEl.className = 'body';
  bodyEl.textContent = body;
  row.appendChild(bodyEl);
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
}

// Per-widget abort controllers so that re-initialising a widget
// (e.g. when the operator switches the task class) cleanly tears
// down the previous init's listeners. Keyed by send-button element
// because the page reuses the same DOM nodes across boots.
const _widgetAbortControllers = new WeakMap();

/**
 * Initialise a chat widget instance.
 *
 * Safe to call multiple times against the same `cfg.sendBtnEl`:
 * each call aborts the previous init's listeners before adding new
 * ones. This matters because both indira_chat.html and
 * dyon_chat.html re-call this on every task-class dropdown change;
 * without the abort, click handlers would accumulate and the
 * oldest (stale `taskClass`) one would fire first.
 *
 * @param {object} cfg
 * @param {string} cfg.taskClass         TaskClass value (e.g.
 *                                        'indira_reasoning').
 * @param {string} cfg.widgetLabel       Human label for the
 *                                        transcript ('You' is fine
 *                                        for the operator side).
 * @param {HTMLSelectElement} cfg.selectEl
 * @param {HTMLTextAreaElement} cfg.inputEl
 * @param {HTMLButtonElement} cfg.sendBtnEl
 * @param {HTMLElement} cfg.logEl
 * @param {HTMLElement} cfg.statusEl
 */
async function initChatWidget(cfg) {
  const previous = _widgetAbortControllers.get(cfg.sendBtnEl);
  if (previous) {
    previous.abort();
  }
  const controller = new AbortController();
  _widgetAbortControllers.set(cfg.sendBtnEl, controller);

  cfg.statusEl.textContent = 'loading providers…';
  try {
    const data = await fetchProviders(cfg.taskClass);
    setProviderOptions(cfg.selectEl, data.providers);
    if (data.providers.length === 0) {
      cfg.statusEl.textContent =
        'No AI providers enabled in the SCVS registry. Enable at least'
        + ' one row (category: ai) to use this chat.';
      cfg.sendBtnEl.disabled = true;
    } else {
      cfg.statusEl.textContent =
        `${data.providers.length} provider(s) eligible for task class`
        + ` "${data.task ?? '(any)'}".`;
    }
  } catch (err) {
    cfg.statusEl.textContent =
      'Failed to load providers from /api/ai/providers — see console.';
    // eslint-disable-next-line no-console
    console.error(err);
    cfg.sendBtnEl.disabled = true;
  }

  cfg.sendBtnEl.addEventListener(
    'click',
    () => {
      const text = cfg.inputEl.value.trim();
      if (!text) {
        return;
      }
      const pinnedId = cfg.selectEl.value || null;
      const meta = pinnedId
        ? `task=${cfg.taskClass} · pinned=${pinnedId}`
        : `task=${cfg.taskClass} · auto (registry order)`;
      appendChatRow(cfg.logEl, 'You', text, meta);
      cfg.inputEl.value = '';
      // Wave-01 stub: turn dispatch lands in wave-02 with streaming +
      // governance routing. For now we echo the routing decision so
      // the operator can verify the dropdown actually drives behaviour.
      appendChatRow(
        cfg.logEl,
        cfg.widgetLabel,
        '[wave-01 skeleton] turn dispatch is stubbed. The router would'
          + ' send this to: '
          + (pinnedId
            ? `the pinned provider (${pinnedId}).`
            : 'every enabled provider in registry order, with fallback.'),
        null,
      );
    },
    { signal: controller.signal },
  );
}

window.DIXChatWidget = { initChatWidget };
