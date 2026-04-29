/**
 * DASH-2 — operator dashboard client.
 *
 * Renders the five Phase 6 widgets (DASH-02 / DASH-EG-01 /
 * DASH-SLP-01 / DASH-04 / DASH-MCP-01) by polling
 * /api/dashboard/summary and posts operator actions to the
 * corresponding /api/dashboard/action/* endpoint, where the request
 * is routed through ControlPlaneRouter (DASH-CP-01) to the
 * OperatorInterfaceBridge (GOV-CP-07). The dashboard never writes
 * the ledger or constructs governance decisions itself.
 */

(() => {
  "use strict";

  const REFRESH_MS = 2000;
  const $ = (sel) => document.querySelector(sel);
  const lastRefresh = $("#last-refresh");
  const autoRefresh = $("#auto-refresh");
  const btnRefresh = $("#btn-refresh");
  const actionLog = $("#action-log");

  let timer = null;

  // -------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------

  function renderMode(state) {
    $("#mode-current").textContent = state.current_mode || "—";
    $("#mode-targets").textContent = (state.legal_targets || []).join(", ") || "—";
    $("#mode-locked").classList.toggle("hidden", !state.is_locked);
  }

  function renderEngines(rows) {
    const tbody = $("#engines-body");
    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="muted">no engines</td></tr>';
      return;
    }
    tbody.innerHTML = rows
      .map((r) => {
        const bucket = (r.bucket || "offline").toLowerCase();
        const plugins = (r.plugin_states || [])
          .map((p) => `${p[0]}:${p[1]}=${p[2]}`)
          .join(", ");
        return `<tr>
          <td>${escapeHtml(r.engine_name)}</td>
          <td class="bucket bucket-${bucket}">${bucket}</td>
          <td>${escapeHtml(r.detail ?? "")}</td>
          <td>${escapeHtml(plugins) || '<span class="muted">—</span>'}</td>
        </tr>`;
      })
      .join("");
  }

  function renderStrategies(columns) {
    const root = $("#strategies-cols");
    const order = ["PROPOSED", "SHADOW", "CANARY", "LIVE", "RETIRED", "FAILED"];
    root.innerHTML = order
      .map((state) => {
        const rows = (columns && columns[state]) || [];
        const inner =
          rows.length === 0
            ? '<div class="muted">—</div>'
            : rows
                .map(
                  (s) =>
                    `<div class="strategy-row">${escapeHtml(s.strategy_id || "?")}</div>`,
                )
                .join("");
        return `<div class="strategy-col">
          <h3>${state}</h3>
          ${inner}
        </div>`;
      })
      .join("");
  }

  function renderDecisions(chains) {
    const root = $("#decisions-list");
    if (!chains || chains.length === 0) {
      root.innerHTML = '<div class="muted">no traced decisions yet</div>';
      return;
    }
    root.innerHTML = chains
      .map((chain) => {
        const events = (chain.events || [])
          .map((e) => escapeHtml(e.kind || e.type || "?"))
          .join(" → ");
        return `<div class="trace-row">
          <div class="trace-symbol">${escapeHtml(chain.symbol || "?")}</div>
          <div class="trace-events">${events || '<span class="muted">(empty)</span>'}</div>
        </div>`;
      })
      .join("");
  }

  function renderMemecoin(state) {
    $("#memecoin-enabled").textContent = state.enabled ? "yes" : "no";
    $("#memecoin-killed").textContent = state.killed ? "yes" : "no";
    $("#memecoin-summary").textContent = state.summary || "";
  }

  // -------------------------------------------------------------------
  // Refresh
  // -------------------------------------------------------------------

  async function refresh() {
    try {
      const r = await fetch("/api/dashboard/summary");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const body = await r.json();
      renderMode(body.mode || {});
      renderEngines(body.engines || []);
      renderStrategies(body.strategies || {});
      renderDecisions(body.chains || []);
      renderMemecoin(body.memecoin || {});
      lastRefresh.textContent = "refreshed " + new Date().toLocaleTimeString();
    } catch (err) {
      lastRefresh.textContent = "refresh failed: " + err.message;
    }
  }

  function startAuto() {
    stopAuto();
    timer = setInterval(refresh, REFRESH_MS);
  }

  function stopAuto() {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  // -------------------------------------------------------------------
  // Action submission
  // -------------------------------------------------------------------

  function logAction(label, body) {
    const row = document.createElement("div");
    const cls = body && body.approved ? "ok" : "err";
    row.className = "log-row " + cls;
    const summary = body && body.summary ? body.summary : "(no summary)";
    const ts = new Date().toLocaleTimeString();
    row.textContent = `[${ts}] ${label} → ${summary}`;
    if (actionLog.firstChild && actionLog.firstChild.classList?.contains("muted")) {
      actionLog.innerHTML = "";
    }
    actionLog.insertBefore(row, actionLog.firstChild);
  }

  async function postAction(label, url, payload) {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      logAction(label, body);
      // Pull a fresh snapshot so the operator sees the effect immediately.
      refresh();
    } catch (err) {
      logAction(label, { approved: false, summary: "client error: " + err.message });
    }
  }

  function bindForm(formId, label, url, mapper) {
    const form = document.getElementById(formId);
    if (!form) return;
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const data = new FormData(form);
      const payload = mapper(data);
      postAction(label, url, payload);
    });
  }

  bindForm("form-mode", "REQUEST_MODE", "/api/dashboard/action/mode", (d) => ({
    target_mode: d.get("target_mode") || "",
    reason: d.get("reason") || "",
    operator_authorized: d.get("operator_authorized") === "on",
  }));

  bindForm("form-kill", "REQUEST_KILL", "/api/dashboard/action/kill", (d) => ({
    reason: d.get("reason") || "operator kill",
  }));

  bindForm(
    "form-lifecycle",
    "REQUEST_PLUGIN_LIFECYCLE",
    "/api/dashboard/action/lifecycle",
    (d) => ({
      plugin_path: d.get("plugin_path") || "",
      target_status: d.get("target_status") || "",
      reason: d.get("reason") || "",
    }),
  );

  bindForm("form-intent", "REQUEST_INTENT", "/api/dashboard/action/intent", (d) => {
    const focusRaw = (d.get("focus") || "").toString().trim();
    const focus = focusRaw === "" ? [] : focusRaw.split(",").map((s) => s.trim()).filter(Boolean);
    return {
      objective: d.get("objective") || "",
      risk_mode: d.get("risk_mode") || "",
      horizon: d.get("horizon") || "",
      focus,
      reason: d.get("reason") || "",
    };
  });

  // -------------------------------------------------------------------
  // Bootstrap
  // -------------------------------------------------------------------

  btnRefresh.addEventListener("click", refresh);
  autoRefresh.addEventListener("change", () => {
    if (autoRefresh.checked) startAuto();
    else stopAuto();
  });

  refresh();
  if (autoRefresh.checked) startAuto();

  // -------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
})();
