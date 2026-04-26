(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = data.detail || `${r.status} ${r.statusText}`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  // ----- health -----------------------------------------------------------

  async function refreshHealth() {
    const grid = $("#health-grid");
    try {
      const data = await getJSON("/api/health");
      grid.innerHTML = "";
      for (const [name, eng] of Object.entries(data.engines)) {
        const div = document.createElement("div");
        div.className = "engine";
        div.innerHTML = `
          <div class="name">${name}</div>
          <div class="tier">${eng.tier}</div>
          <div class="state ${eng.state}">${eng.state}</div>
          <div class="muted small" style="margin-top:4px">${eng.detail}</div>
        `;
        grid.appendChild(div);
      }
    } catch (e) {
      grid.textContent = `health error: ${e.message}`;
    }
  }

  // ----- tick form --------------------------------------------------------

  $("#form-tick").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const body = {
      symbol: f.symbol.value,
      bid: parseFloat(f.bid.value),
      ask: parseFloat(f.ask.value),
      last: parseFloat(f.last.value),
    };
    try {
      const out = await postJSON("/api/tick", body);
      $("#tick-result").textContent = JSON.stringify(out, null, 2);
      refreshEvents();
    } catch (e) {
      $("#tick-result").textContent = `error: ${e.message}`;
    }
  });

  // ----- signal form ------------------------------------------------------

  $("#form-signal").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const body = {
      symbol: f.symbol.value,
      side: f.side.value,
      confidence: parseFloat(f.confidence.value),
    };
    if (f.qty.value !== "") {
      body.qty = parseFloat(f.qty.value);
    }
    try {
      const out = await postJSON("/api/signal", body);
      $("#signal-result").textContent = JSON.stringify(out, null, 2);
      refreshEvents();
    } catch (e) {
      $("#signal-result").textContent = `error: ${e.message}`;
    }
  });

  // ----- events log -------------------------------------------------------

  async function refreshEvents() {
    const tbody = $("#events-table tbody");
    try {
      const data = await getJSON("/api/events?limit=50");
      tbody.innerHTML = "";
      for (const ev of data.events) {
        const kind = ev.kind || "?";
        const symbol = ev.symbol || "";
        const tr = document.createElement("tr");
        tr.className = `kind-${kind}`;
        const detail = renderDetail(ev);
        tr.innerHTML = `
          <td>${ev.seq}</td>
          <td>${ev.source}</td>
          <td>${kind}</td>
          <td>${symbol}</td>
          <td>${detail}</td>
        `;
        tbody.appendChild(tr);
      }
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="5">events error: ${e.message}</td></tr>`;
    }
  }

  function renderDetail(ev) {
    if (ev.kind === "EXECUTION_EVENT") {
      return `${ev.side} ${ev.qty}@${ev.price} ${ev.status} (${ev.order_id || "-"})`;
    }
    if (ev.kind === "SIGNAL_EVENT") {
      return `${ev.side} conf=${ev.confidence}`;
    }
    if (ev.kind === "MARKET_TICK") {
      return `last=${ev.last} bid=${ev.bid} ask=${ev.ask}`;
    }
    if (ev.kind === "HAZARD_EVENT") {
      return `${ev.severity} ${ev.code}: ${ev.detail || ""}`;
    }
    return JSON.stringify(ev);
  }

  // ----- kick off ---------------------------------------------------------

  refreshHealth();
  refreshEvents();

  $("#refresh-health").addEventListener("click", refreshHealth);

  setInterval(() => {
    if ($("#auto-refresh").checked) {
      refreshEvents();
    }
  }, 2000);
})();
