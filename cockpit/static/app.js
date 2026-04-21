// cockpit/static/app.js — minimal SPA, no build step, no frameworks.
// Responsive: same codebase drives desktop split-pane + mobile tab view.
(() => {
  const qs = new URLSearchParams(location.search);
  const TOKEN = qs.get("token") || localStorage.getItem("dix_token") || "";
  if (TOKEN) localStorage.setItem("dix_token", TOKEN);
  const H = { "Authorization": "Bearer " + TOKEN, "Content-Type": "application/json" };

  let I18N = { en: {} }, LANG = "en";
  async function loadI18N() {
    try { I18N = await fetch("/static/i18n.json").then(r => r.json()); }
    catch { I18N = { en: {} }; }
  }
  function t(key, fallback) {
    return (I18N[LANG] && I18N[LANG][key]) || (I18N.en && I18N.en[key]) || fallback || key;
  }
  function applyI18N() {
    document.querySelectorAll("[data-i18n]").forEach(el => {
      const k = el.getAttribute("data-i18n");
      el.textContent = t(k, el.textContent);
    });
  }

  async function j(url, opts) {
    const r = await fetch(url, { headers: H, ...opts });
    if (!r.ok) throw new Error(r.status + " " + url);
    return r.json();
  }

  async function loadStatus() {
    const s = await j("/api/status");
    document.getElementById("status-pill").textContent =
      "v" + s.version + " · charters=" + s.charters + " · ai=" + s.ai_providers;
  }
  async function loadLocale() {
    const l = await j("/api/locale");
    const sel = document.getElementById("locale-select");
    sel.innerHTML = "";
    l.supported_ui.forEach(lang => {
      const o = document.createElement("option");
      o.value = lang; o.textContent = lang.toUpperCase();
      if (lang === l.current.language) o.selected = true;
      sel.appendChild(o);
    });
    LANG = l.current.language || "en";
    document.documentElement.setAttribute("data-lang", LANG);
    sel.onchange = async () => {
      await fetch("/api/locale", { method: "POST", headers: H,
        body: JSON.stringify({ tag: sel.value }) });
      LANG = sel.value;
      document.documentElement.setAttribute("data-lang", LANG);
      applyI18N();
    };
    applyI18N();
  }
  async function loadCharters() {
    const { charters } = await j("/api/charters");
    const d = document.getElementById("charters");
    d.innerHTML = "";
    charters.forEach(c => {
      const div = document.createElement("div");
      div.className = "mono";
      div.style.marginBottom = "10px";
      div.innerHTML =
        "<b style='color:var(--accent)'>" + c.voice + "</b> " +
        "<span class='pill'>domain=" + c.domain + "</span><br/>" +
        "<div>" + c.what + "</div>" +
        "<div style='color:var(--mut); font-size:12px'>tools: " + c.tools.join(", ") + "</div>";
      d.appendChild(div);
    });
  }
  async function loadAI() {
    const { providers } = await j("/api/ai");
    const tb = document.querySelector("#ai-tbl tbody");
    tb.innerHTML = "";
    providers.forEach(p => {
      const tr = document.createElement("tr");
      const status = p.enabled ? "<span class='pill ok'>enabled</span>"
                                : (p.has_key ? "<span class='pill warn'>off</span>"
                                             : "<span class='pill err'>no key</span>");
      tr.innerHTML = "<td class='mono'>" + p.name + "</td>" +
                     "<td>" + p.role + "</td>" +
                     "<td>" + status + "</td>" +
                     "<td class='mono'>$" + p.cost_per_1k_tokens_usd.toFixed(4) + "</td>";
      tb.appendChild(tr);
    });
  }
  async function loadProviders() {
    const { providers } = await j("/api/providers");
    const tb = document.querySelector("#src-tbl tbody");
    tb.innerHTML = "";
    providers.forEach(p => {
      const st = p.enabled ? "<span class='pill ok'>on</span>"
                           : (p.has_key ? "<span class='pill warn'>disabled</span>"
                                        : "<span class='pill err'>no key</span>");
      tb.innerHTML += "<tr><td class='mono'>" + p.name + "</td>" +
                      "<td>" + p.kind + "</td>" +
                      "<td>" + st + "</td></tr>";
    });
  }
  async function loadRisk() {
    const r = await j("/api/risk");
    document.getElementById("risk").textContent = JSON.stringify(r, null, 2);
  }
  async function loadTraderCount() {
    const r = await j("/api/traders/count");
    document.getElementById("trader-count").textContent =
      "traders:" + r.count.traders + " strategies:" + r.count.strategies +
      " statements:" + r.count.statements;
  }
  async function searchTraders(q) {
    const r = await j("/api/traders/search?q=" + encodeURIComponent(q));
    const tb = document.querySelector("#trader-tbl tbody");
    tb.innerHTML = "";
    r.results.forEach(t => {
      tb.innerHTML += "<tr><td>" + t.name + "</td>" +
                      "<td>" + (t.era || "") + "</td>" +
                      "<td>" + (t.region || "") + "</td>" +
                      "<td>" + (t.style_tags || []).join(", ") + "</td></tr>";
    });
  }

  function kvRender(el, obj) {
    el.innerHTML = "";
    Object.keys(obj).forEach(k => {
      const kd = document.createElement("div"); kd.className = "k"; kd.textContent = k;
      const vd = document.createElement("div"); vd.className = "mono"; vd.textContent =
        (typeof obj[k] === "object" ? JSON.stringify(obj[k]) : String(obj[k]));
      el.appendChild(kd); el.appendChild(vd);
    });
  }

  async function loadWallet() {
    const p = await j("/api/wallet/policy");
    kvRender(document.getElementById("wallet-policy"), {
      phase: p.phase,
      day_index: p.day_index,
      warmup_days_remaining: p.warmup_days_remaining,
      supervised_days_remaining: p.supervised_days_remaining,
      system_cap_usd: p.system_cap_usd,
      per_wallet_cap_usd: p.per_wallet_cap_usd,
      spent_system_24h_usd: p.spent_system_24h_usd,
      live_signing_allowed: p.live_signing_allowed,
    });
    const bar = document.getElementById("phase-bar");
    const pct = p.phase === "WARMUP"
                ? Math.min(100, (p.day_index / 30) * 100)
                : p.phase === "SUPERVISED"
                ? Math.min(100, ((p.day_index - 30) / 30) * 100)
                : 100;
    bar.style.width = pct.toFixed(1) + "%";
    bar.parentElement.className = "bar" + (p.phase === "WARMUP" ? " warn" : "");

    const w = await j("/api/wallets");
    const tb = document.querySelector("#wallets-tbl tbody");
    tb.innerHTML = "";
    w.wallets.forEach(x => {
      const live = x.live_signing_allowed
        ? "<span class='pill ok'>yes</span>"
        : "<span class='pill warn'>no</span>";
      tb.innerHTML += "<tr><td>" + x.label + "</td>" +
                      "<td class='mono'>" + x.chain + "</td>" +
                      "<td>" + x.backend + "</td>" +
                      "<td class='mono'>" + x.address_masked + "</td>" +
                      "<td>" + live + "</td></tr>";
    });
  }

  async function loadStrategies() {
    const { strategies } = await j("/api/strategies");
    const tb = document.querySelector("#strategies-tbl tbody");
    tb.innerHTML = "";
    Object.keys(strategies).forEach(name => {
      const s = strategies[name];
      const pill = !s.active
        ? "<span class='pill err'>off</span>"
        : (s.shadow ? "<span class='pill warn'>shadow</span>"
                    : "<span class='pill ok'>active</span>");
      tb.innerHTML += "<tr><td class='mono'>" + name + "</td>" +
                      "<td>" + (s.reward_n || 0) + "</td>" +
                      "<td>" + (s.reward_mean || 0).toFixed(4) + "</td>" +
                      "<td>" + pill + "</td></tr>";
    });
  }

  async function loadSafety() {
    const s = await j("/api/safety");
    kvRender(document.getElementById("safety"), {
      dead_man_last_beat: s.dead_man.last_beat_utc || "(none)",
      dead_man_age_sec: s.dead_man.age_sec,
      dead_man_timeout_sec: s.dead_man.timeout_sec,
      dead_man_tripped: s.dead_man.tripped,
      latency_n: s.latency_guard.n,
      latency_p50_us: s.latency_guard.p50_us,
      latency_p95_us: s.latency_guard.p95_us,
      latency_p99_us: s.latency_guard.p99_us,
      latency_budget_us: s.latency_guard.budget_us,
      latency_tripped: s.latency_guard.tripped,
    });
  }

  async function loadOverview() {
    const [st, p, safety, traders] = await Promise.all([
      j("/api/status"), j("/api/wallet/policy"),
      j("/api/safety"), j("/api/traders/count"),
    ]);
    kvRender(document.getElementById("overview-kv"), {
      version: st.version,
      locale: (st.locale.tag || "en") + " (" + (st.locale.source || "?") + ")",
      charters: st.charters,
      ai_providers: st.ai_providers,
      wallet_phase: p.phase,
      wallet_live_ok: p.live_signing_allowed,
      dead_man_tripped: safety.dead_man.tripped,
      latency_p99_us: safety.latency_guard.p99_us,
      traders_kb: traders.count.traders,
    });
  }

  function chatAppend(who, text, cls) {
    const log = document.getElementById("chat-log");
    const d = document.createElement("div");
    d.className = "turn" + (cls ? " " + cls : "");
    const w = document.createElement("span"); w.className = "who"; w.textContent = who + ":";
    const b = document.createElement("span"); b.textContent = " " + text;
    d.appendChild(w); d.appendChild(b);
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
  }
  async function sendChat(msg, voice) {
    chatAppend("you", msg, "user");
    try {
      const r = await j("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: msg, voice: voice || null }),
      });
      chatAppend(r.voice + (r.model && r.model !== "template" ? "("+r.model+")" : ""),
                 r.answer);
    } catch (e) {
      chatAppend("ERROR", String(e), "err");
    }
  }

  function setupTabs() {
    const tabs = document.querySelectorAll("#tabs button");
    const panels = document.querySelectorAll(".tab-panel");
    tabs.forEach(b => {
      b.onclick = () => {
        tabs.forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        const name = b.dataset.tab;
        panels.forEach(p => {
          if (p.dataset.panel === name) p.classList.add("active");
          else p.classList.remove("active");
        });
      };
    });
  }

  // --- PWA install prompt (Android/desktop Chromium) ---
  let deferredInstall = null;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredInstall = e;
    if (localStorage.getItem("dix_install_dismissed") !== "1") {
      document.getElementById("install-banner").classList.add("show");
    }
  });
  document.getElementById("install-btn").onclick = async () => {
    if (!deferredInstall) return;
    deferredInstall.prompt();
    await deferredInstall.userChoice;
    deferredInstall = null;
    document.getElementById("install-banner").classList.remove("show");
  };
  document.getElementById("install-close").onclick = () => {
    localStorage.setItem("dix_install_dismissed", "1");
    document.getElementById("install-banner").classList.remove("show");
  };

  // --- dead-man heartbeat: ping every 30s while the tab is open ---
  async function heartbeat() {
    try { await fetch("/api/safety/heartbeat", { method: "POST", headers: H }); }
    catch (e) { /* offline ok */ }
  }
  setInterval(heartbeat, 30000);

  async function boot() {
    await loadI18N();
    setupTabs();
    await Promise.all([
      loadStatus(), loadLocale(), loadCharters(),
      loadAI(), loadProviders(), loadRisk(),
      loadTraderCount(), loadWallet(), loadStrategies(),
      loadSafety(), loadOverview(),
    ]);
    document.getElementById("trader-form").onsubmit = e => {
      e.preventDefault();
      searchTraders(document.getElementById("q").value);
    };
    document.getElementById("chat-form").onsubmit = e => {
      e.preventDefault();
      const m = document.getElementById("msg").value.trim();
      if (!m) return;
      sendChat(m, document.getElementById("voice").value);
      document.getElementById("msg").value = "";
    };
    document.getElementById("hb-btn").onclick = async () => {
      await heartbeat();
      await loadSafety();
    };
    document.getElementById("status-pill").onclick = heartbeat;
    chatAppend("system", "Ready. Ask 'what is your role?' or 'why did X happen?'.");
    // periodic refresh of the fast-moving panels
    setInterval(() => { loadSafety().catch(()=>{}); loadWallet().catch(()=>{}); }, 15000);
  }
  boot();
})();
