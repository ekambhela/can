"use strict";

const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropzone = $("dropzone");
const fileMeta = $("fileMeta");
const submitBtn = $("submitBtn");
const form = $("uploadForm");

let selectedFile = null;
let mode = "single";          // "single" | "manual" | "batch"
let lastBatch = null;         // cached cohort result for CSV export
let currentSample = null;     // parsed sample behind the current single result (for sharing)
let FEATURE_LABELS = {};      // feature key -> friendly label, filled from /api/schema

/* ---- live model stats (from /api/health) ---- */
async function loadStats() {
  try {
    const d = await (await fetch("/api/health")).json();
    const m = d.metrics || {};
    const set = (id, val) => { const el = $(id); if (el && val != null) el.textContent = val; };
    set("statTop1", m.top1_accuracy != null ? Math.round(m.top1_accuracy * 100) + "%" : null);
    set("statTop3", m.top10_accuracy != null ? Math.round(m.top10_accuracy * 100) + "%" : null);
    set("statSpear", m.mean_spearman != null ? m.mean_spearman.toFixed(2) : null);
    set("nTherapies", m.n_drugs || null);
    set("nCellLines", m.n_cell_lines || null);
  } catch (_) { /* non-fatal */ }
}
loadStats();

/* ---- tab navigation ---- */
const TABS = ["home", "problem", "how", "science", "matcher", "about"];
function showTab(name) {
  if (!TABS.includes(name)) name = "home";
  document.querySelectorAll(".tab-section").forEach((s) => { s.hidden = s.dataset.pane !== name; });
  document.querySelectorAll(".navlink").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
  if (name === "home") animateCounters();
  if (history.replaceState) history.replaceState(null, "", "#" + name);
  window.scrollTo(0, 0);
  // recompute parallax for the newly shown pane (its bg was display:none)
  window.dispatchEvent(new Event("scroll"));
}
document.querySelectorAll("[data-tab]").forEach((el) => {
  el.addEventListener("click", (e) => { e.preventDefault(); showTab(el.dataset.tab); });
});
window.addEventListener("hashchange", () => {
  showTab((location.hash || "").replace("#", "") || "home");
});

/* ---- animated counters (home) ---- */
let countersDone = false;
function animateCounters() {
  if (countersDone) return;
  countersDone = true;
  document.querySelectorAll(".count").forEach((el) => {
    const to = parseFloat(el.dataset.to);
    const dec = parseInt(el.dataset.dec || "0", 10);
    const suf = el.dataset.suffix || "";
    const t0 = performance.now(), dur = 1100;
    (function step(t) {
      const p = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = (to * e).toFixed(dec) + suf;
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  });
}

/* ---- initial route ---- */
(function initRoute() {
  if (new URLSearchParams(location.search).get("case")) return;  // handled after schema loads
  showTab((location.hash || "").replace("#", "") || "home");
})();

/* ---- scroll-reveal ---- */
(function setupReveal() {
  const sel = ".feature, .flow-step, .stat-cell, .pstep, .rule, .mcard, .drug-col, " +
              ".vision-card, .match-figure, .band.mission, .pipeline, .preview, " +
              ".prob-card, .solution, .section-title, .about-h, .page-head h1, .hl";
  const els = document.querySelectorAll(sel);
  if (!("IntersectionObserver" in window)) {
    els.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
  els.forEach((el) => io.observe(el));
})();

/* ---- scroll-driven effects: progress bar, side spine, parallax ---- */
(function scrollFX() {
  const prog = document.getElementById("scrollProgress");
  const dot = document.getElementById("spineDot");
  const par = Array.from(document.querySelectorAll(".parallax"));
  const vh = () => window.innerHeight;
  let ticking = false;
  function update() {
    ticking = false;
    const st = window.scrollY || 0;
    const max = (document.documentElement.scrollHeight - vh()) || 1;
    const frac = Math.min(1, Math.max(0, st / max));
    if (prog) prog.style.transform = "scaleX(" + frac + ")";
    if (dot) dot.style.top = (frac * 100) + "%";
    for (const el of par) {
      const d = parseFloat(el.dataset.depth || "0.05");
      const r = el.getBoundingClientRect();
      const fromCenter = r.top + r.height / 2 - vh() / 2;
      el.style.transform = "translate3d(0," + (-fromCenter * d).toFixed(1) + "px,0)";
    }
  }
  function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(update); } }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  update();
})();

/* ---- mode toggle ---- */
document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    mode = btn.dataset.mode;
    const single = mode === "single", manual = mode === "manual", batch = mode === "batch";

    $("uploadTitle").textContent = manual ? "1 · Enter tumor profile"
      : batch ? "1 · Upload cohort file" : "1 · Upload tumor sample";
    $("resultTitle").textContent = batch ? "2 · Cohort recommendations" : "2 · Recommended therapy";

    // show the right input affordance
    $("dropzone").hidden = manual;
    $("manualPanel").hidden = !manual;
    document.querySelector(".samples").hidden = manual;
    if (!manual) {
      $("dzSub").textContent = single
        ? "CSV · TSV · JSON · a single tumor sample"
        : "CSV · TSV · JSON · one tumor per row";
      $("samplesLabel").textContent = single ? "Click an example to match it instantly:" : "Click the cohort to rank it:";
      $("singleChips").hidden = !single;
      $("batchChips").hidden = single;
    }

    submitBtn.textContent = batch ? "Match cohort" : "Match therapy";
    // in manual mode the form is always submittable; file modes need a file
    submitBtn.disabled = manual ? false : !selectedFile;
    if (!manual) $("fileMeta").hidden = !selectedFile;

    // reset view
    $("results").hidden = true; $("batchResults").hidden = true; $("errorBox").hidden = true;
    $("placeholder").hidden = false;
    $("placeholder").innerHTML = batch
      ? `<span aria-hidden="true"><svg class="ico"><use href="#ic-table"/></svg></span><p>Upload a cohort to rank every patient.</p>`
      : manual
      ? `<span aria-hidden="true"><svg class="ico"><use href="#ic-sliders"/></svg></span><p>Fill in the profile and match a therapy.</p>`
      : `<span aria-hidden="true"><svg class="ico"><use href="#ic-dna"/></svg></span><p>Upload a sample to see the ranked match.</p>`;
  });
});

/* ---- build the manual-entry form from the schema ---- */
async function loadSchema() {
  try {
    const s = await (await fetch("/api/schema")).json();
    [...s.mutations, ...s.extras].forEach((m) => { FEATURE_LABELS[m.key] = m.label; });
    $("m_tissue").innerHTML = s.tissues
      .map((c) => `<option value="${c.key}">${escapeHtml(c.label)}</option>`).join("");
    $("mutChecks").innerHTML = s.mutations.map((m) => `
      <label class="check"><input type="checkbox" data-key="${m.key}" />
        <span>${escapeHtml(m.label)}</span></label>`).join("");
    $("extraChecks").innerHTML = s.extras.map((m) => `
      <label class="check"><input type="checkbox" data-key="${m.key}" />
        <span>${escapeHtml(m.label)}</span></label>`).join("");
    restoreFromUrl();   // form exists now; replay a shared case if present
  } catch (_) { /* manual mode just won't populate */ }
}
loadSchema();

function collectManualSample() {
  const out = { tissue: $("m_tissue").value };
  document.querySelectorAll('#mutChecks input, #extraChecks input').forEach((cb) => {
    out[cb.dataset.key] = cb.checked ? 1 : 0;
  });
  return out;
}

function applyManualSample(s) {
  if (s.tissue) $("m_tissue").value = s.tissue;
  document.querySelectorAll('#mutChecks input, #extraChecks input').forEach((cb) => {
    cb.checked = Number(s[cb.dataset.key]) >= 0.5;
  });
}

/* ---- shareable case links (profile encoded in the URL) ---- */
function encodeCase(o) {
  return btoa(JSON.stringify(o)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function decodeCase(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  return JSON.parse(atob(s));
}

function restoreFromUrl() {
  const c = new URLSearchParams(location.search).get("case");
  if (!c) return;
  let sample;
  try { sample = decodeCase(c); } catch (_) { return; }
  showTab("matcher");                                           // open the matcher tab
  document.querySelector('.mode[data-mode="manual"]').click();  // switch to manual view
  applyManualSample(sample);
  if (form.requestSubmit) form.requestSubmit();
  else form.dispatchEvent(new Event("submit", { cancelable: true }));
}

/* ---- file selection ---- */
function setFile(file) {
  selectedFile = file;
  if (file) {
    fileMeta.hidden = false;
    fileMeta.innerHTML = `📄 <b>${escapeHtml(file.name)}</b> · ${(file.size / 1024).toFixed(1)} KB`;
    submitBtn.disabled = false;
  } else {
    fileMeta.hidden = true;
    submitBtn.disabled = true;
  }
}
fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
});

/* ---- submit ---- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (mode !== "manual" && !selectedFile) return;

  $("errorBox").hidden = true;
  $("results").hidden = true;
  $("batchResults").hidden = true;
  $("placeholder").hidden = false;
  $("placeholder").innerHTML = `<span class="spinner"></span><p>${mode === "batch" ? "Ranking cohort…" : "Matching therapy…"}</p>`;
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="spinner"></span>Analyzing…`;

  try {
    let r;
    if (mode === "manual") {
      r = await fetch("/api/predict_form", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectManualSample()),
      });
    } else {
      const fd = new FormData();
      fd.append("file", selectedFile);
      r = await fetch(mode === "batch" ? "/api/predict_batch" : "/api/predict", {
        method: "POST", body: fd,
      });
    }
    const data = await r.json();
    if (!r.ok) throw new Error(apiError(data));
    mode === "batch" ? renderBatch(data) : renderSingle(data);
  } catch (err) {
    $("placeholder").hidden = true;
    const box = $("errorBox");
    box.hidden = false;
    box.textContent = "⚠ " + err.message;
  } finally {
    submitBtn.disabled = mode === "manual" ? false : !selectedFile;
    submitBtn.textContent = mode === "batch" ? "Match cohort" : "Match therapy";
  }
});

/* ---- single render ---- */
function renderSingle(d) {
  $("placeholder").hidden = true;
  $("results").hidden = false;
  currentSample = d.parsed_sample || null;
  $("shareBtn").hidden = !currentSample;

  const top = d.ranked[0];
  $("recName").textContent = top.therapy;
  $("recClass").textContent = top.drug_class;
  $("recBand").textContent = (top.ci_low != null)
    ? `predicted sensitivity ${top.match_percent}%  ·  10–90% interval ${top.ci_low}–${top.ci_high}%`
    : `predicted sensitivity ${top.match_percent}%`;

  const confPct = Math.round(d.confidence * 100);
  $("confVal").textContent = confPct + "%";
  $("confRing").style.setProperty("--p", confPct + "%");

  const marginPts = (d.decision_margin * 100).toFixed(1);
  const mc = $("marginChip");
  mc.textContent = `Decision margin: ${marginPts} pts over #2`;
  mc.className = "metric-chip " + (d.decision_margin >= 0.1 ? "good" : d.decision_margin >= 0.04 ? "mid" : "low");

  // comic "burst" stamp: a punchy verdict word based on how sure the call is
  const headline = document.querySelector("#results .headline");
  if (headline) {
    let burst = headline.querySelector(".burst");
    if (!burst) { burst = document.createElement("span"); burst.className = "burst"; headline.appendChild(burst); }
    burst.textContent = d.confidence >= 0.8 ? "High confidence" : d.confidence >= 0.62 ? "Moderate confidence" : "Low confidence";
    burst.style.animation = "none"; void burst.offsetWidth; burst.style.animation = "";  // replay pop
  }

  $("recSupport").innerHTML = top.supporting.map(renderFactor).join("");
  const cautionCol = $("cautionCol");
  if (top.cautions && top.cautions.length) {
    cautionCol.hidden = false;
    $("recCautions").innerHTML = top.cautions.map(renderFactor).join("");
  } else {
    cautionCol.hidden = true;
  }

  $("ranking").innerHTML = d.ranked.map(renderRankRow).join("");

  renderAssumptions(d);

  const warns = d.warnings || [];
  $("warnings").innerHTML = warns.map((w) => `<div class="warn">⚠ ${escapeHtml(w)}</div>`).join("");
  $("parsedSample").textContent = JSON.stringify(d.parsed_sample, null, 2);
}

/* What the model was actually told, versus what it filled in for you.
   Deliberately not inside the collapsed "Parsed sample" details: an unstated
   biomarker is scored as wild-type, which is an assumption about the tumor,
   not a formatting note. */
function renderAssumptions(d) {
  const box = $("assumptions");
  if (!box) return;
  const assumed = d.assumed_features || [];
  const specified = d.specified_features || [];
  if (!assumed.length && !specified.length) { box.hidden = true; return; }

  const total = assumed.length + specified.length;
  const chip = (f, cls) => `<span class="asm ${cls}">${escapeHtml(featureLabel(f))}</span>`;
  const saw = specified.length
    ? `<div class="asm-row"><b>Measured (${specified.length}/${total}):</b> ${specified.map((f) => chip(f, "known")).join(" ")}</div>`
    : "";
  const guessed = assumed.length
    ? `<div class="asm-row"><b>Assumed negative (${assumed.length}/${total}):</b> ${assumed.map((f) => chip(f, "assumed")).join(" ")}</div>`
    : "";

  box.hidden = false;
  box.className = "assumptions" + (assumed.length ? " has-assumed" : "");
  box.innerHTML = `
    <div class="asm-head">${assumed.length
      ? `⚠ ${assumed.length} of ${total} biomarkers weren't specified — the model scored them as negative / wild-type.`
      : `✓ All ${total} biomarkers were specified.`}</div>
    ${saw}${guessed}`;
}

/* Friendly label for a feature key, from /api/schema when available. */
function featureLabel(key) { return FEATURE_LABELS[key] || key; }

/* Turn an API error body into a message. A 422 that names a field and its valid
   values (e.g. a missing tissue) should say what to fix, not just that it failed. */
function apiError(data) {
  let msg = (data && data.detail) || "Prediction failed.";
  if (data && data.field && Array.isArray(data.valid_values) && data.valid_values.length) {
    const shown = data.valid_values.slice(0, 8).join(", ");
    const more = data.valid_values.length > 8 ? `, … (${data.valid_values.length} total)` : "";
    msg += ` Valid ${data.field} values: ${shown}${more}.`;
  }
  return msg;
}

function renderFactor(f) {
  const eff = (f.effect_pct == null) ? ""
    : `<span class="eff ${f.effect_pct >= 0 ? "pos" : "neg"}">${f.effect_pct >= 0 ? "+" : ""}${f.effect_pct} pts</span>`;
  return `<li><div class="factor-head"><b>${escapeHtml(f.label)}</b>${eff}</div><span>${escapeHtml(f.text)}</span></li>`;
}

function renderRankRow(t) {
  // interval markers relative to the 0–100 bar
  let band = "";
  if (t.ci_low != null) {
    const w = Math.max(2, t.ci_high - t.ci_low);
    band = `<span class="ci" style="left:${t.ci_low}%;width:${w}%" title="10–90% interval ${t.ci_low}–${t.ci_high}%"></span>`;
  }
  const caution = (t.cautions && t.cautions.length)
    ? `<span class="cau-badge" title="${escapeHtml(t.cautions.map((c) => c.label).join(', '))}">⚠ ${t.cautions.length}</span>` : "";
  return `
    <li>
      <span class="rank-num">${t.rank}</span>
      <span class="rank-name"><b>${escapeHtml(t.therapy)}</b> ${caution}<small>${escapeHtml(t.drug_class)}</small></span>
      <span class="bar-wrap">
        <span class="bar">${band}<i style="width:${t.match_percent}%"></i></span>
        <span class="bar-val">${t.match_percent}%</span>
      </span>
    </li>`;
}

/* ---- batch render ---- */
function renderBatch(d) {
  $("placeholder").hidden = true;
  $("batchResults").hidden = false;
  lastBatch = d;

  $("batchCount").textContent = d.n;
  $("cohortBody").innerHTML = d.rows.map((r) => {
    const mcls = r.decision_margin >= 0.1 ? "good" : r.decision_margin >= 0.04 ? "mid" : "low";
    return `<tr>
      <td>${r.index}</td>
      <td>${escapeHtml(r.cancer_type)}</td>
      <td><b>${escapeHtml(r.recommendation)}</b><small>${escapeHtml(r.drug_class)}</small></td>
      <td>${r.match_percent}%</td>
      <td>${Math.round(r.confidence * 100)}%</td>
      <td><span class="dot ${mcls}"></span>${(r.decision_margin * 100).toFixed(0)}</td>
      <td class="muted">${escapeHtml(r.runner_up)} · ${r.runner_up_percent}%</td>
    </tr>`;
  }).join("");

  const warns = d.warnings || [];
  $("batchWarnings").innerHTML = warns.length
    ? warns.map((w) => `<div class="warn">⚠ ${escapeHtml(w)}</div>`).join("")
    : "<p class='hint'>No parsing issues.</p>";
}

$("shareBtn").addEventListener("click", async () => {
  if (!currentSample) return;
  const url = location.origin + location.pathname + "?case=" + encodeCase(currentSample);
  try {
    await navigator.clipboard.writeText(url);
    flashShare("✓ Link copied!");
  } catch (_) {
    window.prompt("Copy this shareable link:", url);
  }
});

function flashShare(msg) {
  const b = $("shareBtn");
  const orig = b.textContent;
  b.textContent = msg;
  b.classList.add("ok");
  setTimeout(() => { b.textContent = orig; b.classList.remove("ok"); }, 1800);
}

$("downloadCsv").addEventListener("click", () => {
  if (!lastBatch) return;
  const head = ["index", "cancer_type", "recommendation", "drug_class",
    "match_percent", "confidence", "decision_margin", "runner_up", "runner_up_percent"];
  const lines = [head.join(",")];
  lastBatch.rows.forEach((r) => {
    lines.push(head.map((k) => csvCell(r[k])).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "karkive_cohort_results.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---- click-to-run example samples ---- */
document.querySelectorAll(".chip.run").forEach((a) => {
  a.addEventListener("click", async (e) => {
    e.preventDefault();
    const url = a.getAttribute("href");
    const kind = a.dataset.kind || "single";
    $("errorBox").hidden = true;
    $("results").hidden = true;
    $("batchResults").hidden = true;
    $("placeholder").hidden = false;
    $("placeholder").innerHTML =
      `<span class="spinner"></span><p>${kind === "batch" ? "Ranking cohort…" : "Matching therapy…"}</p>`;
    try {
      const blob = await (await fetch(url)).blob();
      const fd = new FormData();
      fd.append("file", blob, url.split("/").pop());
      const ep = kind === "batch" ? "/api/predict_batch" : "/api/predict";
      const r = await fetch(ep, { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(apiError(data));
      kind === "batch" ? renderBatch(data) : renderSingle(data);
    } catch (err) {
      $("placeholder").hidden = true;
      const box = $("errorBox"); box.hidden = false; box.textContent = "⚠ " + err.message;
    }
  });
});

/* ---- rotating hero example ---- */
(function heroTicker() {
  const el = document.getElementById("rotEx");
  if (!el) return;
  const items = ["BRAF melanoma → Dabrafenib", "EGFR lung → Afatinib",
                 "PIK3CA breast → Alpelisib", "KRAS pancreas → Trametinib",
                 "NRAS melanoma → Trametinib", "HER2 breast → Afatinib"];
  let i = 0;
  setInterval(() => {
    i = (i + 1) % items.length;
    el.classList.add("swap");
    setTimeout(() => { el.textContent = items[i]; el.classList.remove("swap"); }, 260);
  }, 2800);
})();

/* ---- subtle 3D tilt on cards (pointer devices) ---- */
(function cardTilt() {
  if (window.matchMedia("(hover: none)").matches) return;
  document.querySelectorAll(".feature, .vision-card").forEach((card) => {
    card.addEventListener("mousemove", (e) => {
      const r = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width - 0.5;
      const py = (e.clientY - r.top) / r.height - 0.5;
      card.style.setProperty("transform",
        `perspective(720px) rotateX(${(-py * 5).toFixed(2)}deg) rotateY(${(px * 5).toFixed(2)}deg) translateY(-3px)`,
        "important");
    });
    card.addEventListener("mouseleave", () => card.style.removeProperty("transform"));
  });
})();

/* ---- soft drifting "cell field" background ----
   Translucent cells (soft body + faint membrane + nucleus) in brand colours
   float slowly across the page — a calm nod to the cancer cell lines the
   model learns from. Not interactive with the cursor. */
(function bgCells() {
  const canvas = document.getElementById("bgnet");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COLORS = ["79,70,229", "13,148,136", "124,92,246", "45,140,255", "214,90,150"];
  let W, H, DPR, cells = [], raf = null;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.width = Math.floor(innerWidth * DPR);
    H = canvas.height = Math.floor(innerHeight * DPR);
    canvas.style.width = innerWidth + "px";
    canvas.style.height = innerHeight + "px";
    seed();
  }
  function seed() {
    const n = Math.max(12, Math.min(30, Math.round((innerWidth * innerHeight) / 46000)));
    cells = Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      r: (Math.random() * 72 + 32) * DPR,
      vx: (Math.random() - 0.5) * 0.13 * DPR, vy: (Math.random() - 0.5) * 0.13 * DPR,
      c: COLORS[(Math.random() * COLORS.length) | 0],
      ph: Math.random() * 6.28, pv: 0.003 + Math.random() * 0.005,
    }));
  }
  function frame() {
    ctx.clearRect(0, 0, W, H);
    for (const c of cells) {
      c.x += c.vx; c.y += c.vy; c.ph += c.pv;
      const m = c.r * 1.3;
      if (c.x < -m) c.x = W + m; else if (c.x > W + m) c.x = -m;
      if (c.y < -m) c.y = H + m; else if (c.y > H + m) c.y = -m;
      const r = c.r * (1 + 0.05 * Math.sin(c.ph));
      const g = ctx.createRadialGradient(c.x, c.y, r * 0.1, c.x, c.y, r);
      g.addColorStop(0, `rgba(${c.c},0.17)`);
      g.addColorStop(0.7, `rgba(${c.c},0.06)`);
      g.addColorStop(1, `rgba(${c.c},0)`);
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 6.2832); ctx.fill();
      ctx.strokeStyle = `rgba(${c.c},0.1)`; ctx.lineWidth = 1.2 * DPR;
      ctx.beginPath(); ctx.arc(c.x, c.y, r * 0.82, 0, 6.2832); ctx.stroke();
      ctx.fillStyle = `rgba(${c.c},0.13)`;
      ctx.beginPath(); ctx.arc(c.x, c.y, r * 0.22, 0, 6.2832); ctx.fill();
    }
    raf = requestAnimationFrame(frame);
  }
  const start = () => { if (!raf && !reduce) raf = requestAnimationFrame(frame); };
  const stop = () => { if (raf) { cancelAnimationFrame(raf); raf = null; } };
  window.addEventListener("resize", resize, { passive: true });
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  resize();
  if (reduce) frame(); else start();
})();

/* ---- AI diagram: auto-cycle lane highlighting until the user hovers ---- */
(function diagramCycle() {
  const dia = document.getElementById("matchDia");
  if (!dia) return;
  const lanes = Array.from(dia.querySelectorAll(".lane"));
  if (!lanes.length) return;
  let i = 0, timer = null, paused = false;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function step() {
    if (paused) return;
    lanes.forEach((l, k) => l.classList.toggle("active", k === i));
    i = (i + 1) % lanes.length;
  }
  function start() {
    if (reduce || timer) return;
    dia.classList.add("cycling");
    step(); timer = setInterval(step, 1600);
  }
  function stop() {
    dia.classList.remove("cycling");
    lanes.forEach((l) => l.classList.remove("active"));
    if (timer) { clearInterval(timer); timer = null; }
  }
  // pause the auto-tour while the user explores by hand
  dia.addEventListener("mouseenter", () => { paused = true; stop(); });
  dia.addEventListener("mouseleave", () => { paused = false; start(); });
  // only run the tour while the diagram is on screen
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting && !paused) start(); else if (!e.isIntersecting) stop(); });
    }, { threshold: 0.3 }).observe(dia);
  } else { start(); }
})();
