"use strict";

const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropzone = $("dropzone");
const fileMeta = $("fileMeta");
const submitBtn = $("submitBtn");
const form = $("uploadForm");

let selectedFile = null;
let mode = "single";          // "single" | "batch"
let lastBatch = null;         // cached cohort result for CSV export

/* ---- model stats in the header ---- */
async function loadStats() {
  try {
    const d = await (await fetch("/api/health")).json();
    const m = d.metrics || {};
    if (m.top1_accuracy != null) $("statTop1").textContent = (m.top1_accuracy * 100).toFixed(1) + "%";
    if (m.top3_accuracy != null) $("statTop3").textContent = (m.top3_accuracy * 100).toFixed(1) + "%";
    if (m.mean_r2 != null) $("statR2").textContent = m.mean_r2.toFixed(2);
    if (m.interval_coverage != null) $("statCov").textContent = (m.interval_coverage * 100).toFixed(0) + "%";
    if (m.n_therapies) $("nTherapies").textContent = m.n_therapies;
  } catch (_) { /* non-fatal */ }
}
loadStats();

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
        ? "CSV · TSV · JSON — a single tumor sample"
        : "CSV · TSV · JSON — one tumor per row";
      $("samplesLabel").textContent = single ? "No sample handy? Try one:" : "No cohort handy? Try one:";
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
      ? `<span aria-hidden="true">📋</span><p>Upload a cohort to rank every patient.</p>`
      : manual
      ? `<span aria-hidden="true">🧬</span><p>Fill in the profile and match a therapy.</p>`
      : `<span aria-hidden="true">🧬</span><p>Upload a sample to see the ranked match.</p>`;
  });
});

/* ---- build the manual-entry form from the schema ---- */
async function loadSchema() {
  try {
    const s = await (await fetch("/api/schema")).json();
    $("m_cancer_type").innerHTML = s.cancer_types
      .map((c) => `<option value="${c.key}">${escapeHtml(c.label)}</option>`).join("");
    $("mutChecks").innerHTML = s.mutations.map((m) => `
      <label class="check"><input type="checkbox" data-key="${m.key}" />
        <span>${escapeHtml(m.label)}</span></label>`).join("");
    $("contSliders").innerHTML = s.continuous.map((c) => `
      <label class="slider">
        <span class="slider-head">${escapeHtml(c.label)}
          <b id="val_${c.key}">${c.default}</b></span>
        <input type="range" data-key="${c.key}" min="${c.min}" max="${c.max}"
               step="${c.step}" value="${c.default}"
               oninput="document.getElementById('val_${c.key}').textContent = this.value" />
      </label>`).join("");
  } catch (_) { /* manual mode just won't populate */ }
}
loadSchema();

function collectManualSample() {
  const out = { cancer_type: $("m_cancer_type").value };
  document.querySelectorAll('#mutChecks input[type="checkbox"]').forEach((cb) => {
    out[cb.dataset.key] = cb.checked ? 1 : 0;
  });
  document.querySelectorAll('#contSliders input[type="range"]').forEach((r) => {
    out[r.dataset.key] = parseFloat(r.value);
  });
  return out;
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
    if (!r.ok) throw new Error(data.detail || "Prediction failed.");
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

  $("recSupport").innerHTML = top.supporting.map(renderFactor).join("");
  const cautionCol = $("cautionCol");
  if (top.cautions && top.cautions.length) {
    cautionCol.hidden = false;
    $("recCautions").innerHTML = top.cautions.map(renderFactor).join("");
  } else {
    cautionCol.hidden = true;
  }

  $("ranking").innerHTML = d.ranked.map(renderRankRow).join("");

  const warns = d.warnings || [];
  $("warnings").innerHTML = warns.map((w) => `<div class="warn">⚠ ${escapeHtml(w)}</div>`).join("");
  $("parsedSample").textContent = JSON.stringify(d.parsed_sample, null, 2);
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
