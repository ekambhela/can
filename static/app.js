"use strict";

const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropzone = $("dropzone");
const fileMeta = $("fileMeta");
const submitBtn = $("submitBtn");
const form = $("uploadForm");

let selectedFile = null;

/* ---- model stats in the header ---- */
async function loadStats() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const m = d.metrics || {};
    if (m.top1_accuracy != null) $("statTop1").textContent = (m.top1_accuracy * 100).toFixed(1) + "%";
    if (m.top3_accuracy != null) $("statTop3").textContent = (m.top3_accuracy * 100).toFixed(1) + "%";
    if (m.mean_r2 != null) $("statR2").textContent = m.mean_r2.toFixed(2);
    if (m.n_therapies) $("nTherapies").textContent = m.n_therapies;
  } catch (_) { /* non-fatal */ }
}
loadStats();

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
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
});

/* ---- submit ---- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  $("errorBox").hidden = true;
  $("results").hidden = true;
  $("placeholder").hidden = false;
  $("placeholder").innerHTML = `<span class="spinner"></span><p>Matching therapy…</p>`;
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="spinner"></span>Analyzing…`;

  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Prediction failed.");
    renderResults(data);
  } catch (err) {
    $("placeholder").hidden = true;
    $("results").hidden = true;
    const box = $("errorBox");
    box.hidden = false;
    box.textContent = "⚠ " + err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Match therapy";
  }
});

/* ---- render ---- */
function renderResults(d) {
  $("placeholder").hidden = true;
  $("results").hidden = false;

  const top = d.ranked[0];
  $("recName").textContent = top.therapy;
  $("recClass").textContent = top.drug_class;

  const confPct = Math.round(d.confidence * 100);
  $("confVal").textContent = confPct + "%";
  $("confRing").style.setProperty("--p", confPct + "%");

  $("recRationale").innerHTML = top.rationale.map((r) => `<li>${escapeHtml(r)}</li>`).join("");

  $("ranking").innerHTML = d.ranked.map((t) => `
    <li>
      <span class="rank-num">${t.rank}</span>
      <span class="rank-name"><b>${escapeHtml(t.therapy)}</b><small>${escapeHtml(t.drug_class)}</small></span>
      <span class="bar-wrap">
        <span class="bar"><i style="width:${t.match_percent}%"></i></span>
        <span class="bar-val">${t.match_percent}%</span>
      </span>
    </li>`).join("");

  // warnings
  const warns = (d.warnings || []);
  $("warnings").innerHTML = warns.length
    ? warns.map((w) => `<div class="warn">⚠ ${escapeHtml(w)}</div>`).join("")
    : "";

  $("parsedSample").textContent = JSON.stringify(d.parsed_sample, null, 2);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
