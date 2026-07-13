// pipeline_viz.js — Workflow visual del pipeline de análisis.
// Desktop: rail horizontal (DAG). Móvil: timeline vertical.
// Estados: pending | running | done | error | skipped
// Learn mode: educational explainers (English) via pipeline_edu.js

import * as edu from "./pipeline_edu.js";

const ICONS = {
  input: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></svg>`,
  preprocess: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/><circle cx="12" cy="12" r="4"/><path d="m4.9 4.9 2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/></svg>`,
  vectorize: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`,
  nb: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M4 19V5M4 19h16"/><path d="M8 16v-5M12 16V8M16 16v-3"/></svg>`,
  logreg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M4 19V5M4 19h16"/><path d="M7 15c3-8 7-10 13-10"/></svg>`,
  transformer: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2"/></svg>`,
  sensationalism: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a9 9 0 0 1 0 14"/></svg>`,
  sentiment: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/></svg>`,
  consenso: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4 12 14.01l-3-3"/></svg>`,
};

const ETAPAS = [
  { id: "input", label: "Entrada", sub: "Texto", flechaDespues: true },
  { id: "preprocess", label: "Preproceso", sub: "Tokens", flechaDespues: true },
  { id: "vectorize", label: "TF-IDF", sub: "Vectores", flechaDespues: true },
  { id: "nb", label: "Naive Bayes", sub: "Clásico", iniciaGrupo: true },
  { id: "logreg", label: "LogReg", sub: "Clásico" },
  { id: "transformer", label: "ELECTRA", sub: "Transformer ONNX", cierraGrupo: true, flechaDespues: true },
  { id: "sensationalism", label: "Tono", sub: "Sensacionalismo", flechaDespues: true },
  { id: "sentiment", label: "Sentimiento", sub: "RoBERTuito + léxico", flechaDespues: true },
  { id: "consenso", label: "Consenso", sub: "Veredicto NLP" },
];

let containerRef = null;
let learnMode = true;
let focusedStage = null;

function $pv(suffix) {
  if (!containerRef) return null;
  return containerRef.querySelector(`[data-pv="${suffix}"]`);
}

function teachSlot() {
  return containerRef ? containerRef.querySelector('[data-pv="teach-slot"]') : null;
}

function nodoHTML(etapa) {
  return `
    <div class="pipeline-nodo" data-pv="nodo-${etapa.id}" data-stage="${etapa.id}" role="button" tabindex="0" title="Clic para aprender">
      <div class="pipeline-nodo-icono" aria-hidden="true">${ICONS[etapa.id] || ICONS.input}</div>
      <div class="pipeline-nodo-contenido">
        <div class="pipeline-nodo-titulo">${etapa.label}</div>
        <div class="pipeline-nodo-sub" data-pv="sub-${etapa.id}">${etapa.sub}</div>
        <div class="pipeline-nodo-resultado" data-pv="resultado-${etapa.id}"></div>
        <div class="pipeline-nodo-progress" data-pv="progress-${etapa.id}" hidden>
          <div class="pipeline-progress-track">
            <div class="pipeline-progress-bar" data-pv="bar-${etapa.id}"></div>
          </div>
          <span class="pipeline-progress-pct" data-pv="pct-${etapa.id}">0%</span>
        </div>
      </div>
      <div class="pipeline-nodo-status" data-pv="spinner-${etapa.id}" aria-hidden="true"></div>
    </div>`;
}

function conectorHTML() {
  return `<div class="pipeline-conector" aria-hidden="true">
    <span class="pipeline-conector-line"></span>
  </div>`;
}

function bindNodeClicks() {
  if (!containerRef) return;
  containerRef.querySelectorAll(".pipeline-nodo").forEach((nodo) => {
    const id = nodo.getAttribute("data-stage");
    const open = () => focusStage(id, true);
    nodo.addEventListener("click", open);
    nodo.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open();
      }
    });
  });
}

function bindLearnToggle() {
  const btn = $pv("learn-toggle");
  if (!btn) return;
  const syncLabel = () => {
    const ui = edu.uiStrings();
    btn.setAttribute("aria-pressed", learnMode ? "true" : "false");
    btn.classList.toggle("activo", learnMode);
    btn.textContent = learnMode ? ui.learnOn : ui.learnOff;
  };
  syncLabel();
  btn.onclick = () => {
    learnMode = !learnMode;
    syncLabel();
    containerRef.classList.toggle("learn-on", learnMode);
    if (!learnMode) {
      edu.hideTeachPanel(teachSlot());
      clearFocusHighlight();
    } else if (focusedStage) {
      focusStage(focusedStage, true);
    }
  };
}

function bindLangToggle() {
  const group = $pv("lang-toggle");
  if (!group) return;
  const current = edu.getLang();
  group.querySelectorAll("[data-lang]").forEach((btn) => {
    const on = btn.getAttribute("data-lang") === current;
    btn.classList.toggle("activo", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.onclick = () => {
      const next = btn.getAttribute("data-lang");
      edu.setLang(next);
      group.querySelectorAll("[data-lang]").forEach((b) => {
        const active = b.getAttribute("data-lang") === next;
        b.classList.toggle("activo", active);
        b.setAttribute("aria-pressed", active ? "true" : "false");
      });
      bindLearnToggle();
      if (learnMode && focusedStage) {
        focusStage(focusedStage, false);
      }
    };
  });
}

function clearFocusHighlight() {
  if (!containerRef) return;
  containerRef.querySelectorAll(".pipeline-nodo.focus-edu").forEach((n) => {
    n.classList.remove("focus-edu");
  });
}

/**
 * Focus a stage for education. force=true opens even if learn was toggled mid-way via click.
 */
export function focusStage(id, fromUser = false) {
  focusedStage = id;
  if (!learnMode && !fromUser) return;
  if (!learnMode && fromUser) {
    learnMode = true;
    bindLearnToggle();
    if (containerRef) containerRef.classList.add("learn-on");
  }
  clearFocusHighlight();
  const nodo = $pv(`nodo-${id}`);
  if (nodo) nodo.classList.add("focus-edu");
  const slot = teachSlot();
  if (slot) edu.renderTeachPanel(slot, id);
}

export function setLiveData(id, data) {
  edu.setLiveData(id, data);
  edu.refreshTeachIf(id);
}

export function isLearnMode() {
  return learnMode;
}

/**
 * Dibuja el diagrama del pipeline dentro del contenedor dado.
 */
export function render(container) {
  if (!container) return;
  containerRef = container;
  container.className = "pipeline-viz-container" + (learnMode ? " learn-on" : "");
  focusedStage = null;
  edu.clearLiveData();

  let html = "";
  for (const etapa of ETAPAS) {
    if (etapa.iniciaGrupo) {
      html += `<div class="pipeline-grupo-wrap">`;
      html += `<div class="pipeline-grupo-label">Comparación de modelos</div>`;
      html += `<div class="pipeline-grupo">`;
    }

    html += nodoHTML(etapa);

    if (etapa.cierraGrupo) {
      html += `</div></div>`;
    }

    if (etapa.flechaDespues) {
      html += conectorHTML();
    }
  }

  container.innerHTML = `
    <div class="pipeline-viz-header">
      <div class="pipeline-viz-header-text">
        <h3 class="pipeline-viz-titulo">Pipeline NLP</h3>
        <p class="pipeline-viz-hint">Categoría · tono · sentimiento</p>
      </div>
      <div class="pipeline-viz-controls">
        <div class="pipeline-lang-toggle" data-pv="lang-toggle" role="group" aria-label="Idioma del explicador">
          <button type="button" class="pipeline-lang-btn" data-lang="es" aria-pressed="true">ES</button>
          <button type="button" class="pipeline-lang-btn" data-lang="en" aria-pressed="false">EN</button>
        </div>
        <button type="button" class="pipeline-learn-toggle activo" data-pv="learn-toggle" aria-pressed="true">
          Modo aprender ON
        </button>
      </div>
    </div>
    <div class="pipeline-viz-nodos">${html}</div>
    <div class="pipeline-teach-slot" data-pv="teach-slot" hidden></div>`;

  bindLearnToggle();
  bindLangToggle();
  bindNodeClicks();
}

export function reset() {
  if (!containerRef) return;
  focusedStage = null;
  edu.clearLiveData();
  edu.hideTeachPanel(teachSlot());
  for (const nodo of containerRef.querySelectorAll(".pipeline-nodo")) {
    nodo.className = "pipeline-nodo";
  }
  for (const sp of containerRef.querySelectorAll(".pipeline-nodo-status")) {
    sp.className = "pipeline-nodo-status";
  }
  for (const r of containerRef.querySelectorAll(".pipeline-nodo-resultado")) {
    r.innerHTML = "";
  }
  for (const p of containerRef.querySelectorAll(".pipeline-nodo-progress")) {
    p.hidden = true;
  }
  for (const c of containerRef.querySelectorAll(".pipeline-conector")) {
    c.classList.remove("activo", "done");
  }
  for (const etapa of ETAPAS) {
    const sub = $pv(`sub-${etapa.id}`);
    if (sub) sub.textContent = etapa.sub;
  }
}

function marcarConectorPrev(id) {
  const nodo = $pv(`nodo-${id}`);
  if (!nodo) return;
  const prev = nodo.previousElementSibling;
  if (prev && prev.classList.contains("pipeline-conector")) {
    prev.classList.add("activo");
  }
  const grupoWrap = nodo.closest(".pipeline-grupo-wrap");
  if (grupoWrap) {
    const before = grupoWrap.previousElementSibling;
    if (before && before.classList.contains("pipeline-conector")) {
      before.classList.add("activo");
    }
  }
}

function completarConectorPrev(id) {
  const nodo = $pv(`nodo-${id}`);
  if (!nodo) return;
  const prev = nodo.previousElementSibling;
  if (prev && prev.classList.contains("pipeline-conector")) {
    prev.classList.remove("activo");
    prev.classList.add("done");
  }
  const grupoWrap = nodo.closest(".pipeline-grupo-wrap");
  if (grupoWrap && id === "transformer") {
    const before = grupoWrap.previousElementSibling;
    if (before && before.classList.contains("pipeline-conector")) {
      before.classList.remove("activo");
      before.classList.add("done");
    }
  }
}

export async function iniciarEtapa(id) {
  const nodo = $pv(`nodo-${id}`);
  if (!nodo) return;
  nodo.className = "pipeline-nodo running";
  const spinner = $pv(`spinner-${id}`);
  if (spinner) spinner.className = "pipeline-nodo-status activo";
  marcarConectorPrev(id);
  if (learnMode) focusStage(id, false);
  await sleep(220);
}

export function actualizarProgreso(id, pct, file) {
  const wrap = $pv(`progress-${id}`);
  const bar = $pv(`bar-${id}`);
  const pctEl = $pv(`pct-${id}`);
  const sub = $pv(`sub-${id}`);
  if (wrap) wrap.hidden = false;
  const p = Math.max(0, Math.min(100, Number(pct) || 0));
  if (bar) bar.style.width = `${p}%`;
  if (pctEl) pctEl.textContent = `${p.toFixed(0)}%`;
  if (sub && file) {
    const corto = String(file).split("/").pop();
    sub.textContent = corto;
  }
}

export async function completarEtapa(id, resultado) {
  const nodo = $pv(`nodo-${id}`);
  if (!nodo) return;
  const spinner = $pv(`spinner-${id}`);
  if (spinner) spinner.className = "pipeline-nodo-status";
  const progress = $pv(`progress-${id}`);
  if (progress) progress.hidden = true;
  nodo.className = "pipeline-nodo done";
  if (focusedStage === id) nodo.classList.add("focus-edu");
  completarConectorPrev(id);
  const res = $pv(`resultado-${id}`);
  if (res && resultado) {
    res.innerHTML = `<span class="pipeline-resultado-badge">${resultado}</span>`;
  }
  await sleep(160);
}

export function errorEtapa(id, msg) {
  const nodo = $pv(`nodo-${id}`);
  if (!nodo) return;
  const spinner = $pv(`spinner-${id}`);
  if (spinner) spinner.className = "pipeline-nodo-status";
  const progress = $pv(`progress-${id}`);
  if (progress) progress.hidden = true;
  nodo.className = "pipeline-nodo error";
  completarConectorPrev(id);
  const res = $pv(`resultado-${id}`);
  if (res) {
    res.innerHTML = `<span class="pipeline-resultado-badge error">${msg || "Error"}</span>`;
  }
}

export function omitirEtapa(id, msg) {
  const nodo = $pv(`nodo-${id}`);
  if (!nodo) return;
  const spinner = $pv(`spinner-${id}`);
  if (spinner) spinner.className = "pipeline-nodo-status";
  const progress = $pv(`progress-${id}`);
  if (progress) progress.hidden = true;
  nodo.className = "pipeline-nodo skipped";
  completarConectorPrev(id);
  const res = $pv(`resultado-${id}`);
  if (res) {
    res.innerHTML = `<span class="pipeline-resultado-badge skipped">${msg || "Omitido"}</span>`;
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
