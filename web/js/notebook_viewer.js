// notebook_viewer.js — Static Jupyter-like viewer for nbformat 4 JSON.

import { marked } from "https://cdn.jsdelivr.net/npm/marked@12.0.2/+esm";

const NOTEBOOK_URL = "assets/notebooks/model_comparison.ipynb";

let loaded = false;

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sourceText(cell) {
  const src = cell.source;
  if (Array.isArray(src)) return src.join("");
  return src || "";
}

function renderMarkdown(md) {
  marked.setOptions({ gfm: true, breaks: false });
  return marked.parse(md || "");
}

function mimeText(data, mime) {
  if (!data || data[mime] == null) return "";
  const v = data[mime];
  return Array.isArray(v) ? v.join("") : String(v);
}

function renderOutput(output, outIdx) {
  if (!output) return "";
  const type = output.output_type;

  if (type === "stream") {
    const text = Array.isArray(output.text) ? output.text.join("") : output.text || "";
    const cls = output.name === "stderr" ? "nb-stream nb-stderr" : "nb-stream";
    return `<pre class="${cls}">${escapeHtml(text)}</pre>`;
  }

  if (type === "error") {
    const tb = (output.traceback || []).join("\n");
    return `<pre class="nb-stream nb-stderr">${escapeHtml(tb || `${output.ename}: ${output.evalue}`)}</pre>`;
  }

  const data = output.data || {};
  if (data["image/png"]) {
    const b64 = mimeText(data, "image/png").replace(/\s/g, "");
    return `<div class="nb-media"><img src="data:image/png;base64,${b64}" alt="Salida gráfica de la celda" /></div>`;
  }
  if (data["text/html"]) {
    return `<div class="nb-html-out">${mimeText(data, "text/html")}</div>`;
  }
  if (data["text/plain"]) {
    // Skip Jupyter widget placeholders when we already show stream metrics nearby.
    const plain = mimeText(data, "text/plain");
    if (plain.includes("FloatProgress") || plain.includes("widget")) {
      return `<pre class="nb-stream nb-muted">${escapeHtml(plain)}</pre>`;
    }
    return `<pre class="nb-stream">${escapeHtml(plain)}</pre>`;
  }
  if (data["application/vnd.jupyter.widget-view+json"]) {
    return `<p class="nb-widget-skip">Widget de progreso (omitido en el visor web).</p>`;
  }

  return outIdx === 0 ? "" : "";
}

function renderCell(cell, index) {
  const exec = cell.execution_count;
  if (cell.cell_type === "markdown") {
    return `<div class="nb-cell nb-md" data-cell="${index}">
      <div class="nb-md-body">${renderMarkdown(sourceText(cell))}</div>
    </div>`;
  }
  if (cell.cell_type !== "code") return "";

  const prompt = exec != null ? `In [${exec}]:` : "In [ ]:";
  const code = escapeHtml(sourceText(cell));
  const outputs = (cell.outputs || []).map((o, i) => renderOutput(o, i)).join("");
  const outPrompt =
    outputs.trim() && exec != null
      ? `<div class="nb-prompt nb-prompt-out">Out[${exec}]:</div>`
      : outputs.trim()
        ? `<div class="nb-prompt nb-prompt-out">Out[ ]:</div>`
        : "";

  return `<div class="nb-cell nb-code" data-cell="${index}">
    <div class="nb-code-row">
      <div class="nb-prompt">${escapeHtml(prompt)}</div>
      <pre class="nb-src"><code>${code}</code></pre>
    </div>
    ${
      outputs.trim()
        ? `<div class="nb-out-row">${outPrompt}<div class="nb-out-body">${outputs}</div></div>`
        : ""
    }
  </div>`;
}

/**
 * Lazy-load and render the comparison notebook into `#notebook-root`.
 */
export async function ensureNotebookLoaded() {
  if (loaded) return;
  const root = document.getElementById("notebook-root");
  if (!root) return;

  root.innerHTML = `<p class="muted">Cargando notebook…</p>`;
  try {
    const resp = await fetch(NOTEBOOK_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const nb = await resp.json();
    const cells = nb.cells || [];
    root.innerHTML = `<div class="nb-notebook">${cells.map(renderCell).join("")}</div>`;
    loaded = true;
  } catch (e) {
    console.error(e);
    root.innerHTML = `<p class="status error">No se pudo cargar el notebook: ${escapeHtml(e.message)}</p>`;
  }
}

export function notebookDownloadUrl() {
  return NOTEBOOK_URL;
}
