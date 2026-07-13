// app.js — Lógica principal de la interfaz del Clasificador de Noticias.
// Orquesta: preprocess → vectorize → NB → LogReg → transformer → tono → sentimiento → consenso.

import * as preprocess from "./preprocess.js";
import * as vectorize from "./vectorize.js";
import * as nb from "./naive_bayes.js";
import * as logreg from "./logreg.js";
import * as sensationalism from "./sensationalism.js";
import * as sentimentLex from "./sentiment_lexicon.js";
import * as transformer from "./transformer.js";
import { extraerNoticiaDeURL, validarEsNoticia } from "./url_extractor.js";
import * as pv from "./pipeline_viz.js";
import { ensureNotebookLoaded } from "./notebook_viewer.js";
import {
  dibujarBarras,
  dibujarDonut,
  dibujarBarraConfianza,
  COLORES_CATEGORIA,
  COLORES_SENTIMIENTO,
} from "./charts.js";

let inicializado = false;
let modelosCargados = { nb: false, logreg: false, lexico: false, reglas: false };
let transformerListo = { categoria: false, sentimiento: false };
let ejemplosNoticias = [];
let precargaIniciada = false;

async function inicializar() {
  if (inicializado) return;
  mostrarEstado("Cargando modelos clásicos...", "info");

  try {
    const [nbData, logregData, stopw, lexic, reglas, ejemplos] = await Promise.all([
      nb.cargarNB(),
      logreg.cargarLogReg(),
      preprocess.cargarStopwords(),
      sentimentLex.cargarLexico(),
      sensationalism.cargarReglas(),
      cargarEjemplos(),
    ]);

    const vi = nb.getVocabInfo();
    vectorize.setVocab(vi.vocabulary, vi.idf, vi.ngram_range);
    preprocess.construirLemmaMap(vi.vocabulary);

    modelosCargados = { nb: true, logreg: true, lexico: true, reglas: true };
    inicializado = true;
    ejemplosNoticias = ejemplos;
    mostrarBotonesEjemplos();
    if (!transformer.cacheDisponible) {
      mostrarEstado(
        "Listo. Caché del navegador no disponible (usa localhost o HTTPS para cachear modelos).",
        "warn"
      );
    } else {
      mostrarEstado("Listo. Puedes analizar una noticia.", "ok");
    }
    programarPrecarga();
  } catch (e) {
    console.error(e);
    mostrarEstado("Error cargando modelos: " + e.message, "error");
  }
}

function programarPrecarga() {
  if (precargaIniciada) return;
  precargaIniciada = true;
  const run = () => precargarModelosONNX();
  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(run, { timeout: 2500 });
  } else {
    setTimeout(run, 1200);
  }
}

async function precargarModelosONNX() {
  if (!transformerListo.categoria) {
    try {
      mostrarEstado("Precargando ELECTRA en segundo plano (~14 MB)…", "info");
      await transformer.cargarClasificadorCategoria((prog, file) => {
        mostrarEstado(`Precargando ELECTRA: ${prog.toFixed(0)}%`, "info");
      });
      transformerListo.categoria = true;
      mostrarEstado("ELECTRA listo. Puedes analizar.", "ok");
    } catch (e) {
      console.warn("Precarga ELECTRA falló:", e);
      mostrarEstado("Listo (ELECTRA se cargará al analizar).", "ok");
    }
  }
  if (!transformerListo.sentimiento) {
    try {
      await transformer.cargarClasificadorSentimiento();
      transformerListo.sentimiento = true;
    } catch (e) {
      console.warn("Precarga RoBERTuito falló (se usará léxico si hace falta):", e);
    }
  }
}

async function cargarEjemplos() {
  try {
    const resp = await fetch("assets/sample_news.json");
    return await resp.json();
  } catch {
    return [];
  }
}

async function analizarTexto() {
  const textarea = document.getElementById("input-texto");
  const texto = textarea.value.trim();
  if (!texto) {
    mostrarEstado("Pega una noticia para analizar.", "warn");
    return;
  }
  await ejecutarAnalisis(texto);
}

async function analizarURL() {
  const input = document.getElementById("input-url");
  const url = input.value.trim();
  if (!url) {
    mostrarEstado("Pega una URL de noticia para analizar.", "warn");
    return;
  }

  setAnalizandoURL(true);
  document.getElementById("url-resultados").style.display = "none";
  document.getElementById("url-advertencia").style.display = "none";

  try {
    const noticia = await extraerNoticiaDeURL(url, (msg, tipo) => mostrarEstado(msg, tipo));

    const metaDiv = document.getElementById("url-meta");
    metaDiv.innerHTML = `
      <div class="url-meta-item"><strong>Titular:</strong> ${escaparHTML(noticia.titulo)}</div>
      <div class="url-meta-item"><strong>URL:</strong> <a href="${escaparHTML(noticia.url)}" target="_blank" rel="noopener">${escaparHTML(noticia.url)}</a></div>
      <div class="url-meta-item"><strong>Servicio:</strong> ${escaparHTML(noticia.readerUsado)} · <strong>Párrafos:</strong> ${noticia.nParrafos} · <strong>Caracteres:</strong> ${noticia.texto.length}</div>`;

    const textoDiv = document.getElementById("url-texto-extraido");
    const textoTrunc = noticia.texto.length > 600 ? noticia.texto.slice(0, 600) + "..." : noticia.texto;
    textoDiv.textContent = textoTrunc;

    document.getElementById("url-resultados").style.display = "block";

    const validacion = validarEsNoticia(noticia.texto);

    if (!validacion.esNoticia) {
      const advDiv = document.getElementById("url-advertencia");
      const senalesHTML = validacion.senales.length
        ? `<ul class="url-senales">${validacion.senales.map((s) => `<li>${escaparHTML(s)}</li>`).join("")}</ul>`
        : "";
      advDiv.innerHTML = `
        <div class="url-advertencia-box">
          <h4>Esta URL no parece ser una noticia</h4>
          <p>${escaparHTML(validacion.razon)}</p>
          ${senalesHTML}
          <div class="url-advertencia-stats">
            <span class="stat stat-bad">Marketing: ${validacion.nMarketing}</span>
            <span class="stat stat-good">Periodismo: ${validacion.nPeriodismo}</span>
            <span class="stat">Palabras: ${validacion.nPalabras}</span>
          </div>
          <button id="btn-forzar-analisis" class="btn btn-secondary">Analizar de todos modos</button>
        </div>`;
      advDiv.style.display = "block";
      document.getElementById("btn-forzar-analisis").onclick = async () => {
        advDiv.style.display = "none";
        mostrarEstado("Analizando texto (forzado)...", "info");
        await ejecutarAnalisis(noticia.texto);
        mostrarEstado("Análisis forzado completado.", "ok");
      };
      mostrarEstado(validacion.razon, "warn");
      return;
    }

    mostrarEstado("Analizando texto extraído...", "info");
    await ejecutarAnalisis(noticia.texto);
    mostrarEstado(`Análisis completo de: ${noticia.titulo.slice(0, 60)}`, "ok");
  } catch (e) {
    console.error(e);
    mostrarEstado("Error al procesar la URL: " + e.message, "error");
  } finally {
    setAnalizandoURL(false);
  }
}

/**
 * Muestra el workflow inmediatamente y ejecuta el pipeline.
 */
async function ejecutarAnalisis(texto) {
  mostrarEstado("Analizando...", "info");
  setAnalizando(true);

  // Mostrar pipeline YA (resultados solo al final).
  const vivos = el("analisis-vivo");
  if (vivos) vivos.style.display = "block";
  const resultados = el("resultados");
  if (resultados) resultados.style.display = "none";

  const detalle = el("resultados-detalle");
  if (detalle) detalle.style.display = "none";
  const consensoSec = el("consenso-section");
  if (consensoSec) consensoSec.style.display = "none";

  const hintTrunc = document.querySelector(".tab-panel.active .hint-truncado-electra");
  if (hintTrunc) hintTrunc.hidden = true;

  const pvContainer = el("pipeline-viz");
  if (pvContainer) {
    pv.render(pvContainer);
    pvContainer.style.display = "block";
    pv.reset();
    const nWords = texto.split(/\s+/).filter(Boolean).length;
    pv.setLiveData("input", { words: nWords, chars: texto.length });
    await pv.completarEtapa("input", `${nWords} palabras`);
    if (pv.isLearnMode()) pv.focusStage("input");
  }

  // Llevar el pipeline al viewport para ver el progreso en vivo (sobre todo en móvil).
  if (vivos) {
    const móvil = window.matchMedia("(max-width: 900px)").matches;
    vivos.scrollIntoView({ behavior: "smooth", block: móvil ? "start" : "nearest" });
  }

  let rTrans = null;

  try {
    await pv.iniciarEtapa("preprocess");
    const tokens = preprocess.preproceso(texto);
    const textoPrep = tokens.join(" ");
    const nTokens = tokens.length;
    pv.setLiveData("preprocess", {
      nTokens,
      sample: tokens.slice(0, 12),
    });
    await pv.completarEtapa("preprocess", `${nTokens} tokens`);

    await pv.iniciarEtapa("vectorize");
    const vector = vectorize.vectorizar(textoPrep);
    pv.setLiveData("vectorize", {
      nTerms: vector.size,
      topTerms: topTfIdfTerms(vector, 6),
    });
    await pv.completarEtapa("vectorize", `${vector.size} términos`);

    await pv.iniciarEtapa("nb");
    const rNB = nb.predecir(vector);
    pv.setLiveData("nb", { label: rNB.label, confidence: rNB.confidence });
    await pv.completarEtapa("nb", `${rNB.label} (${(rNB.confidence * 100).toFixed(0)}%)`);

    await pv.iniciarEtapa("logreg");
    const rLogReg = logreg.predecir(vector);
    pv.setLiveData("logreg", { label: rLogReg.label, confidence: rLogReg.confidence });
    await pv.completarEtapa("logreg", `${rLogReg.label} (${(rLogReg.confidence * 100).toFixed(0)}%)`);

    // Transformer con fallback (no bloquea el resto del pipeline).
    await pv.iniciarEtapa("transformer");
    try {
      if (!transformerListo.categoria) {
        mostrarEstado("Cargando ELECTRA (~14 MB)…", "info");
        await transformer.cargarClasificadorCategoria((prog, file) => {
          pv.actualizarProgreso("transformer", prog, file);
          actualizarProgresoCarga("ELECTRA", prog, file);
        });
        transformerListo.categoria = true;
      }
      rTrans = await transformer.predecirCategoria(texto);
      const badgeTrans = `${rTrans.label} (${(rTrans.confidence * 100).toFixed(0)}%)${
        rTrans.truncado ? " · trunc." : ""
      }`;
      pv.setLiveData("transformer", {
        label: rTrans.label,
        confidence: rTrans.confidence,
        truncado: !!rTrans.truncado,
      });
      await pv.completarEtapa("transformer", badgeTrans);
      if (rTrans.truncado) {
        mostrarHintTruncado();
      }
    } catch (e) {
      console.warn("Transformer categoría falló:", e);
      const corto = transformer.mensajeErrorCorto(e);
      pv.setLiveData("transformer", { error: corto, truncado: false });
      pv.errorEtapa("transformer", corto);
      rTrans = null;
      mostrarEstado(`ELECTRA no disponible (${corto}); se continúa con NB y LogReg.`, "warn");
    }

    await pv.iniciarEtapa("sensationalism");
    const rSens = sensationalism.analizar(texto);
    const sig = rSens.signals || {};
    pv.setLiveData("sensationalism", {
      label: rSens.label,
      score: rSens.score,
      signals: `clickbait×${(sig.clickbait_hits || []).length}, CAPS ${((sig.prop_mayusculas || 0) * 100).toFixed(0)}%, !×${sig.n_exclamaciones || 0}`,
    });
    await pv.completarEtapa("sensationalism", `${rSens.label} (${(rSens.score * 100).toFixed(0)}%)`);

    await pv.iniciarEtapa("sentiment");
    const rSentLex = sentimentLex.analizar(texto);
    let rSentONNX = null;
    try {
      if (!transformerListo.sentimiento) {
        mostrarEstado("Cargando RoBERTuito (~25 MB)…", "info");
        await transformer.cargarClasificadorSentimiento((prog, file) => {
          pv.actualizarProgreso("sentiment", prog, file);
          actualizarProgresoCarga("sentimiento", prog, file);
        });
        transformerListo.sentimiento = true;
      }
      rSentONNX = await transformer.predecirSentimiento(texto);
    } catch (e) {
      console.warn("Sentimiento ONNX falló, usando léxico:", e);
    }
    const sentFinal = rSentONNX || rSentLex;
    pv.setLiveData("sentiment", {
      label: sentFinal.label,
      source: rSentONNX ? "RoBERTuito ONNX" : "lexicon",
      lexLabel: rSentLex.label,
      onnxLabel: rSentONNX ? rSentONNX.label : null,
    });
    if (rSentONNX) {
      await pv.completarEtapa("sentiment", `${sentFinal.label}`);
    } else {
      pv.omitirEtapa("sentiment", `Léxico: ${sentFinal.label}`);
    }

    renderResultadosClasicos(texto, rNB, rLogReg, rSens, rSentLex);
    if (rTrans) {
      renderResultadosTransformer(rTrans, rSentONNX);
    } else if (rSentONNX) {
      renderResultadosTransformer(
        { label: rNB.label, confidence: rNB.confidence, scores: rNB.scores },
        rSentONNX
      );
      const ts = el("trans-section");
      if (ts) ts.style.display = "none";
    }

    await pv.iniciarEtapa("consenso");
    const catConsenso = categoriaConsenso(rNB, rLogReg, rTrans);
    const voteParts = [`NB=${rNB.label}`, `LR=${rLogReg.label}`];
    if (rTrans) voteParts.push(`ELECTRA=${rTrans.label}`);
    pv.setLiveData("consenso", {
      consensus: catConsenso,
      votes: voteParts.join(" · "),
      tone: rSens.label,
      sentiment: sentFinal.label,
    });
    renderConsenso(rNB, rLogReg, rTrans, rSens, sentFinal, catConsenso);
    await pv.completarEtapa("consenso", catConsenso);

    if (resultados) resultados.style.display = "block";
    if (detalle) detalle.style.display = "block";
    mostrarEstado("Análisis completo.", "ok");
  } catch (e) {
    console.error(e);
    mostrarEstado("Error en el análisis: " + e.message, "error");
  } finally {
    setAnalizando(false);
  }
}

/** Top TF-IDF terms for the Learn-mode live snapshot. */
function topTfIdfTerms(vector, n = 6) {
  const info = nb.getVocabInfo();
  if (!info?.vocabulary || !vector?.size) return [];
  const inv = {};
  for (const [term, idx] of Object.entries(info.vocabulary)) {
    inv[idx] = term;
  }
  return [...vector.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([idx, w]) => `${inv[idx] || `#${idx}`} (${w.toFixed(2)})`);
}

function categoriaConsenso(rNB, rLogReg, rTrans) {
  const votos = {};
  const labels = [rNB.label, rLogReg.label];
  if (rTrans) labels.push(rTrans.label);
  labels.forEach((c) => {
    votos[c] = (votos[c] || 0) + 1;
  });
  let consenso = rTrans ? rTrans.label : rNB.label;
  let maxVotos = 0;
  for (const [c, v] of Object.entries(votos)) {
    if (v > maxVotos) {
      maxVotos = v;
      consenso = c;
    }
  }
  return consenso;
}

function el(id) {
  const panelActivo = document.querySelector(".tab-panel.active") || document;
  return panelActivo.querySelector("#" + id) || document.getElementById(id);
}

function renderResultadosClasicos(texto, rNB, rLogReg, rSens, rSentLex) {
  const res = el("resultados");
  if (res) res.style.display = "block";
  const detalle = el("resultados-detalle");
  if (detalle) detalle.style.display = "block";

  const catLabel = el("cat-label");
  const catBar = el("cat-bar");
  catLabel.textContent = rNB.label;
  catLabel.style.color = COLORES_CATEGORIA[rNB.label] || "#0f172a";
  dibujarBarraConfianza(catBar, rNB.confidence, COLORES_CATEGORIA[rNB.label] || "#0f766e");

  dibujarBarras(el("cat-dist-nb"), rNB.scores, COLORES_CATEGORIA);
  dibujarBarras(el("cat-dist-logreg"), rLogReg.scores, COLORES_CATEGORIA);

  const tonoLabel = el("tono-label");
  tonoLabel.textContent = rSens.label === "sensacionalista" ? "Sensacionalista" : "Informativo";
  tonoLabel.style.color = rSens.label === "sensacionalista" ? "#be123c" : "#0f766e";
  el("tono-score").textContent = `Score: ${(rSens.score * 100).toFixed(0)}%`;
  const signals = rSens.signals;
  el("tono-signals").innerHTML = `
    <small>
      Clickbait: ${signals.clickbait_hits.length} ·
      Emocional: ${signals.emocional_hits.length} ·
      Exclamaciones: ${signals.n_exclamaciones} ·
      Mayúsculas: ${(signals.prop_mayusculas * 100).toFixed(0)}%
    </small>`;

  const sentLabel = el("sent-label");
  sentLabel.textContent = rSentLex.label;
  sentLabel.style.color = COLORES_SENTIMIENTO[rSentLex.label] || "#64748b";
  el("sent-score").textContent = `Intensidad: ${(rSentLex.score * 100).toFixed(0)}%`;
  dibujarDonut(
    el("sent-donut-lex"),
    {
      positivo: rSentLex.polaridad > 0 ? rSentLex.score : 0.1,
      negativo: rSentLex.polaridad < 0 ? rSentLex.score : 0.1,
      neutro: rSentLex.label === "neutro" ? 1 : 0.2,
    },
    COLORES_SENTIMIENTO,
    rSentLex.label
  );
}

function renderResultadosTransformer(rTrans, rSentONNX) {
  if (rTrans && el("trans-section")) {
    const transLabel = el("trans-label");
    transLabel.textContent = rTrans.label;
    transLabel.style.color = COLORES_CATEGORIA[rTrans.label] || "#0f172a";
    dibujarBarraConfianza(
      el("trans-bar"),
      rTrans.confidence,
      COLORES_CATEGORIA[rTrans.label] || "#0f766e"
    );
    dibujarBarras(el("cat-dist-trans"), rTrans.scores, COLORES_CATEGORIA);
    el("trans-section").style.display = "block";
  }

  if (rSentONNX) {
    const sentONNXLabel = el("sent-onnx-label");
    sentONNXLabel.textContent = rSentONNX.label;
    sentONNXLabel.style.color = COLORES_SENTIMIENTO[rSentONNX.label] || "#64748b";
    el("sent-onnx-score").textContent = `Confianza: ${(rSentONNX.confidence * 100).toFixed(0)}%`;
    dibujarDonut(el("sent-donut-onnx"), rSentONNX.scores, COLORES_SENTIMIENTO, rSentONNX.label);
    el("sent-onnx-section").style.display = "block";
  }
}

function renderConsenso(rNB, rLogReg, rTrans, rSens, rSent, consenso) {
  const trLabel = rTrans ? rTrans.label : "—";
  const cons = el("consenso");
  cons.innerHTML = `
    <div class="consenso-hero">
      <span class="consenso-hero-label">Veredicto NLP · Categoría</span>
      <span class="consenso-hero-valor" style="color:${COLORES_CATEGORIA[consenso] || "#0f172a"}">${consenso}</span>
    </div>
    <div class="consenso-grid">
      <div class="consenso-item">
        <span class="consenso-label">Votos</span>
        <span class="consenso-extra">NB ${rNB.label} · LR ${rLogReg.label} · TR ${trLabel}</span>
      </div>
      <div class="consenso-item">
        <span class="consenso-label">Tono</span>
        <span class="consenso-valor" style="color:${rSens.label === "sensacionalista" ? "#be123c" : "#0f766e"}">${rSens.label}</span>
      </div>
      <div class="consenso-item">
        <span class="consenso-label">Sentimiento</span>
        <span class="consenso-valor" style="color:${COLORES_SENTIMIENTO[rSent.label]}">${rSent.label}</span>
      </div>
    </div>`;
  const sec = el("consenso-section");
  if (sec) sec.style.display = "block";
}

async function analizarCSV(file) {
  mostrarEstado("Procesando CSV...", "info");
  const text = await file.text();
  const lineas = text.split(/\r?\n/).filter((l) => l.trim());
  if (lineas.length < 2) {
    mostrarEstado("El CSV necesita cabecera + al menos 1 fila.", "warn");
    return;
  }

  const cabecera = lineas[0].split(",").map((c) => c.trim().toLowerCase());
  const textoIdx = cabecera.findIndex((c) => c.includes("text") || c.includes("texto"));
  const textos = [];
  for (let i = 1; i < lineas.length; i++) {
    let linea = lineas[i];
    if (textoIdx > 0) {
      const partes = linea.split(",");
      textos.push(partes.slice(textoIdx).join(","));
    } else {
      textos.push(linea);
    }
  }

  const resultados = textos.map((t) => {
    const prep = preprocess.preprocesoTexto(t);
    const vec = vectorize.vectorizar(prep);
    const rNB = nb.predecir(vec);
    const rSens = sensationalism.analizar(t);
    const rSent = sentimentLex.analizar(t);
    return {
      texto: t,
      categoria: rNB.label,
      tono: rSens.label,
      sentimiento: rSent.label,
    };
  });

  renderResumenLote(resultados);
  mostrarEstado(`CSV procesado: ${resultados.length} noticias.`, "ok");
}

function renderResumenLote(resultados) {
  document.getElementById("lote-resultados").style.display = "block";

  const catDist = {};
  const sentDist = {};
  const tonoDist = {};
  resultados.forEach((r) => {
    catDist[r.categoria] = (catDist[r.categoria] || 0) + 1;
    sentDist[r.sentimiento] = (sentDist[r.sentimiento] || 0) + 1;
    tonoDist[r.tono] = (tonoDist[r.tono] || 0) + 1;
  });

  const norm = (d) => {
    const t = resultados.length;
    const out = {};
    for (const [k, v] of Object.entries(d)) out[k] = v / t;
    return out;
  };

  dibujarBarras(document.getElementById("lote-cat-dist"), norm(catDist), COLORES_CATEGORIA);
  dibujarDonut(
    document.getElementById("lote-sent-donut"),
    norm(sentDist),
    COLORES_SENTIMIENTO,
    `${resultados.length}`
  );
  dibujarDonut(
    document.getElementById("lote-tono-donut"),
    norm(tonoDist),
    { informativo: "#0f766e", sensacionalista: "#be123c" },
    ""
  );

  const tabla = document.getElementById("lote-tabla");
  tabla.innerHTML = `
    <table>
      <thead><tr><th>#</th><th>Texto</th><th>Categoría</th><th>Tono</th><th>Sent.</th></tr></thead>
      <tbody>
        ${resultados
          .slice(0, 20)
          .map(
            (r, i) => `
          <tr>
            <td>${i + 1}</td>
            <td title="${r.texto.replace(/"/g, "&quot;")}">${r.texto.slice(0, 60)}...</td>
            <td style="color:${COLORES_CATEGORIA[r.categoria]}">${r.categoria}</td>
            <td style="color:${r.tono === "sensacionalista" ? "#be123c" : "#0f766e"}">${r.tono}</td>
            <td style="color:${COLORES_SENTIMIENTO[r.sentimiento]}">${r.sentimiento}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>`;
}

async function cargarMetricas() {
  try {
    const resp = await fetch("assets/metrics.json");
    const data = await resp.json();
    const cont = document.getElementById("metricas-tabla");
    if (!cont) return;
    cont.innerHTML = `
      <table>
        <thead>
          <tr><th>Modelo</th><th>Tipo</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        </thead>
        <tbody>
          ${data.modelos
            .map(
              (m) => `
            <tr>
              <td>${m.nombre}</td>
              <td>${m.tipo}</td>
              <td class="num">${(m.accuracy * 100).toFixed(2)}%</td>
              <td class="num">${(m.precision_macro * 100).toFixed(2)}%</td>
              <td class="num">${(m.recall_macro * 100).toFixed(2)}%</td>
              <td class="num"><strong>${(m.f1_macro * 100).toFixed(2)}%</strong></td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>
      <p class="muted">Set de test: ${data.n_test} noticias · ${data.labels.length} categorías</p>`;

    const mats = document.getElementById("matrices");
    mats.innerHTML = "";
    for (const detalle of data.detalle) {
      const nombre = detalle.nombre.split(" ")[0].toLowerCase();
      const div = document.createElement("div");
      div.className = "matriz-item";
      const img = `assets/confusion_${
        nombre.includes("naive") ? "nb" : nombre.includes("logistic") ? "logreg" : "transformer"
      }.png`;
      div.innerHTML = `<h4>${detalle.nombre}</h4><img src="${img}" alt="Matriz ${detalle.nombre}" onerror="this.style.display='none'"/>`;
      mats.appendChild(div);
    }
  } catch (e) {
    console.warn("No se pudieron cargar métricas:", e);
  }
}

function mostrarEstado(msg, tipo = "info") {
  const panel = document.querySelector(".tab-panel.active") || document;
  const nodo =
    panel.querySelector(".estado") ||
    document.getElementById("estado") ||
    document.querySelector(".estado");
  if (!nodo) return;
  nodo.textContent = msg;
  nodo.className = "estado " + tipo;
  nodo.style.display = "block";
}

function setAnalizando(b) {
  const btn = document.getElementById("btn-analizar");
  if (btn) {
    btn.disabled = b;
    btn.textContent = b ? "Analizando…" : "Analizar noticia";
  }
}

function setAnalizandoURL(b) {
  const btn = document.getElementById("btn-analizar-url");
  if (btn) {
    btn.disabled = b;
    btn.textContent = b ? "Procesando…" : "Analizar URL";
  }
}

function escaparHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function actualizarProgresoCarga(tipo, prog, file) {
  const corto = file ? String(file).split("/").pop() : "";
  mostrarEstado(`Cargando ${tipo}${corto ? `: ${corto}` : ""} (${prog.toFixed(0)}%)`, "info");
}

/** Badge discreto si ELECTRA truncó el texto (artículos largos vía URL). */
function mostrarHintTruncado() {
  const panel = document.querySelector(".tab-panel.active") || document;
  let hint = panel.querySelector(".hint-truncado-electra");
  if (!hint) {
    const vivo = panel.querySelector("#analisis-vivo") || panel.querySelector(".analisis-vivo");
    if (!vivo) return;
    hint = document.createElement("p");
    hint.className = "hint-truncado-electra";
    vivo.appendChild(hint);
  }
  hint.textContent = "Texto truncado para ELECTRA (máx. ~512 tokens)";
  hint.hidden = false;
}

function mostrarBotonesEjemplos() {
  const cont = document.getElementById("ejemplos");
  if (!ejemplosNoticias.length) return;
  cont.innerHTML = '<span class="ejemplos-label">Ejemplos</span>';
  ejemplosNoticias.slice(0, 6).forEach((n) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-ejemplo";
    btn.textContent = `${n.categoria} · ${n.tono === "sensacionalista" ? "S" : "I"}`;
    btn.title = n.texto.slice(0, 80);
    btn.onclick = () => {
      document.getElementById("input-texto").value = n.texto;
    };
    cont.appendChild(btn);
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("btn-analizar").addEventListener("click", analizarTexto);
  document.getElementById("input-texto").addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") analizarTexto();
  });

  const btnURL = document.getElementById("btn-analizar-url");
  if (btnURL) btnURL.addEventListener("click", analizarURL);
  const inputURL = document.getElementById("input-url");
  if (inputURL) {
    inputURL.addEventListener("keydown", (e) => {
      if (e.key === "Enter") analizarURL();
    });
  }

  const csvInput = document.getElementById("csv-input");
  csvInput.addEventListener("change", (e) => {
    if (e.target.files[0]) analizarCSV(e.target.files[0]);
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.target).classList.add("active");
      if (tab.dataset.target === "panel-notebook") {
        ensureNotebookLoaded();
      }
    });
  });

  await inicializar();
  await cargarMetricas();
});
