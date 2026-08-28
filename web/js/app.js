// app.js — Lógica principal de la interfaz del Clasificador de Noticias.
// Orquesta: preprocess → vectorize → NB → LogReg → transformer → tono → sentimiento → consenso.

import * as preprocess from "./preprocess.js";
import * as vectorize from "./vectorize.js";
import * as nb from "./naive_bayes.js";
import * as logreg from "./logreg.js";
import * as sensationalism from "./sensationalism.js";
import * as sentimentLex from "./sentiment_lexicon.js";
import * as lda from "./lda.js";
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

// Paleta fija para los 7 temas LDA (tonos desaturados para no competir con las categorías).
const COLORES_TEMAS = {
  "Tema 0": "#0f766e",
  "Tema 1": "#2563eb",
  "Tema 2": "#7c3aed",
  "Tema 3": "#0891b2",
  "Tema 4": "#ea580c",
  "Tema 5": "#db2777",
  "Tema 6": "#65a30d",
};

let inicializado = false;
let modelosCargados = { nb: false, logreg: false, lexico: false, reglas: false, lda: false };
let transformerListo = { categoria: false, sentimiento: false };
let cargandoSentimiento = false;
let ejemplosNoticias = [];
let precargaIniciada = false;

async function inicializar() {
  if (inicializado) return;
  mostrarEstado("Cargando modelos clásicos...", "info");

  try {
    const [nbData, logregData, stopw, lexic, reglas, temas, ejemplos] = await Promise.all([
      nb.cargarNB(),
      logreg.cargarLogReg(),
      preprocess.cargarStopwords(),
      sentimentLex.cargarLexico(),
      sensationalism.cargarReglas(),
      // LDA es complementario: si el JSON falla, el análisis sigue sin temas.
      lda.cargarLDA().catch((e) => {
        console.warn("Temas LDA no disponibles:", e);
        return null;
      }),
      cargarEjemplos(),
    ]);

    const vi = nb.getVocabInfo();
    vectorize.setVocab(vi.vocabulary, vi.idf, vi.ngram_range);
    preprocess.construirLemmaMap(vi.vocabulary);

    modelosCargados = {
      nb: true,
      logreg: true,
      lexico: true,
      reglas: true,
      lda: temas !== null,
    };
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
    actualizarBotonPrecargaSentimiento();
  } catch (e) {
    console.error(e);
    const msg = "Error cargando modelos: " + e.message;
    mostrarEstado(msg, "error");
    anunciar(msg);
    // Sin los modelos clásicos la app no sirve: ofrecer reintento sin recargar.
    const panel = document.querySelector(".tab-panel.active") || document;
    const nodo = panel.querySelector(".estado") || document.getElementById("estado");
    if (nodo && !nodo.querySelector(".btn-reintentar")) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-secondary btn-reintentar";
      btn.textContent = "Reintentar";
      btn.addEventListener("click", () => {
        btn.remove();
        inicializar();
      });
      nodo.appendChild(btn);
    }
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

/**
 * Precarga en segundo plano SOLO ELECTRA (~14 MB).
 * RoBERTuito (~25 MB) se deja para el primer análisis o para el botón de precarga:
 * es la mejora de carga diferida prevista en docs/informe.md ("lazy-loading
 * diferido del modelo de sentimiento") y recorta ~25 MB de la primera visita.
 */
async function precargarModelosONNX() {
  if (transformerListo.categoria) return;
  try {
    mostrarEstado("Precargando ELECTRA en segundo plano (~14 MB)…", "info");
    await transformer.cargarClasificadorCategoria((prog, file) => {
      mostrarEstado(`Precargando ELECTRA: ${prog.toFixed(0)}%`, "info");
      actualizarProgresoCarga("ELECTRA", prog, file, { silencioso: true });
    });
    transformerListo.categoria = true;
    ocultarProgresoCarga();
    mostrarEstado("ELECTRA listo. Puedes analizar.", "ok");
  } catch (e) {
    console.warn("Precarga ELECTRA falló:", e);
    ocultarProgresoCarga();
    mostrarEstado("Listo (ELECTRA se cargará al analizar).", "ok");
  }
  actualizarBotonPrecargaSentimiento();
}

/**
 * Descarga el modelo de sentimiento RoBERTuito (~25 MB) a petición.
 * Devuelve true si quedó listo (ya estaba, o se cargó ahora).
 */
async function cargarModeloSentimiento() {
  if (transformerListo.sentimiento) return true;
  try {
    await transformer.cargarClasificadorSentimiento((prog, file) => {
      actualizarProgresoCarga("sentimiento", prog, file);
    });
    transformerListo.sentimiento = true;
    return true;
  } catch (e) {
    console.warn("Carga de RoBERTuito falló (se usará léxico si hace falta):", e);
    return false;
  } finally {
    ocultarProgresoCarga();
    actualizarBotonPrecargaSentimiento();
  }
}

/**
 * Estado del botón de precarga del modelo de sentimiento.
 * Se oculta cuando ya está descargado (o si no existe el botón).
 */
function actualizarBotonPrecargaSentimiento() {
  const btn = document.getElementById("btn-precargar-sentimiento");
  if (!btn) return;
  if (transformerListo.sentimiento) {
    btn.style.display = "none";
    return;
  }
  btn.style.display = "inline-flex";
  btn.disabled = cargandoSentimiento;
  btn.textContent = cargandoSentimiento
    ? "Precargando RoBERTuito…"
    : "Precargar modelo de sentimiento (~25 MB)";
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

    // Temas LDA (Semana 3): asignación por similitud coseno contra los temas precomputados.
    let rTema = null;
    if (modelosCargados.lda) {
      await pv.iniciarEtapa("temas");
      try {
        rTema = lda.predecirTema(tokens);
        pv.setLiveData("temas", {
          topicId: rTema.topicId,
          similarity: rTema.similarity,
          topWords: rTema.topWords.map(([w]) => w).slice(0, 5),
        });
        await pv.completarEtapa("temas", `Tema ${rTema.topicId}`);
      } catch (e) {
        console.warn("Temas LDA falló:", e);
        pv.errorEtapa("temas", "LDA");
      }
    } else {
      pv.omitirEtapa("temas", "LDA no disponible");
    }

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
          actualizarProgresoCarga("sentimiento", prog, file, { silencioso: true });
        });
        transformerListo.sentimiento = true;
        actualizarBotonPrecargaSentimiento();
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

    renderResultadosClasicos(texto, rNB, rLogReg, rSens, rSentLex, rTema);
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
    const veredicto = categoriaConsenso(rNB, rLogReg, rTrans);
    const catConsenso = veredicto.label;
    const voteParts = [`NB=${rNB.label}`, `LR=${rLogReg.label}`];
    if (rTrans) voteParts.push(`ELECTRA=${rTrans.label}`);
    pv.setLiveData("consenso", {
      consensus: catConsenso,
      votes: voteParts.join(" · "),
      reason: veredicto.reason,
      tone: rSens.label,
      sentiment: sentFinal.label,
    });
    renderConsenso(rNB, rLogReg, rTrans, rSens, sentFinal, veredicto);
    await pv.completarEtapa("consenso", catConsenso);

    if (resultados) resultados.style.display = "block";
    if (detalle) detalle.style.display = "block";
    mostrarEstado("Análisis completo.", "ok");
    anunciar(
      `Análisis completo. Categoría: ${catConsenso}. Tono: ${rSens.label}. Sentimiento: ${sentFinal.label}.`
    );
  } catch (e) {
    console.error(e);
    mostrarEstado("Error en el análisis: " + e.message, "error");
    anunciar("Error en el análisis: " + e.message);
  } finally {
    ocultarProgresoCarga();
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

/**
 * Margen entre la clase ganadora y la 2ª (0–1). Mayor = predicción más clara.
 */
function margenConfianza(scores, label) {
  if (!scores || typeof scores !== "object") return 0;
  const vals = Object.entries(scores)
    .map(([k, v]) => ({ k, v: Number(v) || 0 }))
    .sort((a, b) => b.v - a.v);
  if (!vals.length) return 0;
  const top = vals[0].k === label ? vals[0].v : Number(scores[label]) || 0;
  const second = vals.find((x) => x.k !== label)?.v ?? 0;
  return Math.max(0, Math.min(1, top - second));
}

/**
 * Consenso inteligente de categoría.
 * Mezcla las distribuciones de probabilidad ponderadas por fiabilidad del modelo,
 * refuerza el acuerdo NB↔LR y da más peso a ELECTRA cuando es claro y confiable.
 * Devuelve { label, reason, blend }.
 */
function categoriaConsenso(rNB, rLogReg, rTrans) {
  const modelos = [
    { id: "NB", pred: rNB, pesoBase: 1.0 },
    { id: "LR", pred: rLogReg, pesoBase: 0.9 },
  ];
  if (rTrans) modelos.push({ id: "ELECTRA", pred: rTrans, pesoBase: 1.85 });

  const blend = {};
  const aportes = [];

  for (const m of modelos) {
    const conf = Math.max(0, Math.min(1, Number(m.pred.confidence) || 0));
    const margin = margenConfianza(m.pred.scores, m.pred.label);
    let peso = m.pesoBase * (0.55 + 0.45 * conf) * (0.7 + 0.3 * margin);

    // ELECTRA muy seguro: amplificar (cubre NB+LR alineados pero equivocados).
    if (m.id === "ELECTRA" && conf >= 0.8) peso *= 1.45;
    // ELECTRA dudoso: atenuar y dejar más voz a los clásicos.
    if (m.id === "ELECTRA" && conf < 0.45) peso *= 0.55;

    const dist = m.pred.scores && typeof m.pred.scores === "object"
      ? m.pred.scores
      : { [m.pred.label]: conf || 1 };

    for (const [cat, p] of Object.entries(dist)) {
      const mass = peso * (Number(p) || 0);
      blend[cat] = (blend[cat] || 0) + mass;
    }
    aportes.push({
      id: m.id,
      label: m.pred.label,
      conf,
      peso: Math.round(peso * 100) / 100,
    });
  }

  // Acuerdo clásico NB=LR: empujón moderado a esa categoría.
  if (rNB.label === rLogReg.label) {
    const avg = ((Number(rNB.confidence) || 0) + (Number(rLogReg.confidence) || 0)) / 2;
    blend[rNB.label] = (blend[rNB.label] || 0) + 0.4 * avg;
  }

  // Acuerdo total de los tres: refuerzo fuerte.
  if (rTrans && rNB.label === rLogReg.label && rLogReg.label === rTrans.label) {
    blend[rTrans.label] = (blend[rTrans.label] || 0) + 0.6;
  }

  let label = rNB.label;
  let best = -Infinity;
  for (const [cat, v] of Object.entries(blend)) {
    if (v > best) {
      best = v;
      label = cat;
    }
  }

  const electra = aportes.find((a) => a.id === "ELECTRA");
  const clasicos = aportes.filter((a) => a.id !== "ELECTRA");
  const mismosClasicos = rNB.label === rLogReg.label;
  let reason;
  if (electra && label === electra.label && (!mismosClasicos || rNB.label !== electra.label)) {
    reason =
      electra.conf >= 0.8
        ? `ELECTRA ${(electra.conf * 100).toFixed(0)}% (peso alto)`
        : `Mezcla ponderada → ${label} (lidera ELECTRA)`;
  } else if (mismosClasicos && label === rNB.label && (!electra || electra.label !== label)) {
    reason = `NB+LR de acuerdo (${rNB.label}) frente a ELECTRA`;
  } else if (mismosClasicos && electra && electra.label === label) {
    reason = "Unanimidad de modelos";
  } else {
    reason = `Mezcla ponderada · ${aportes.map((a) => `${a.id}=${a.label}`).join(" · ")}`;
  }

  return { label, reason, blend, aportes };
}

function el(id) {
  const panelActivo = document.querySelector(".tab-panel.active") || document;
  if (panelActivo.id === "panel-url") {
    const urlId = id === "resultados" ? "url-analisis-resultados" : `url-${id}`;
    return (
      panelActivo.querySelector("#" + urlId) ||
      document.getElementById(urlId) ||
      panelActivo.querySelector("#" + id) ||
      document.getElementById(id)
    );
  }
  return panelActivo.querySelector("#" + id) || document.getElementById(id);
}

function renderResultadosClasicos(texto, rNB, rLogReg, rSens, rSentLex, rTema) {
  const res = el("resultados");
  if (res) res.style.display = "block";
  const detalle = el("resultados-detalle");
  if (detalle) detalle.style.display = "block";

  renderTemas(rTema);

  const catLabel = el("cat-label");
  const catBar = el("cat-bar");
  catLabel.textContent = rNB.label;
  catLabel.style.color = COLORES_CATEGORIA[rNB.label] || cssVar("--ink", "#0f172a");
  dibujarBarraConfianza(
    catBar,
    rNB.confidence,
    COLORES_CATEGORIA[rNB.label] || cssVar("--accent", "#0f766e")
  );

  dibujarBarras(el("cat-dist-nb"), rNB.scores, COLORES_CATEGORIA);
  dibujarBarras(el("cat-dist-logreg"), rLogReg.scores, COLORES_CATEGORIA);

  const tonoLabel = el("tono-label");
  tonoLabel.textContent = rSens.label === "sensacionalista" ? "Sensacionalista" : "Informativo";
  tonoLabel.style.color = colorTono()[rSens.label === "sensacionalista" ? "sensacionalista" : "informativo"];
  el("tono-score").textContent = `Sensacionalismo: ${(rSens.score * 100).toFixed(0)}% · ${
    rSens.label === "sensacionalista" ? "señales de clickbait" : "sin señales de clickbait"
  }`;
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

/**
 * Card "Temas (LDA)": tema dominante + top palabras + distribución de similitudes.
 * Sin LDA cargado la card se oculta (no bloquea el resto de resultados).
 */
function renderTemas(rTema) {
  const sec = el("tema-section");
  if (!sec) return;
  if (!rTema) {
    sec.style.display = "none";
    return;
  }

  const dist = {};
  rTema.distribution.forEach((p, i) => {
    dist[`Tema ${i}`] = p;
  });
  dibujarBarras(el("tema-dist"), dist, COLORES_TEMAS);

  const topWords = rTema.topWords.map(([w]) => w);
  el("tema-topwords").textContent = topWords.join(" · ");
  el("tema-sim").textContent = `Similitud coseno: ${(rTema.similarity * 100).toFixed(0)}%`;

  const label = el("tema-label");
  label.textContent = `Tema ${rTema.topicId}`;
  label.style.color = COLORES_TEMAS[`Tema ${rTema.topicId}`] || cssVar("--accent", "#0f766e");
  sec.style.display = "block";
}

function renderResultadosTransformer(rTrans, rSentONNX) {
  if (rTrans && el("trans-section")) {
    const transLabel = el("trans-label");
    transLabel.textContent = rTrans.label;
    transLabel.style.color = COLORES_CATEGORIA[rTrans.label] || cssVar("--ink", "#0f172a");
    dibujarBarraConfianza(
      el("trans-bar"),
      rTrans.confidence,
      COLORES_CATEGORIA[rTrans.label] || cssVar("--accent", "#0f766e")
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

function renderConsenso(rNB, rLogReg, rTrans, rSens, rSent, veredicto) {
  const consenso = typeof veredicto === "string" ? veredicto : veredicto.label;
  const reason = typeof veredicto === "object" && veredicto.reason ? veredicto.reason : "";
  const trLabel = rTrans ? rTrans.label : "—";
  const trConf = rTrans ? ` ${(rTrans.confidence * 100).toFixed(0)}%` : "";
  const cons = el("consenso");
  cons.innerHTML = `
    <div class="consenso-hero">
      <span class="consenso-hero-label">Veredicto NLP · Categoría</span>
      <span class="consenso-hero-valor" style="color:${COLORES_CATEGORIA[consenso] || cssVar("--ink", "#0f172a")}">${consenso}</span>
      ${reason ? `<span class="consenso-hero-reason muted">${reason}</span>` : ""}
    </div>
    <div class="consenso-grid">
      <div class="consenso-item">
        <span class="consenso-label">Modelos</span>
        <span class="consenso-extra">NB ${rNB.label} ${(rNB.confidence * 100).toFixed(0)}% · LR ${rLogReg.label} ${(rLogReg.confidence * 100).toFixed(0)}% · TR ${trLabel}${trConf}</span>
      </div>
      <div class="consenso-item">
        <span class="consenso-label">Tono</span>
        <span class="consenso-valor" style="color:${colorTono()[rSens.label === "sensacionalista" ? "sensacionalista" : "informativo"]}">${rSens.label}</span>
      </div>
      <div class="consenso-item">
        <span class="consenso-label">Sentimiento</span>
        <span class="consenso-valor" style="color:${COLORES_SENTIMIENTO[rSent.label]}">${rSent.label}</span>
      </div>
    </div>`;
  const sec = el("consenso-section");
  if (sec) sec.style.display = "block";
}

/**
 * Parser CSV con soporte de comillas dobles, comillas escapadas ("")
 * y celdas multilínea (RFC 4180). Devuelve array de filas (array de celdas).
 */
function parsearCSV(texto) {
  const filas = [];
  let fila = [];
  let celda = "";
  let entreComillas = false;

  for (let i = 0; i < texto.length; i++) {
    const ch = texto[i];
    if (entreComillas) {
      if (ch === '"') {
        if (texto[i + 1] === '"') {
          celda += '"';
          i++;
        } else {
          entreComillas = false;
        }
      } else {
        celda += ch;
      }
      continue;
    }
    if (ch === '"') {
      entreComillas = true;
    } else if (ch === ",") {
      fila.push(celda);
      celda = "";
    } else if (ch === "\n") {
      fila.push(celda);
      celda = "";
      filas.push(fila);
      fila = [];
    } else if (ch === "\r") {
      // ignorado: \r\n y \r suelto se tratan como fin de línea
      if (texto[i + 1] !== "\n") {
        fila.push(celda);
        celda = "";
        filas.push(fila);
        fila = [];
      }
    } else {
      celda += ch;
    }
  }
  if (celda !== "" || fila.length) {
    fila.push(celda);
    filas.push(fila);
  }
  return filas.filter((f) => f.some((c) => c.trim()));
}

async function analizarCSV(file) {
  mostrarEstado("Procesando CSV...", "info");
  const text = await file.text();
  const filas = parsearCSV(text);
  if (filas.length < 2) {
    mostrarEstado("El CSV necesita cabecera + al menos 1 fila.", "warn");
    return;
  }

  const cabecera = filas[0].map((c) => c.trim().toLowerCase());
  const textoIdx = cabecera.findIndex((c) => c.includes("text") || c.includes("texto"));
  if (textoIdx === -1) {
    mostrarEstado('El CSV debe incluir una columna «texto» (o «text»).', "warn");
    return;
  }

  const textos = filas.slice(1).map((fila) => (fila[textoIdx] ?? "").trim()).filter(Boolean);
  if (!textos.length) {
    mostrarEstado("No se encontró texto en la columna «texto».", "warn");
    return;
  }

  // Procesado por chunks: cede el hilo principal entre tandas para que la UI
  // siga respondiendo con CSVs grandes y se pueda mostrar el progreso.
  const TAMANO_TANDA = 25;
  const resultados = [];
  const progreso = document.getElementById("progreso-modelos");

  try {
    for (let i = 0; i < textos.length; i += TAMANO_TANDA) {
      const tanda = textos.slice(i, i + TAMANO_TANDA);
      for (const t of tanda) {
        const prep = preprocess.preprocesoTexto(t);
        const vec = vectorize.vectorizar(prep);
        const rNB = nb.predecir(vec);
        const rLogReg = logreg.predecir(vec);
        const rTema = modelosCargados.lda ? lda.predecirTema(prep) : null;
        const rSens = sensationalism.analizar(t);
        const rSent = sentimentLex.analizar(t);
        resultados.push({
          texto: t,
          categoria: rNB.label,
          logreg: rLogReg.label,
          tema: rTema ? `Tema ${rTema.topicId}` : "—",
          tono: rSens.label,
          sentimiento: rSent.label,
        });
      }

      const hechos = Math.min(i + TAMANO_TANDA, textos.length);
      if (progreso) {
        actualizarProgresoLote(progreso, hechos, textos.length);
      } else {
        mostrarEstado(`Procesando CSV: ${hechos}/${textos.length}…`, "info");
      }
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (e) {
    console.error(e);
    mostrarEstado("Error procesando el CSV: " + e.message, "error");
    anunciar("Error procesando el CSV: " + e.message);
    return;
  } finally {
    ocultarProgresoCarga();
  }

  renderResumenLote(resultados);
  mostrarEstado(`CSV procesado: ${resultados.length} noticias.`, "ok");
  anunciar(`CSV procesado: ${resultados.length} noticias.`);
}

/** Barra de progreso del modo lote, reutilizando el contenedor de modelos. */
function actualizarProgresoLote(cont, hechos, total) {
  const pct = total ? (hechos / total) * 100 : 100;
  const etiqueta = `Procesando CSV: ${hechos} de ${total} noticias`;
  cont.innerHTML = `
    <div class="progreso-modelos-head">
      <span class="progreso-modelos-label">${etiqueta}</span>
      <span class="progreso-modelos-pct">${pct.toFixed(0)}%</span>
    </div>
    <div
      class="progreso-modelos-track"
      role="progressbar"
      aria-label="${etiqueta}"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow="${pct.toFixed(0)}"
    >
      <div class="progreso-modelos-fill" style="width:${pct}%"></div>
    </div>`;
  cont.style.display = "block";
}

/** Descarga los resultados del lote como CSV (con comillas y escape RFC 4180). */
function descargarLoteCSV(resultados) {
  const celda = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const lineas = [
    ["id", "texto", "categoria_nb", "categoria_logreg", "tema_lda", "tono", "sentimiento"]
      .join(","),
    ...resultados.map((r, i) =>
      [i, r.texto, r.categoria, r.logreg, r.tema, r.tono, r.sentimiento].map(celda).join(",")
    ),
  ];
  const blob = new Blob(["﻿" + lineas.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "resultados_lote.csv";
  a.click();
  URL.revokeObjectURL(url);
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
    colorTono(),
    ""
  );

  const tabla = document.getElementById("lote-tabla");
  const tituloDetalle =
    resultados.length > 20
      ? `Detalle (primeras 20 de ${resultados.length})`
      : "Detalle (primeras 20)";
  const head = document.querySelector("#panel-lote .lote-head h3");
  if (head) head.textContent = tituloDetalle;
  tabla.innerHTML = `
    <table>
      <thead><tr><th>#</th><th>Texto</th><th>NB</th><th>LogReg</th><th>Tema LDA</th><th>Tono</th><th>Sent.</th></tr></thead>
      <tbody>
        ${resultados
          .slice(0, 20)
          .map((r, i) => {
            const resumen = escaparHTML(r.texto.slice(0, 60)) + (r.texto.length > 60 ? "..." : "");
            return `
          <tr>
            <td>${i + 1}</td>
            <td title="${escaparHTML(r.texto)}">${resumen}</td>
            <td style="color:${COLORES_CATEGORIA[r.categoria]}">${escaparHTML(r.categoria)}</td>
            <td style="color:${COLORES_CATEGORIA[r.logreg]}">${escaparHTML(r.logreg)}</td>
            <td class="muted">${escaparHTML(r.tema)}</td>
            <td style="color:${colorTono()[r.tono === "sensacionalista" ? "sensacionalista" : "informativo"]}">${escaparHTML(r.tono)}</td>
            <td style="color:${COLORES_SENTIMIENTO[r.sentimiento]}">${escaparHTML(r.sentimiento)}</td>
          </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;

  const btnDescarga = document.getElementById("lote-descargar");
  if (btnDescarga) {
    btnDescarga.style.display = "inline-flex";
    btnDescarga.onclick = () => descargarLoteCSV(resultados);
  }
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

/**
 * Anuncia un mensaje en la región viva global (para lectores de pantalla).
 * Se vacía antes de escribir para que dos mensajes iguales seguidos se anuncien igual.
 */
function anunciar(msg) {
  const nodo = document.getElementById("anuncio-live");
  if (!nodo) return;
  nodo.textContent = "";
  window.setTimeout(() => {
    nodo.textContent = msg;
  }, 60);
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

/**
 * Resuelve una custom property de :root (con alternativa para el caso raro de
 * que no exista). Se consulta al renderizar, así que respeta el tema activo.
 */
function cssVar(nombre, alternativa) {
  const valor = getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
  return valor || alternativa;
}

/** Colores de tono (sensacionalista / informativo) según el tema activo. */
const colorTono = () => ({
  sensacionalista: cssVar("--danger", "#be123c"),
  informativo: cssVar("--accent", "#0f766e"),
});

/**
 * Progreso de descarga de modelos: barra visual global (#progreso-modelos)
 * + mensaje de estado, salvo que se pida silencio (precarga en segundo plano).
 * Con progreso indeterminado (prog == null) la barra se anima sin porcentaje.
 */
function actualizarProgresoCarga(tipo, prog, file, { silencioso = false } = {}) {
  const cont = document.getElementById("progreso-modelos");
  const corto = file ? String(file).split("/").pop() : "";
  const etiqueta = `Cargando ${tipo}${corto ? `: ${corto}` : ""}`;
  const indefinido = prog == null;
  const pct = indefinido ? null : Math.max(0, Math.min(100, prog));

  if (!cont) {
    if (!silencioso) mostrarEstado(`${etiqueta} (${prog?.toFixed(0) ?? "…"}%)`, "info");
    return;
  }

  const valor = pct == null ? "" : `<span class="progreso-modelos-pct">${pct.toFixed(0)}%</span>`;
  cont.innerHTML = `
    <div class="progreso-modelos-head">
      <span class="progreso-modelos-label">${etiqueta}</span>
      ${valor}
    </div>
    <div
      class="progreso-modelos-track${indefinido ? " indeterminado" : ""}"
      role="progressbar"
      aria-label="${etiqueta}"
      aria-valuemin="0"
      aria-valuemax="100"
      ${pct == null ? "" : `aria-valuenow="${pct.toFixed(0)}"`}
    >
      <div class="progreso-modelos-fill" style="${pct == null ? "" : `width:${pct}%`}"></div>
    </div>`;
  cont.style.display = "block";

  if (!silencioso) {
    mostrarEstado(`${etiqueta} (${pct == null ? "…" : pct.toFixed(0) + "%"})`, "info");
  }
}

/** Oculta la barra de progreso de modelos. */
function ocultarProgresoCarga() {
  const cont = document.getElementById("progreso-modelos");
  if (cont) {
    cont.style.display = "none";
    cont.innerHTML = "";
  }
}

/** Badge discreto si ELECTRA truncó el texto (artículos largos vía URL). */
function mostrarHintTruncado() {
  const panel = document.querySelector(".tab-panel.active") || document;
  let hint = panel.querySelector(".hint-truncado-electra");
  if (!hint) {
    const vivo =
      panel.querySelector("#analisis-vivo") ||
      panel.querySelector("#url-analisis-vivo") ||
      panel.querySelector(".analisis-vivo");
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
      const area = document.getElementById("input-texto");
      area.value = n.texto;
      area.focus();
      mostrarEstado(`Ejemplo de ${n.categoria} cargado. Pulsa «Analizar noticia».`, "info");
    };
    cont.appendChild(btn);
  });
}

/**
 * Botones de URL de ejemplo del panel "Por URL". Rellenan el input sin lanzar
 * el análisis, para que el usuario vea el flujo completo.
 */
function mostrarEjemplosURL() {
  const cont = document.getElementById("ejemplos-url");
  if (!cont) return;
  // Portadas de medios en español: existen de forma estable y permiten ver el
  // aviso del extractor cuando la página no es un artículo (botón "forzar").
  const urls = [
    ["BBC Mundo (portada)", "https://www.bbc.com/mundo"],
    ["El País (portada)", "https://elpais.com/"],
  ];
  cont.innerHTML = '<span class="ejemplos-label">Ejemplos</span>';
  urls.forEach(([nombre, url]) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-ejemplo";
    btn.textContent = nombre;
    btn.title = url;
    btn.onclick = () => {
      const input = document.getElementById("input-url");
      input.value = url;
      input.focus();
    };
    cont.appendChild(btn);
  });
}

/**
 * Activa una pestaña y su panel (patrón ARIA tabs).
 * Gestiona .active, aria-selected y el roving tabindex.
 */
function activarTab(tab, tabs) {
  tabs.forEach((t) => {
    const activo = t === tab;
    t.classList.toggle("active", activo);
    t.setAttribute("aria-selected", activo ? "true" : "false");
    t.tabIndex = activo ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  const panel = document.getElementById(tab.dataset.target);
  if (panel) panel.classList.add("active");
  // El skip-link debe llevar siempre al panel visible (los ocultos no son enfocables).
  const skip = document.querySelector(".skip-link");
  if (skip) skip.setAttribute("href", "#" + tab.dataset.target);
  if (tab.dataset.target === "panel-notebook") {
    ensureNotebookLoaded();
  }
}

/**
 * Navegación por teclado del tablist: ←/→ (y ↑/↓ en vertical), Home, End.
 * La tecla no se propaga para no hacer scroll de la página.
 */
function navegarTabs(e, tab, tabs) {
  const teclas = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"];
  if (!teclas.includes(e.key)) return;
  e.preventDefault();

  const idx = tabs.indexOf(tab);
  let destino = idx;
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") destino = (idx - 1 + tabs.length) % tabs.length;
  else if (e.key === "ArrowRight" || e.key === "ArrowDown") destino = (idx + 1) % tabs.length;
  else if (e.key === "Home") destino = 0;
  else if (e.key === "End") destino = tabs.length - 1;

  const siguiente = tabs[destino];
  if (!siguiente || siguiente === tab) return;
  activarTab(siguiente, tabs);
  siguiente.focus();
}

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("btn-analizar").addEventListener("click", analizarTexto);
  document.getElementById("input-texto").addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") analizarTexto();
  });

  const btnSent = document.getElementById("btn-precargar-sentimiento");
  if (btnSent) {
    btnSent.addEventListener("click", async () => {
      if (cargandoSentimiento) return;
      cargandoSentimiento = true;
      actualizarBotonPrecargaSentimiento();
      mostrarEstado("Precargando RoBERTuito (~25 MB)…", "info");
      const ok = await cargarModeloSentimiento();
      cargandoSentimiento = false;
      if (ok) {
        mostrarEstado("Modelo de sentimiento listo.", "ok");
        anunciar("Modelo de sentimiento listo.");
      } else {
        mostrarEstado("No se pudo cargar RoBERTuito; se usará el léxico.", "warn");
      }
    });
  }

  const btnURL = document.getElementById("btn-analizar-url");
  if (btnURL) btnURL.addEventListener("click", analizarURL);
  mostrarEjemplosURL();
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

  const tabs = [...document.querySelectorAll(".tab")];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activarTab(tab, tabs));
    tab.addEventListener("keydown", (e) => navegarTabs(e, tab, tabs));
  });

  await inicializar();
  await cargarMetricas();
});
