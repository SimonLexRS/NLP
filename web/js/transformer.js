// transformer.js — Carga los modelos ONNX con transformers.js (@huggingface/transformers)
// Dos modelos:
//   1. Clasificador de categoría (ELECTRA-small fine-tuneado, local en assets/model_onnx/)
//   2. Análisis de sentimiento (robertuito-sentiment-analysis-ONNX, desde HuggingFace Hub)

import { pipeline, env } from "@huggingface/transformers";

/** Cache API solo en contexto seguro (HTTPS o localhost). Evita el warning en file:// / http://IP. */
export const cacheDisponible =
  typeof caches !== "undefined" &&
  typeof window !== "undefined" &&
  !!window.isSecureContext;

env.allowLocalModels = true;
env.allowRemoteModels = true;
env.useBrowserCache = cacheDisponible;

const TIMEOUT_CATEGORIA_MS = 60000;
const TIMEOUT_SENTIMIENTO_MS = 90000;
/** ~512 tokens en español ≈ 1800 caracteres. */
const MAX_CHARS_ONNX = 1800;

let categoriaClassifier = null;
let sentimientoClassifier = null;
let loadingCategoria = null;
let loadingSentimiento = null;

/** Cola serializada: evita carrera allowLocalModels entre ELECTRA (local) y RoBERTuito (remoto). */
let onnxLoadLock = Promise.resolve();

function withOnnxLock(fn) {
  const run = onnxLoadLock.then(fn, fn);
  onnxLoadLock = run.then(
    () => undefined,
    () => undefined
  );
  return run;
}

function conTimeout(promesa, ms, etiqueta) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`Timeout cargando ${etiqueta} (${Math.round(ms / 1000)}s)`)),
      ms
    );
  });
  return Promise.race([promesa, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Trunca texto para inferencia ONNX (ELECTRA max 512 posiciones).
 * Devuelve { texto, truncado }.
 */
export function truncarParaModelo(texto, maxChars = MAX_CHARS_ONNX) {
  const t = (texto || "").trim();
  if (t.length <= maxChars) return { texto: t, truncado: false };
  // Preferir corte en límite de palabra.
  let corte = t.slice(0, maxChars);
  const ultimoEspacio = corte.lastIndexOf(" ");
  if (ultimoEspacio > maxChars * 0.7) corte = corte.slice(0, ultimoEspacio);
  return { texto: corte.trim(), truncado: true };
}

/** Base de assets relativa a la página (compatible con GitHub Pages /NLP/). */
function baseAssetsHref() {
  if (typeof window === "undefined" || !window.location?.href) {
    return "assets/";
  }
  return new URL("assets/", window.location.href).href;
}

/**
 * Carga el clasificador de CATEGORÍA (ELECTRA-small fine-tuneado, ONNX local).
 * onProgress: callback(progreso 0-100, file)
 */
export async function cargarClasificadorCategoria(onProgress = null) {
  if (categoriaClassifier) return categoriaClassifier;
  if (loadingCategoria) return loadingCategoria;

  loadingCategoria = withOnnxLock(async () => {
    if (categoriaClassifier) return categoriaClassifier;
    const prevLocal = env.allowLocalModels;
    const prevLocalPath = env.localModelPath;
    try {
      env.localModelPath = baseAssetsHref();
      env.allowLocalModels = true;
      categoriaClassifier = await conTimeout(
        pipeline("text-classification", "model_onnx", {
          dtype: "q8",
          progress_callback: (data) => {
            if (onProgress && data.status === "progress") {
              onProgress(data.progress, data.file);
            }
          },
        }),
        TIMEOUT_CATEGORIA_MS,
        "Transformer ELECTRA"
      );
      return categoriaClassifier;
    } catch (e) {
      loadingCategoria = null;
      categoriaClassifier = null;
      throw e;
    } finally {
      env.allowLocalModels = prevLocal;
      env.localModelPath = prevLocalPath;
    }
  });

  return loadingCategoria;
}

/**
 * Carga el clasificador de SENTIMIENTO (robertuito desde HuggingFace Hub).
 * Desactiva probes locales temporales para evitar 404 en /models/...
 */
export async function cargarClasificadorSentimiento(onProgress = null) {
  if (sentimientoClassifier) return sentimientoClassifier;
  if (loadingSentimiento) return loadingSentimiento;

  loadingSentimiento = withOnnxLock(async () => {
    if (sentimientoClassifier) return sentimientoClassifier;
    const prevLocal = env.allowLocalModels;
    try {
      env.allowLocalModels = false;
      sentimientoClassifier = await conTimeout(
        pipeline(
          "text-classification",
          "onnx-community/robertuito-sentiment-analysis-ONNX",
          {
            dtype: "q8",
            progress_callback: (data) => {
              if (onProgress && data.status === "progress") {
                onProgress(data.progress, data.file);
              }
            },
          }
        ),
        TIMEOUT_SENTIMIENTO_MS,
        "RoBERTuito"
      );
      return sentimientoClassifier;
    } catch (e) {
      loadingSentimiento = null;
      sentimientoClassifier = null;
      throw e;
    } finally {
      env.allowLocalModels = prevLocal;
    }
  });

  return loadingSentimiento;
}

/**
 * Predice la categoría de un texto usando el transformer ONNX.
 * Devuelve { label, confidence, scores, truncado }.
 */
export async function predecirCategoria(texto) {
  if (!categoriaClassifier) {
    await cargarClasificadorCategoria();
  }
  const { texto: input, truncado } = truncarParaModelo(texto);
  const output = await categoriaClassifier(input, { top_k: 7 });

  let resultados;
  if (Array.isArray(output)) {
    resultados = output;
  } else {
    resultados = [output];
  }

  const id2label = {
    LABEL_0: "politica",
    politica: "politica",
    LABEL_1: "economia",
    economia: "economia",
    LABEL_2: "deportes",
    deportes: "deportes",
    LABEL_3: "tecnologia",
    tecnologia: "tecnologia",
    LABEL_4: "salud",
    salud: "salud",
    LABEL_5: "internacional",
    internacional: "internacional",
    LABEL_6: "cultura",
    cultura: "cultura",
  };

  const scores = {};
  let best = { label: "politica", score: 0 };
  for (const r of resultados) {
    const label = id2label[r.label] || r.label;
    scores[label] = r.score;
    if (r.score > best.score) {
      best = { label, score: r.score };
    }
  }

  return {
    label: best.label,
    confidence: best.score,
    scores,
    truncado,
  };
}

/**
 * Predice el sentimiento de un texto usando robertuito ONNX.
 * Devuelve { label, confidence, scores, truncado }.
 */
export async function predecirSentimiento(texto) {
  if (!sentimientoClassifier) {
    await cargarClasificadorSentimiento();
  }
  const { texto: input, truncado } = truncarParaModelo(texto);
  const output = await sentimientoClassifier(input, { top_k: 3 });

  let resultados;
  if (Array.isArray(output)) {
    resultados = output;
  } else {
    resultados = [output];
  }

  const labelMap = {
    POS: "positivo",
    positivo: "positivo",
    POSITIVE: "positivo",
    NEG: "negativo",
    negativo: "negativo",
    NEGATIVE: "negativo",
    NEU: "neutro",
    neutro: "neutro",
    NEUTRAL: "neutro",
  };

  const scores = {};
  let best = { label: "neutro", score: 0 };
  for (const r of resultados) {
    const label = labelMap[r.label] || r.label.toLowerCase();
    scores[label] = r.score;
    if (r.score > best.score) {
      best = { label, score: r.score };
    }
  }

  return {
    label: best.label,
    confidence: best.score,
    scores,
    truncado,
  };
}

export function estadoCarga() {
  return {
    categoria: !!categoriaClassifier,
    sentimiento: !!sentimientoClassifier,
  };
}

/** Resume un error de carga/inferencia para la UI (corto). */
export function mensajeErrorCorto(err) {
  const msg = (err && err.message) || String(err || "Error");
  if (/timeout/i.test(msg)) return "Timeout";
  if (/local file missing/i.test(msg)) return "No se encontró el modelo ONNX local";
  if (/network|fetch|failed to fetch/i.test(msg)) return "Red";
  if (/memory|oom|out of memory/i.test(msg)) return "Memoria";
  if (/404|not found/i.test(msg)) return "Modelo no encontrado";
  if (msg.length > 40) return msg.slice(0, 37) + "…";
  return msg;
}
