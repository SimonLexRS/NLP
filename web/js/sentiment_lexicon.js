// sentiment_lexicon.js — Espejo JS de backend/pipeline/sentiment.py (enfoque léxico)
// Análisis de sentimiento con diccionario + ventana de negación + intensificadores.
// Es un fallback rápido; el modelo principal es robertuito-sentiment-analysis-ONNX.

let LEXICON = null;
let NEGACIONES = null;
let INTENSIFICADORES = null;
let UMBRAL_POS = 0.15;
let UMBRAL_NEG = -0.15;

/**
 * Carga el léxico desde assets/lexicon.json.
 */
export async function cargarLexico(url = "assets/lexicon.json") {
  const resp = await fetch(url);
  const data = await resp.json();
  LEXICON = data.lexicon;
  NEGACIONES = new Set(data.negaciones);
  INTENSIFICADORES = data.intensificadores;
  UMBRAL_POS = data.umbral_positivo;
  UMBRAL_NEG = data.umbral_negativo;
  return data;
}

function normalizarPalabra(p) {
  return p
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/ñ/g, "n");
}

/**
 * Analiza el sentimiento de un texto con el enfoque léxico.
 * Espejo de sentiment.sentimiento_lexico().
 *
 * Devuelve: { label, score, polaridad }
 */
export function analizar(texto) {
  if (!LEXICON) throw new Error("Léxico no cargado.");
  if (typeof texto !== "string" || !texto.trim()) {
    return { label: "neutro", score: 0, polaridad: 0 };
  }

  const palabras = texto.split(/\s+/).map(normalizarPalabra);
  let polaridad = 0;
  let nLex = 0;

  for (let i = 0; i < palabras.length; i++) {
    const p = palabras[i];
    if (p in LEXICON) {
      let pol = LEXICON[p];
      // Ventana de negación (3 palabras hacia atrás).
      for (let j = Math.max(0, i - 3); j < i; j++) {
        if (NEGACIONES.has(palabras[j])) {
          pol = -pol;
          break;
        }
      }
      // Intensificador en la palabra inmediatamente anterior.
      if (i > 0 && palabras[i - 1] in INTENSIFICADORES) {
        pol *= INTENSIFICADORES[palabras[i - 1]];
      }
      polaridad += pol;
      nLex++;
    }
  }

  const polaridadNorm = nLex > 0 ? polaridad / nLex : 0;
  let label;
  if (polaridadNorm > UMBRAL_POS) label = "positivo";
  else if (polaridadNorm < UMBRAL_NEG) label = "negativo";
  else label = "neutro";

  const score = Math.min(Math.abs(polaridadNorm), 1);
  return {
    label,
    score: Math.round(score * 10000) / 10000,
    polaridad: Math.round(polaridadNorm * 10000) / 10000,
  };
}
