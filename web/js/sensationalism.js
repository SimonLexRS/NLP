// sensationalism.js — Espejo JS de backend/pipeline/sensationalism.py
// Detecta tono sensacionalista con reglas léxicas (clickbait, emocional, exclamaciones, mayúsculas).

let REGLAS = null;

/**
 * Carga las reglas desde assets/sensationalism_rules.json.
 */
export async function cargarReglas(url = "assets/sensationalism_rules.json") {
  const resp = await fetch(url);
  REGLAS = await resp.json();
  return REGLAS;
}

function normalizarParaMatch(texto) {
  return texto
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // quitar tildes
    .replace(/ñ/g, "n");
}

/**
 * Analiza el tono sensacionalista de un texto.
 * Espejo de sensationalism.analizar_sensacionalismo().
 *
 * Devuelve: { score: 0-1, label: "sensacionalista"|"informativo", signals: {...} }
 */
export function analizar(texto) {
  if (!REGLAS) throw new Error("Reglas no cargadas. Llama a cargarReglas() primero.");
  if (typeof texto !== "string" || !texto.trim()) {
    return { score: 0, label: "informativo", signals: {} };
  }

  const { clickbait_patterns, palabras_emocionales, umbral, pesos, caps } = REGLAS;
  const textoNorm = normalizarParaMatch(texto);
  const palabras = textoNorm.split(/\s+/);
  const nPalabras = Math.max(palabras.length, 1);

  // Señal 1: patrón clickbait al inicio.
  const inicioNorm = textoNorm.slice(0, 80);
  const clickbaitHits = clickbait_patterns.filter((p) => {
    try {
      return new RegExp(p).test(inicioNorm);
    } catch {
      return inicioNorm.includes(p);
    }
  });
  const clickbaitScore = Math.min(clickbaitHits.length * pesos.clickbait, caps.clickbait);

  // Señal 2: palabras emocionales.
  const emocionalHits = palabras_emocionales.filter((p) => textoNorm.includes(p));
  const emocionalScore = Math.min(emocionalHits.length * pesos.emocional, caps.emocional);

  // Señal 3: exclamaciones.
  const nExclam = (texto.match(/[!¡]+/g) || []).length;
  const exclamScore = Math.min(nExclam * pesos.exclamacion, caps.exclamacion);

  // Señal 4: proporción de mayúsculas.
  const mayusPalabras = (texto.match(/\b[A-ZÁÉÍÓÚÑ]{2,}\b/g) || []).length;
  const propMayus = mayusPalabras / nPalabras;
  const mayusScore = Math.min(propMayus * pesos.mayusculas, caps.mayusculas);

  // Señal 5: interrogaciones.
  const nInterrog = (texto.match(/[?¿]+/g) || []).length;
  const interrogScore = Math.min(nInterrog * pesos.interrogacion, caps.interrogacion);

  const score = Math.min(
    clickbaitScore + emocionalScore + exclamScore + mayusScore + interrogScore,
    1
  );
  const label = score >= umbral ? "sensacionalista" : "informativo";

  return {
    score: Math.round(score * 10000) / 10000,
    label,
    signals: {
      clickbait_hits: clickbaitHits,
      emocional_hits: emocionalHits,
      n_exclamaciones: nExclam,
      prop_mayusculas: Math.round(propMayus * 10000) / 10000,
      n_interrogaciones: nInterrog,
    },
  };
}
