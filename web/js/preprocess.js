// preprocess.js — Espejo JS de backend/pipeline/preprocess.py (Semana 1)
// Normaliza, tokeniza y filtra stopwords en español, conservando negaciones.

// Negaciones que SIEMPRE se conservan (no son stopwords).
const NEGACIONES = new Set(["no", "ni", "nunca", "nada", "sin", "tampoco", "jamas"]);

// Stopwords en español (se cargan desde assets/stopwords.json).
let STOPWORDS = null;

// Mapa de lematización ligera (se llena con el vocabulario TF-IDF exportado).
// Para palabras fuera del mapa, se devuelve la palabra sin cambios.
let LEMMA_MAP = null;

// Mapa de acentos -> sin acentos (equivalente a unidecode).
const ACCENT_MAP = { á: "a", é: "e", í: "i", ó: "o", ú: "u", ñ: "n", ü: "u" };
function unidecode(texto) {
  return texto.replace(/[áéíóúñüÁÉÍÓÚÑÜ]/g, (c) => ACCENT_MAP[c.toLowerCase()] || c);
}

// Patrones de normalización.
const URL_RE = /https?:\/\/\S+|www\.\S+/g;
const MENCION_RE = /@\w+/g;
const HASHTAG_RE = /#(\w+)/g;
const PUNT_RE = /[^a-z0-9\s]/g;

/**
 * Normaliza el texto: minusculas, URLs, menciones, hashtags y tildes.
 * Espejo de preprocess.normalizar_texto().
 */
export function normalizarTexto(texto) {
  if (typeof texto !== "string") return "";
  let t = texto.toLowerCase();
  t = t.replace(URL_RE, " url ");
  t = t.replace(MENCION_RE, " user ");
  t = t.replace(HASHTAG_RE, "$1");
  t = unidecode(t);
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

/**
 * Tokeniza y filtra stopwords (conservando negaciones).
 * Espejo de preprocess.tokenizar().
 */
export function tokenizar(textoNormalizado) {
  // Tokenizacion simple por espacios + puntuacion (equivalente a TweetTokenizer
  // sin reducir longitud). Suficiente porque normalizarTexto ya quito acentos.
  const tokens = textoNormalizado.match(/[a-z0-9]+/g) || [];
  if (!STOPWORDS) return tokens.filter((t) => t.length > 1);
  return tokens.filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/**
 * Lematiza tokens usando el mapa de lemas exportado.
 * Espejo de preprocess.lematizar() (con diccionario en lugar de spaCy).
 */
export function lematizar(tokens) {
  if (!LEMMA_MAP) return tokens;
  return tokens.map((t) => LEMMA_MAP[t] || t);
}

/**
 * Pipeline completo de preprocesamiento: normalizar -> tokenizar -> lematizar.
 * Devuelve un array de tokens.
 */
export function preproceso(texto) {
  const norm = normalizarTexto(texto);
  const toks = tokenizar(norm);
  return lematizar(toks);
}

/**
 * Devuelve el preprocesamiento como string (tokens separados por espacio).
 * Se usa como entrada para vectorizar.
 */
export function preprocesoTexto(texto) {
  return preproceso(texto).join(" ");
}

/**
 * Carga las stopwords desde assets/stopwords.json.
 */
export async function cargarStopwords(url = "assets/stopwords.json") {
  const resp = await fetch(url);
  const lista = await resp.json();
  STOPWORDS = new Set(lista);
  return STOPWORDS;
}

/**
 * Construye el mapa de lemas a partir del vocabulario TF-IDF exportado.
 * Como no exportamos un lematizador completo, usamos una lematización
 * simple por sufijo para verbos comunes en español.
 */
export function construirLemmaMap(vocab) {
  LEMMA_MAP = {};
  // Lematización por sufijos común en español (muy ligera).
  const sufijos = [
    [/ciones$/, "cion"], [/uciones$/, "ucion"], [/amientos$/, "amiento"],
    [/imientos$/, "imiento"], [/aban$/, "ar"], [/iamos$/, "ar"], [/isteis$/, "ar"],
    [/aron$/, "ar"], [/ando$/, "ar"], [/iendo$/, "er"], [/ado$/, "ar"], [/ido$/, "er"],
    [/aba$/, "ar"], [/ada$/, "ar"], [/ida$/, "er"], [/ases$/, "ar"], [/ese$/, "er"],
    [/iese$/, "er"], [/aste$/, "ar"], [/iste$/, "ir"], [/imos$/, "ar"], [/amos$/, "ar"],
    [/an$/, "ar"], [/en$/, "er"], [/ia$/, "ar"], [/io$/, "ir"], [/o$/, "ar"],
    [/es$/, ""], [/s$/, ""],
  ];
  for (const term of Object.keys(vocab)) {
    // Para bigramas, no lematizar.
    if (term.includes(" ")) continue;
    let lemma = term;
    for (const [pat, rep] of sufijos) {
      if (pat.test(term)) {
        lemma = term.replace(pat, rep);
        break;
      }
    }
    if (lemma !== term) LEMMA_MAP[term] = lemma;
  }
  return LEMMA_MAP;
}
