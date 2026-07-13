// vectorize.js — Espejo JS de backend/pipeline/vectorize.py (Semana 2)
// Construye el vector TF-IDF de un texto usando el vocabulario e IDF exportados.

let VOCAB = null; // { termino: indice }
let IDF = null; // [idf_0, idf_1, ...]
let NGRAM_RANGE = [1, 1];

/**
 * Carga el vocabulario TF-IDF desde assets/tfidf_vocab.json.
 * (Generado a partir del vectorizador del modelo clásico.)
 */
export async function cargarVocab(url = "assets/nb_weights.json") {
  // El vocabulario está embebido en nb_weights.json y logreg_weights.json.
  // Lo cargamos desde nb_weights.json (son el mismo vocabulario por construcción).
  const resp = await fetch(url);
  const data = await resp.json();
  VOCAB = data.vocabulary;
  IDF = data.idf;
  NGRAM_RANGE = data.ngram_range || [1, 1];
  return VOCAB;
}

/**
 * Permite pasar vocab/idf directamente (cuando ya se cargaron para otro modelo).
 */
export function setVocab(vocab, idf, ngramRange = [1, 1]) {
  VOCAB = vocab;
  IDF = idf;
  NGRAM_RANGE = ngramRange;
}

/**
 * Genera los n-gramas de un texto tokenizado.
 * ngramRange = [minN, maxN], ej [1, 2] -> unigramas + bigramas.
 */
function generarNgramas(tokens, minN, maxN) {
  const ngramas = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i <= tokens.length - n; i++) {
      ngramas.push(tokens.slice(i, i + n).join(" "));
    }
  }
  return ngramas;
}

/**
 * Construye el vector TF-IDF disperso de un texto.
 * Devuelve un Map<indice, tfidf> con solo los términos presentes.
 *
 * Fórmula (consistente con sklearn TfidfVectorizer sublinear_tf=True):
 *   tf = 1 + log(count)   (sublinear)
 *   tfidf = tf * idf[term]
 *   luego se normaliza L2.
 */
export function vectorizar(texto) {
  if (!VOCAB || !IDF) {
    throw new Error("Vocabulario no cargado. Llama a cargarVocab() primero.");
  }
  // Re-tokenizar el texto (asumiendo que ya viene preprocesado como string de tokens).
  // Si viene texto crudo, lo tokenizamos por espacios.
  const tokens = typeof texto === "string" ? texto.split(" ").filter(Boolean) : texto;
  const [minN, maxN] = NGRAM_RANGE;
  const ngramas = generarNgramas(tokens, minN, maxN);

  // Contar ocurrencias por término del vocabulario.
  const counts = new Map();
  for (const ng of ngramas) {
    if (ng in VOCAB) {
      const idx = VOCAB[ng];
      counts.set(idx, (counts.get(idx) || 0) + 1);
    }
  }

  // Calcular TF-IDF (sublinear).
  const tfidf = new Map();
  for (const [idx, count] of counts) {
    const tf = 1 + Math.log(count);
    const weight = tf * (IDF[idx] || 1.0);
    tfidf.set(idx, weight);
  }

  // Normalización L2.
  let norm = 0;
  for (const v of tfidf.values()) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (const [idx, v] of tfidf) tfidf.set(idx, v / norm);
  }

  return tfidf; // Map<indice, tfidf>
}
