// lda.js — Espejo JS de backend/pipeline/topics.py (Semana 3)
// Asigna el tema LDA a un texto por similitud coseno contra los temas precomputados.
//
// No corre Gibbs sampling en el navegador (sería lento). En su lugar:
//   1. Vectoriza el texto sobre el vocabulario LDA (conteos).
//   2. Calcula similitud coseno entre el vector del texto y cada tema (fila de components).
//   3. Devuelve el tema más similar + su distribución.

let LDA_DATA = null;

/**
 * Carga los temas LDA desde assets/lda_topics.json.
 */
export async function cargarLDA(url = "assets/lda_topics.json") {
  const resp = await fetch(url);
  LDA_DATA = await resp.json();
  return LDA_DATA;
}

/**
 * Asigna el tema a un texto preprocesado (string de tokens separados por espacio).
 *
 * Devuelve:
 *   {
 *     topicId: number,
 *     topWords: [[palabra, peso], ...],  // top palabras del tema
 *     distribution: [sim_0, sim_1, ...], // similitudes coseno normalizadas
 *   }
 */
export function predecirTema(textoPreprocesado) {
  if (!LDA_DATA) throw new Error("LDA no cargado. Llama a cargarLDA() primero.");

  const { vocabulary, components, topics, n_topics } = LDA_DATA;
  const tokens = typeof textoPreprocesado === "string"
    ? textoPreprocesado.split(" ").filter(Boolean)
    : textoPreprocesado;

  // Vector de conteos del texto (solo términos en el vocabulario LDA).
  const textVec = new Array(Object.keys(vocabulary).length).fill(0);
  for (const t of tokens) {
    if (t in vocabulary) {
      textVec[vocabulary[t]] += 1;
    }
  }

  // Similitud coseno entre textVec y cada tema.
  const textNorm = Math.sqrt(textVec.reduce((s, v) => s + v * v, 0));
  const sims = [];
  for (let k = 0; k < n_topics; k++) {
    const topic = components[k];
    let dot = 0;
    let topicNorm = 0;
    for (let i = 0; i < textVec.length; i++) {
      dot += textVec[i] * topic[i];
      topicNorm += topic[i] * topic[i];
    }
    topicNorm = Math.sqrt(topicNorm);
    const sim = textNorm > 0 && topicNorm > 0 ? dot / (textNorm * topicNorm) : 0;
    sims.push(sim);
  }

  // Normalizar similitudes a distribución (softmax-like).
  const maxSim = Math.max(...sims, 0);
  const expSims = sims.map((s) => Math.exp((s - maxSim) * 4)); // factor 4 para afilar
  const sumExp = expSims.reduce((a, b) => a + b, 0);
  const dist = expSims.map((e) => e / sumExp);

  // Tema más similar.
  let bestIdx = 0;
  let bestSim = sims[0];
  for (let k = 1; k < n_topics; k++) {
    if (sims[k] > bestSim) {
      bestSim = sims[k];
      bestIdx = k;
    }
  }

  return {
    topicId: bestIdx,
    similarity: bestSim,
    distribution: dist,
    topWords: topics[bestIdx].top_words.slice(0, 8),
  };
}
