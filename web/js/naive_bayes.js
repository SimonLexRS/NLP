// naive_bayes.js — Espejo JS de backend/pipeline/models_classic.py (MultinomialNB)
// Inferencia de Naive Bayes Multinomial desde los pesos exportados a JSON.
//
// MultinomialNB:
//   score(clase) = class_log_prior[clase] + sum(tfidf_i * feature_log_prob[clase, i])
//   para las features presentes en el vector. argmax -> clase.
//   probabilidad aprox = softmax(score).

let NB_WEIGHTS = null;

/**
 * Carga los pesos de Naive Bayes desde assets/nb_weights.json.
 */
export async function cargarNB(url = "assets/nb_weights.json") {
  const resp = await fetch(url);
  NB_WEIGHTS = await resp.json();
  return NB_WEIGHTS;
}

/**
 * Permite pasar los pesos directamente (cuando ya están cargados).
 */
export function setWeights(weights) {
  NB_WEIGHTS = weights;
}

/**
 * Predice la categoría de un vector TF-IDF disperso (Map<indice, tfidf>).
 *
 * Devuelve:
 *   {
 *     label: string,            // categoría predicha
 *     confidence: number,       // probabilidad de la clase predicha
 *     scores: { cat: prob },    // distribución de probabilidades
 *   }
 */
export function predecir(vectorTfidf) {
  if (!NB_WEIGHTS) throw new Error("Pesos NB no cargados. Llama a cargarNB() primero.");

  const { labels, feature_log_prob, class_log_prior } = NB_WEIGHTS;
  const nClases = labels.length;

  // Calcular score por clase.
  const scores = new Array(nClases);
  for (let c = 0; c < nClases; c++) {
    let score = class_log_prior[c];
    const logProbs = feature_log_prob[c];
    for (const [idx, tfidf] of vectorTfidf) {
      score += tfidf * logProbs[idx];
    }
    scores[c] = score;
  }

  // Softmax para obtener probabilidades.
  const maxScore = Math.max(...scores);
  const expScores = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map((e) => e / sumExp);

  // argmax.
  let bestIdx = 0;
  let bestProb = probs[0];
  for (let c = 1; c < nClases; c++) {
    if (probs[c] > bestProb) {
      bestProb = probs[c];
      bestIdx = c;
    }
  }

  const scoresObj = {};
  for (let c = 0; c < nClases; c++) {
    scoresObj[labels[c]] = probs[c];
  }

  return {
    label: labels[bestIdx],
    confidence: bestProb,
    scores: scoresObj,
  };
}

/**
 * Devuelve el vocabulario e IDF (para que vectorize.js los use).
 */
export function getVocabInfo() {
  if (!NB_WEIGHTS) return null;
  return {
    vocabulary: NB_WEIGHTS.vocabulary,
    idf: NB_WEIGHTS.idf,
    ngram_range: NB_WEIGHTS.ngram_range,
  };
}
