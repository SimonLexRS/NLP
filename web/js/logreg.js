// logreg.js — Espejo JS de backend/pipeline/models_classic.py (LogisticRegression)
// Inferencia de Logistic Regression desde los pesos exportados a JSON.
//
// LogisticRegression:
//   score(clase) = intercept[clase] + sum(tfidf_i * coef[clase, i])
//   softmax(score) -> probabilidad.

let LOGREG_WEIGHTS = null;

/**
 * Carga los pesos de Logistic Regression desde assets/logreg_weights.json.
 */
export async function cargarLogReg(url = "assets/logreg_weights.json") {
  const resp = await fetch(url);
  LOGREG_WEIGHTS = await resp.json();
  return LOGREG_WEIGHTS;
}

export function setWeights(weights) {
  LOGREG_WEIGHTS = weights;
}

/**
 * Predice la categoría de un vector TF-IDF disperso (Map<indice, tfidf>).
 * Devuelve { label, confidence, scores }.
 */
export function predecir(vectorTfidf) {
  if (!LOGREG_WEIGHTS) throw new Error("Pesos LogReg no cargados.");

  const { labels, coef, intercept } = LOGREG_WEIGHTS;
  const nClases = labels.length;

  const scores = new Array(nClases);
  for (let c = 0; c < nClases; c++) {
    let score = intercept[c];
    const coefs = coef[c];
    for (const [idx, tfidf] of vectorTfidf) {
      score += tfidf * coefs[idx];
    }
    scores[c] = score;
  }

  // Softmax.
  const maxScore = Math.max(...scores);
  const expScores = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map((e) => e / sumExp);

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

export function getVocabInfo() {
  if (!LOGREG_WEIGHTS) return null;
  return {
    vocabulary: LOGREG_WEIGHTS.vocabulary,
    idf: LOGREG_WEIGHTS.idf,
    ngram_range: LOGREG_WEIGHTS.ngram_range,
  };
}
