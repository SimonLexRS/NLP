// pipeline_edu.js — Explainers educativos por etapa (ES por defecto, EN opcional).
// Inspired by Transformer Explainer (poloclub): diagrama + qué/por qué/cómo + snapshot vivo.

const LANG_KEY = "pipeline-edu-lang";

let lang = "es";
try {
  const saved = localStorage.getItem(LANG_KEY);
  if (saved === "es" || saved === "en") lang = saved;
} catch (_) {
  /* ignore */
}

const UI = {
  es: {
    banner: "Explicación educativa · Español",
    what: "Qué hace",
    why: "Por qué importa",
    how: "Cómo funciona",
    live: "Snapshot en vivo",
    emptyRun: "Ejecuta el análisis para ver valores vivos de esta etapa.",
    emptySnap: "Sin snapshot aún.",
    learnOn: "Modo aprender ON",
    learnOff: "Modo aprender OFF",
  },
  en: {
    banner: "Educational explainer · English",
    what: "What it does",
    why: "Why it matters",
    how: "How it works",
    live: "Live snapshot",
    emptyRun: "Run analysis to see live values for this stage.",
    emptySnap: "No snapshot yet.",
    learnOn: "Learn mode ON",
    learnOff: "Learn mode OFF",
  },
};

/** Stage copy: es + en. SVG built per language. */
const EDU = {
  input: {
    es: {
      title: "Texto de entrada",
      what: "Recibe la noticia en bruto (titular + cuerpo) que recorrerá el pipeline NLP.",
      why: "Todas las etapas posteriores dependen de este texto: tokenización, vectores y modelos neurales parten de aquí.",
      how: [
        "El usuario pega una noticia en español o la extrae de una URL.",
        "Contamos palabras y caracteres como chequeo rápido de tamaño.",
        "Aún no corre ningún modelo: es solo el documento fuente.",
      ],
    },
    en: {
      title: "Input text",
      what: "Receives the raw news article (headline + body) that will flow through the NLP pipeline.",
      why: "Every later stage depends on this string: tokenization, vectors, and neural models all start here.",
      how: [
        "User pastes Spanish news text or extracts it from a URL.",
        "We count words/characters for a quick size check.",
        "No model runs yet — this is just the source document.",
      ],
    },
    svg: (L) =>
      L === "es"
        ? svgFlow([
            ["Documento", "noticia"],
            ["Pipeline", "inicia"],
          ])
        : svgFlow([
            ["Document", "raw news"],
            ["Pipeline", "starts"],
          ]),
  },
  preprocess: {
    es: {
      title: "Preprocesamiento",
      what: "Limpia y normaliza el español en tokens que los modelos pueden usar.",
      why: "El ruido (URLs, puntuación, stopwords) daña las features clásicas; tokens consistentes mejoran TF-IDF.",
      how: [
        "Normalizar: minúsculas, quitar acentos, reemplazar URLs/menciones.",
        "Tokenizar y quitar stopwords en español (conservando negaciones como «no»).",
        "Lematización ligera para que formas relacionadas compartan features.",
      ],
    },
    en: {
      title: "Preprocessing",
      what: "Cleans and normalizes Spanish text into tokens models can use.",
      why: "Noise (URLs, punctuation, stopwords) hurts classical features; consistent tokens improve TF-IDF.",
      how: [
        "Normalize: lowercase, strip accents, replace URLs/mentions.",
        "Tokenize and drop Spanish stopwords (keeping negations like “no”).",
        "Light lemmatization so related word forms share features.",
      ],
    },
    svg: (L) =>
      L === "es"
        ? svgFlow([
            ["Normalizar", "limpiar"],
            ["Tokenizar", "filtrar"],
            ["Lematizar", "formas base"],
          ])
        : svgFlow([
            ["Normalize", "clean"],
            ["Tokenize", "filter"],
            ["Lemmatize", "base forms"],
          ]),
  },
  vectorize: {
    es: {
      title: "Vectorización TF-IDF",
      what: "Convierte tokens en un vector numérico disperso que pondera términos importantes.",
      why: "Los clasificadores clásicos necesitan números. TF-IDF potencia palabras raras e informativas y resta peso a las comunes.",
      how: [
        "Construir n-gramas (uni/bi) a partir de los tokens preprocesados.",
        "Calcular TF sublineal × IDF para términos del vocabulario presentes en el texto.",
        "Normalizar L2 el vector disperso para comparar magnitudes entre documentos.",
      ],
    },
    en: {
      title: "TF-IDF vectorization",
      what: "Turns tokens into a sparse numeric vector that weights important terms.",
      why: "Classical classifiers need numbers. TF-IDF boosts rare, informative words and downweights common ones.",
      how: [
        "Build n-grams (uni/bi) from the preprocessed tokens.",
        "Compute sublinear TF × IDF for vocabulary terms present in the text.",
        "L2-normalize the sparse vector so magnitude is comparable across docs.",
      ],
    },
    svg: (L) =>
      L === "es"
        ? svgFlow([
            ["Tokens", "palabras"],
            ["Conteos", "TF"],
            ["× IDF", "pesos"],
            ["Vector", "disperso"],
          ])
        : svgFlow([
            ["Tokens", "words"],
            ["Counts", "TF"],
            ["× IDF", "weights"],
            ["Vector", "sparse"],
          ]),
  },
  temas: {
    es: {
      title: "Modelado de temas (LDA)",
      what: "Descubre de qué temas habla el corpus y sitúa cada noticia entre ellos, sin usar etiquetas.",
      why: "Complementa la categoría supervisada: describe el contenido con vocabulario propio del tema y permite explorar agrupaciones no anotadas.",
      how: [
        "Los temas ya fueron entrenados con LDA y exportados como matriz tema×término.",
        "El texto preprocesado se convierte en conteos sobre el vocabulario LDA.",
        "Similitud coseno contra cada tema → distribución afilada y tema ganador.",
      ],
    },
    en: {
      title: "Topic modeling (LDA)",
      what: "Discovers what the corpus talks about and places each article across those topics, with no labels.",
      why: "Complements supervised categories: it describes the article with the topic's own vocabulary and reveals unlabeled groupings.",
      how: [
        "Topics were pre-trained with LDA and exported as a topic×term matrix.",
        "The preprocessed text becomes counts over the LDA vocabulary.",
        "Cosine similarity against each topic → sharpened distribution and winning topic.",
      ],
    },
    svg: (L) =>
      L === "es"
        ? svgFlow([
            ["Conteos", "vocab LDA"],
            ["Coseno", "× temas"],
            ["Tema", "top palabras"],
          ])
        : svgFlow([
            ["Counts", "LDA vocab"],
            ["Cosine", "× topics"],
            ["Topic", "top words"],
          ]),
  },
  nb: {
    es: {
      title: "Naive Bayes",
      what: "Clasificador probabilístico clásico: predice la categoría combinando verosimilitudes de features.",
      why: "Baseline rápido para texto. Fuerte cuando las features (términos TF-IDF) son algo independientes dada la clase.",
      how: [
        "Por cada feature TF-IDF activa, consultar pesos condicionales por clase.",
        "Combinar con priors de clase para puntuar cada categoría.",
        "Elegir la clase con mayor probabilidad a posteriori.",
      ],
    },
    en: {
      title: "Naive Bayes",
      what: "A classical probabilistic classifier: predicts category by combining feature likelihoods.",
      why: "Fast baseline for text. Strong when features (TF-IDF terms) are somewhat independent given the class.",
      how: [
        "For each active TF-IDF feature, look up class-conditional weights.",
        "Combine with class priors to score every category.",
        "Pick the class with the highest posterior probability.",
      ],
    },
    svg: (L) =>
      svgBarsSketch(
        "P(class | text)",
        L === "es" ? ["clase top", "segunda", "otras"] : ["top class", "runner-up", "others"]
      ),
  },
  logreg: {
    es: {
      title: "Regresión logística",
      what: "Aprende una frontera de decisión lineal en el espacio TF-IDF para categorías de noticias.",
      why: "Suele estar más calibrada que Naive Bayes y compite bien con bolsa de palabras.",
      how: [
        "Producto punto del vector TF-IDF con los vectores de pesos por clase.",
        "Aplicar softmax (o scores one-vs-rest) entre categorías.",
        "Seleccionar la etiqueta con mejor puntuación.",
      ],
    },
    en: {
      title: "Logistic Regression",
      what: "Learns a linear decision boundary in TF-IDF space for multi-class news categories.",
      why: "Often more calibrated than Naive Bayes and competitive on bag-of-words features.",
      how: [
        "Dot product of the TF-IDF vector with class weight vectors.",
        "Apply softmax (or one-vs-rest scores) across categories.",
        "Select the best-scoring label as the prediction.",
      ],
    },
    svg: (L) => svgBoundarySketch(L),
  },
  transformer: {
    es: {
      title: "Transformer (ELECTRA)",
      what: "Un encoder neuronal clasifica la secuencia completa con self-attention — la misma familia de ideas que GPT, pero para clasificación.",
      why: "Captura contexto entre palabras mejor que la bolsa de palabras. Conceptos del Explainer: embeddings, atención y probabilidades de salida.",
      how: [
        "Tokenizar → vectores de embedding (significado + posición).",
        "Bloques Transformer: la self-attention comparte contexto; el MLP refina cada token.",
        "Capa lineal final + softmax → probabilidad sobre categorías (ONNX en el navegador).",
      ],
    },
    en: {
      title: "Transformer (ELECTRA)",
      what: "A neural encoder classifies the full text sequence with self-attention — the same family of ideas as GPT, but for classification.",
      why: "Captures context between words better than bag-of-words. Inspired by concepts in Transformer Explainer: embeddings, attention, and output probabilities.",
      how: [
        "Tokenize → embedding vectors (meaning + position).",
        "Transformer blocks: self-attention lets tokens share context; MLP refines each token.",
        "Final linear layer + softmax → probability over news categories (ONNX in the browser).",
      ],
    },
    svg: (L) => svgTransformerSketch(L),
  },
  sensationalism: {
    es: {
      title: "Tono / sensacionalismo",
      what: "Detector basado en reglas de tono clickbait o sensacional frente a estilo informativo.",
      why: "Alfabetización mediática: las etiquetas solas omiten el estilo. Señales: léxico emocional, MAYÚSCULAS y exclamaciones.",
      how: [
        "Buscar hits de clickbait / léxico emocional.",
        "Contar intensidad de puntuación y proporción de mayúsculas.",
        "Combinar señales en un score de sensacionalismo y etiqueta.",
      ],
    },
    en: {
      title: "Tone / sensationalism",
      what: "Rule-based detector for clickbait-like or sensational tone versus informative style.",
      why: "News literacy: labels alone miss style. Signals include emotional words, ALL CAPS, and exclamations.",
      how: [
        "Scan for clickbait / emotional lexicon hits.",
        "Count punctuation intensity and uppercase ratio.",
        "Combine signals into a sensationalism score and label.",
      ],
    },
    svg: (L) => svgToneSketch(L),
  },
  sentiment: {
    es: {
      title: "Análisis de sentimiento",
      what: "Estima la polaridad: positivo, negativo o neutro.",
      why: "Complementa la categoría: la misma noticia política puede alabar o atacar.",
      how: [
        "Ruta rápida: léxico en español + reglas de negación.",
        "Ruta fuerte: sentimiento RoBERTuito ONNX (Transformer).",
        "Preferir la salida neural si está cargada; si no, conservar el léxico.",
      ],
    },
    en: {
      title: "Sentiment analysis",
      what: "Estimates polarity: positive, negative, or neutral.",
      why: "Complements category: the same politics article can praise or attack.",
      how: [
        "Fast path: Spanish lexicon + negation rules.",
        "Stronger path: RoBERTuito sentiment ONNX (Transformer).",
        "Prefer neural output when loaded; otherwise keep the lexicon result.",
      ],
    },
    svg: (L) =>
      L === "es"
        ? svgFlow([
            ["Léxico", "rápido"],
            ["RoBERTuito", "ONNX"],
            ["Etiqueta", "POS/NEG/NEU"],
          ])
        : svgFlow([
            ["Lexicon", "fast"],
            ["RoBERTuito", "ONNX"],
            ["Label", "POS/NEG/NEU"],
          ]),
  },
  consenso: {
    es: {
      title: "Consenso final",
      what: "Fusiona las distribuciones de probabilidad de los modelos en un veredicto ponderado.",
      why: "No basta el voto a mano alzada: se mezclan scores, confianza, margen top-1/top-2 y acuerdo entre modelos.",
      how: [
        "Pesar cada modelo (ELECTRA > NB > LogReg) por confianza y claridad de la predicción.",
        "Sumar la masa de probabilidad de cada categoría (blend).",
        "Bonificar acuerdo NB↔LR y unanimidad; amplificar ELECTRA si confía ≥80%.",
        "Elegir la categoría con mayor score mezclado y explicar el criterio en la UI.",
      ],
    },
    en: {
      title: "Final consensus",
      what: "Fuses model probability distributions into a weighted verdict.",
      why: "Hard majority is weak; blend scores, confidence, top-1/top-2 margin, and inter-model agreement.",
      how: [
        "Weight each model (ELECTRA > NB > LogReg) by confidence and prediction clarity.",
        "Accumulate probability mass per category (blend).",
        "Boost NB↔LR agreement and unanimity; amplify ELECTRA when confidence ≥80%.",
        "Pick the top blended category and show the rationale in the UI.",
      ],
    },
    svg: (L) => svgVoteSketch(L),
  },
};

let liveStore = {};
let teachRoot = null;

export function getLang() {
  return lang;
}

export function setLang(next) {
  lang = next === "en" ? "en" : "es";
  try {
    localStorage.setItem(LANG_KEY, lang);
  } catch (_) {
    /* ignore */
  }
  if (teachRoot && !teachRoot.hidden) {
    const open = teachRoot.querySelector(".pipeline-teach");
    const stageId = open && open.getAttribute("data-teach-stage");
    if (stageId) renderTeachPanel(teachRoot, stageId);
  }
}

export function uiStrings() {
  return UI[lang] || UI.es;
}

export function clearLiveData() {
  liveStore = {};
}

export function setLiveData(stageId, data) {
  liveStore[stageId] = { ...(liveStore[stageId] || {}), ...data };
}

export function getLiveData(stageId) {
  return liveStore[stageId] || null;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function svgFlow(nodes) {
  const w = Math.max(320, nodes.length * 110);
  const h = 72;
  let parts = "";
  nodes.forEach((n, i) => {
    const x = 16 + i * 110;
    // Colores via var() CSS para seguir al tema activo sin redibujar.
    parts += `<rect x="${x}" y="16" width="88" height="40" rx="6" style="fill:var(--surface-raised);stroke:var(--accent)" stroke-width="1.5"/>`;
    parts += `<text x="${x + 44}" y="34" text-anchor="middle" font-size="10" font-family="IBM Plex Sans,sans-serif" style="fill:var(--ink)" font-weight="600">${escapeHtml(n[0])}</text>`;
    parts += `<text x="${x + 44}" y="48" text-anchor="middle" font-size="9" font-family="IBM Plex Mono,monospace" style="fill:var(--muted)">${escapeHtml(n[1])}</text>`;
    if (i < nodes.length - 1) {
      const ax = x + 92;
      parts += `<path d="M${ax} 36 L${ax + 14} 36" style="stroke:var(--signal)" stroke-width="2" marker-end="url(#eduArrow)"/>`;
    }
  });
  return `<svg viewBox="0 0 ${w} ${h}" class="edu-svg" role="img" aria-label="Stage flow diagram">${markerDefs()}${parts}</svg>`;
}

function markerDefs() {
  return `<defs><marker id="eduArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" style="fill:var(--signal)"/></marker></defs>`;
}

function svgBarsSketch(label, barLabels) {
  const [a, b, c] = barLabels;
  return `<svg viewBox="0 0 340 90" class="edu-svg" role="img" aria-label="Probability bars">
    ${markerDefs()}
    <text x="8" y="14" font-size="10" style="fill:var(--muted)" font-family="IBM Plex Mono,monospace">${escapeHtml(label)}</text>
    <rect x="8" y="28" width="140" height="12" rx="3" style="fill:var(--accent)" opacity="0.9"/>
    <rect x="8" y="48" width="95" height="12" rx="3" style="fill:var(--signal)" opacity="0.75"/>
    <rect x="8" y="68" width="55" height="12" rx="3" style="fill:var(--muted)" opacity="0.7"/>
    <text x="160" y="38" font-size="10" style="fill:var(--ink)" font-family="IBM Plex Sans,sans-serif">${escapeHtml(a)}</text>
    <text x="160" y="58" font-size="10" style="fill:var(--muted)" font-family="IBM Plex Sans,sans-serif">${escapeHtml(b)}</text>
    <text x="160" y="78" font-size="10" style="fill:var(--muted)" font-family="IBM Plex Sans,sans-serif">${escapeHtml(c)}</text>
  </svg>`;
}

function svgBoundarySketch(L) {
  const caption =
    L === "es" ? "separador lineal en el espacio de features" : "linear separator in feature space";
  return `<svg viewBox="0 0 340 90" class="edu-svg" role="img" aria-label="Linear decision boundary">
    <circle cx="70" cy="55" r="5" style="fill:var(--accent)"/><circle cx="95" cy="40" r="5" style="fill:var(--accent)"/>
    <circle cx="55" cy="35" r="5" style="fill:var(--accent)"/><circle cx="220" cy="55" r="5" style="fill:var(--danger)"/>
    <circle cx="250" cy="40" r="5" style="fill:var(--danger)"/><circle cx="235" cy="70" r="5" style="fill:var(--danger)"/>
    <line x1="130" y1="80" x2="200" y2="15" style="stroke:var(--signal)" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="8" y="14" font-size="10" style="fill:var(--muted)" font-family="IBM Plex Mono,monospace">${escapeHtml(caption)}</text>
  </svg>`;
}

function svgTransformerSketch(L) {
  const t =
    L === "es"
      ? {
          tokens: "Tokens",
          embed: "embed",
          block: "Transformer",
          attn: "self-attention",
          mlp: "+ MLP",
          logits: "Logits",
          soft: "softmax",
          labels: "Etiquetas",
        }
      : {
          tokens: "Tokens",
          embed: "embed",
          block: "Transformer",
          attn: "self-attention",
          mlp: "+ MLP",
          logits: "Logits",
          soft: "softmax",
          labels: "Labels",
        };
  return `<svg viewBox="0 0 420 110" class="edu-svg" role="img" aria-label="Simplified Transformer block">
    ${markerDefs()}
    <rect x="8" y="30" width="70" height="44" rx="6" style="fill:var(--surface-raised);stroke:var(--accent)"/>
    <text x="43" y="48" text-anchor="middle" font-size="9" style="fill:var(--ink)" font-weight="600">${t.tokens}</text>
    <text x="43" y="62" text-anchor="middle" font-size="8" style="fill:var(--muted)">${t.embed}</text>
    <path d="M82 52 L98 52" style="stroke:var(--signal)" stroke-width="2" marker-end="url(#eduArrow)"/>
    <rect x="102" y="18" width="120" height="68" rx="6" style="fill:var(--accent-subtle);stroke:var(--signal)"/>
    <text x="162" y="36" text-anchor="middle" font-size="9" style="fill:var(--ink)" font-weight="600">${t.block}</text>
    <text x="162" y="52" text-anchor="middle" font-size="8" style="fill:var(--signal)">${t.attn}</text>
    <text x="162" y="66" text-anchor="middle" font-size="8" style="fill:var(--signal)">${t.mlp}</text>
    <path d="M226 52 L242 52" style="stroke:var(--signal)" stroke-width="2" marker-end="url(#eduArrow)"/>
    <rect x="246" y="30" width="70" height="44" rx="6" style="fill:var(--surface-raised);stroke:var(--accent)"/>
    <text x="281" y="48" text-anchor="middle" font-size="9" style="fill:var(--ink)" font-weight="600">${t.logits}</text>
    <text x="281" y="62" text-anchor="middle" font-size="8" style="fill:var(--muted)">${t.soft}</text>
    <path d="M320 52 L336 52" style="stroke:var(--signal)" stroke-width="2" marker-end="url(#eduArrow)"/>
    <rect x="340" y="30" width="70" height="44" rx="6" style="fill:var(--accent-subtle);stroke:var(--accent)"/>
    <text x="375" y="55" text-anchor="middle" font-size="9" style="fill:var(--ink)" font-weight="600">${t.labels}</text>
  </svg>`;
}

function svgToneSketch(L) {
  const a = L === "es" ? "clickbait" : "clickbait";
  const b = "CAPS / !";
  const c = L === "es" ? "score → etiqueta" : "score → label";
  return `<svg viewBox="0 0 340 80" class="edu-svg" role="img" aria-label="Tone signals">
    <rect x="10" y="20" width="90" height="40" rx="6" style="fill:var(--estado-warn-fondo);stroke:var(--warning)"/>
    <text x="55" y="44" text-anchor="middle" font-size="10" style="fill:var(--estado-warn-texto)">${escapeHtml(a)}</text>
    <rect x="120" y="20" width="90" height="40" rx="6" style="fill:var(--estado-error-fondo);stroke:var(--danger)"/>
    <text x="165" y="44" text-anchor="middle" font-size="10" style="fill:var(--estado-error-texto)">${escapeHtml(b)}</text>
    <rect x="230" y="20" width="90" height="40" rx="6" style="fill:var(--accent-subtle);stroke:var(--accent)"/>
    <text x="275" y="44" text-anchor="middle" font-size="10" style="fill:var(--estado-ok-texto)">${escapeHtml(c)}</text>
  </svg>`;
}

function svgVoteSketch(L) {
  const vote = L === "es" ? "voto" : "vote";
  return `<svg viewBox="0 0 340 90" class="edu-svg" role="img" aria-label="Consensus voting">
    ${markerDefs()}
    <rect x="10" y="28" width="60" height="32" rx="5" style="fill:var(--surface-raised);stroke:var(--accent)"/><text x="40" y="48" text-anchor="middle" font-size="10">NB</text>
    <rect x="85" y="28" width="60" height="32" rx="5" style="fill:var(--surface-raised);stroke:var(--accent)"/><text x="115" y="48" text-anchor="middle" font-size="10">LR</text>
    <rect x="160" y="28" width="60" height="32" rx="5" style="fill:var(--surface-raised);stroke:var(--accent)"/><text x="190" y="48" text-anchor="middle" font-size="10">TR</text>
    <path d="M230 44 L250 44" style="stroke:var(--signal)" stroke-width="2" marker-end="url(#eduArrow)"/>
    <rect x="256" y="22" width="74" height="44" rx="6" style="fill:var(--accent-subtle);stroke:var(--accent)"/>
    <text x="293" y="48" text-anchor="middle" font-size="10" font-weight="600">${escapeHtml(vote)}</text>
  </svg>`;
}

function formatLive(stageId, data) {
  const L = lang;
  if (!data) return `<p class="edu-live-empty">${escapeHtml(UI[L].emptyRun)}</p>`;
  const rows = [];
  const add = (k, v) => {
    if (v === undefined || v === null || v === "") return;
    rows.push(
      `<div class="edu-live-row"><span class="edu-live-k">${escapeHtml(k)}</span><span class="edu-live-v">${escapeHtml(String(v))}</span></div>`
    );
  };

  const labels =
    L === "es"
      ? {
          words: "Palabras",
          chars: "Caracteres",
          tokens: "Tokens",
          sample: "Muestra",
          terms: "Términos activos",
          top: "Top TF-IDF",
          pred: "Predicción",
          conf: "Confianza",
          trunc: "Entrada truncada",
          yes: "sí (~512 tokens)",
          no: "no",
          status: "Estado",
          topic: "Tema",
          topicN: (n) => `Tema ${n}`,
          keywords: "Keywords",
          label: "Etiqueta",
          score: "Score",
          signals: "Señales",
          final: "Final",
          source: "Fuente",
          lex: "Léxico",
          consensus: "Categoría consenso",
          votes: "Votos",
          reason: "Criterio",
          tone: "Tono",
          sentiment: "Sentimiento",
        }
      : {
          words: "Words",
          chars: "Characters",
          tokens: "Tokens",
          sample: "Sample",
          terms: "Active terms",
          top: "Top TF-IDF",
          pred: "Prediction",
          conf: "Confidence",
          trunc: "Truncated input",
          yes: "yes (~512 tokens)",
          no: "no",
          status: "Status",
          topic: "Topic",
          topicN: (n) => `Topic ${n}`,
          keywords: "Keywords",
          label: "Label",
          score: "Score",
          signals: "Signals",
          final: "Final",
          source: "Source",
          lex: "Lexicon",
          consensus: "Consensus category",
          votes: "Votes",
          reason: "Rationale",
          tone: "Tone",
          sentiment: "Sentiment",
        };

  switch (stageId) {
    case "input":
      add(labels.words, data.words);
      add(labels.chars, data.chars);
      break;
    case "preprocess":
      add(labels.tokens, data.nTokens);
      add(labels.sample, Array.isArray(data.sample) ? data.sample.join(" · ") : data.sample);
      break;
    case "vectorize":
      add(labels.terms, data.nTerms);
      add(labels.top, Array.isArray(data.topTerms) ? data.topTerms.join(", ") : data.topTerms);
      break;
    case "temas":
      add(labels.topic, data.topicId != null ? labels.topicN(data.topicId) : null);
      add(labels.keywords, Array.isArray(data.topWords) ? data.topWords.join(", ") : data.topWords);
      add(
        labels.score,
        data.similarity != null ? `${(data.similarity * 100).toFixed(0)}% coseno` : null
      );
      break;
    case "nb":
    case "logreg":
      add(labels.pred, data.label);
      add(labels.conf, data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : null);
      break;
    case "transformer":
      add(labels.pred, data.label);
      add(labels.conf, data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : null);
      add(labels.trunc, data.truncado ? labels.yes : labels.no);
      if (data.error) add(labels.status, data.error);
      break;
    case "sensationalism":
      add(labels.label, data.label);
      add(labels.score, data.score != null ? `${(data.score * 100).toFixed(0)}%` : null);
      add(labels.signals, data.signals);
      break;
    case "sentiment":
      add(labels.final, data.label);
      add(labels.source, data.source);
      add(labels.lex, data.lexLabel);
      add("RoBERTuito", data.onnxLabel || "—");
      break;
    case "consenso":
      add(labels.consensus, data.consensus);
      add(labels.votes, data.votes);
      if (data.reason) add(labels.reason, data.reason);
      add(labels.tone, data.tone);
      add(labels.sentiment, data.sentiment);
      break;
    default:
      Object.entries(data).forEach(([k, v]) => add(k, v));
  }

  if (!rows.length) return `<p class="edu-live-empty">${escapeHtml(UI[L].emptySnap)}</p>`;
  return `<div class="edu-live-grid">${rows.join("")}</div>`;
}

/**
 * Renders educational panel into a container for the given stage.
 */
export function renderTeachPanel(container, stageId) {
  teachRoot = container;
  const entry = EDU[stageId];
  if (!container) return;
  if (!entry) {
    container.innerHTML = "";
    container.hidden = true;
    return;
  }
  const copy = entry[lang] || entry.es;
  const ui = UI[lang] || UI.es;
  container.hidden = false;
  const live = formatLive(stageId, liveStore[stageId]);
  const diagram = typeof entry.svg === "function" ? entry.svg(lang) : entry.svg;
  container.innerHTML = `
    <div class="pipeline-teach" data-teach-stage="${escapeHtml(stageId)}">
      <div class="pipeline-teach-lang">${escapeHtml(ui.banner)}</div>
      <h4 class="pipeline-teach-title">${escapeHtml(copy.title)}</h4>
      <div class="pipeline-teach-diagram">${diagram}</div>
      <div class="pipeline-teach-cols">
        <div class="pipeline-teach-block">
          <h5>${escapeHtml(ui.what)}</h5>
          <p>${escapeHtml(copy.what)}</p>
        </div>
        <div class="pipeline-teach-block">
          <h5>${escapeHtml(ui.why)}</h5>
          <p>${escapeHtml(copy.why)}</p>
        </div>
      </div>
      <div class="pipeline-teach-block">
        <h5>${escapeHtml(ui.how)}</h5>
        <ol class="pipeline-teach-how">${copy.how.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol>
      </div>
      <div class="pipeline-teach-live">
        <h5>${escapeHtml(ui.live)}</h5>
        ${live}
      </div>
    </div>`;
}

/** Refresh live section if the same stage is open. */
export function refreshTeachIf(stageId) {
  if (!teachRoot || teachRoot.hidden) return;
  const open = teachRoot.querySelector(".pipeline-teach");
  if (open && open.getAttribute("data-teach-stage") === stageId) {
    renderTeachPanel(teachRoot, stageId);
  }
}

export function hideTeachPanel(container) {
  const c = container || teachRoot;
  if (!c) return;
  c.innerHTML = "";
  c.hidden = true;
}

export function stageIds() {
  return Object.keys(EDU);
}
