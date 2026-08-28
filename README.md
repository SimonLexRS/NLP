# 📰 Clasificador de Noticias en Español — Proyecto Final PLN

Sistema de Procesamiento de Lenguaje Natural **punta a punta** que clasifica noticias en español por **categoría**, **tono sensacionalista**, **sentimiento** y **tema**, comparando modelos clásicos (Naive Bayes, Logistic Regression) con un Transformer afinado (ELECTRA-small).

**Caso de uso 9** del proyecto final del Módulo de PLN.

---

## ✨ Características

- **7 categorías**: política, economía, deportes, tecnología, salud, internacional, cultura
- **Tono**: informativo vs. sensacionalista (detección de clickbait)
- **Sentimiento**: positivo / negativo / neutro (RoBERTuito + léxico)
- **Temas**: modelado LDA con 7 temas
- **Comparación de modelos**: Naive Bayes + Logistic Regression (clásicos) vs. ELECTRA-small (neuronal)
- **100% estático**: toda la inferencia ocurre en el navegador, sin backend
- **Desplegable en GitHub Pages** (github.io)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│  Navegador (GitHub Pages)                                │
│                                                          │
│  HTML/CSS/JS vanilla                                     │
│  ├── preprocess.js    (normalizar, tokenizar, lematizar) │
│  ├── vectorize.js     (TF-IDF)                           │
│  ├── naive_bayes.js   (inferencia desde pesos JSON)      │
│  ├── logreg.js        (inferencia desde pesos JSON)      │
│  ├── lda.js           (similitud coseno)                 │
│  ├── sensationalism.js (reglas léxicas)                  │
│  ├── sentiment_lexicon.js (léxico + negación)            │
│  └── transformer.js   (ONNX vía transformers.js)         │
│       ├── ELECTRA-small fine-tuneado (~14MB, local)      │
│       └── RoBERTuito sentimiento (~25MB, CDN HF)         │
│                                                          │
│  assets/                                                  │
│  ├── model_onnx/        (modelo ONNX cuantizado)         │
│  ├── nb_weights.json    (pesos Naive Bayes)              │
│  ├── logreg_weights.json (pesos Logistic Regression)     │
│  ├── lda_topics.json    (matriz tema×término)            │
│  ├── metrics.json       (tabla comparativa)              │
│  └── ...                                                 │
└─────────────────────────────────────────────────────────┘

          ▲ generados por (offline, no se despliega) ▲

┌─────────────────────────────────────────────────────────┐
│  backend/ (Python — solo entrenamiento/exportación)      │
│  ├── pipeline/                                           │
│  │   ├── preprocess.py  (Semana 1: spaCy + NLTK)        │
│  │   ├── vectorize.py   (Semana 2: TF-IDF sklearn)      │
│  │   ├── topics.py      (Semana 3: LDA sklearn)          │
│  │   ├── sentiment.py   (Semana 3: léxico + LogReg)     │
│  │   ├── models_classic.py (Semana 2: NB + LogReg)      │
│  │   ├── transformer.py (Semana 4: ELECTRA fine-tune)   │
│  │   └── sensationalism.py (reglas clickbait)            │
│  ├── train/                                              │
│  │   ├── train_all.py   (orquesta entrenamiento)         │
│  │   └── evaluate.py    (métricas + matrices)            │
│  └── export/                                             │
│      ├── export_js.py   (pesos -> JSON)                  │
│      └── export_onnx.py (modelo -> ONNX cuantizado)      │
└─────────────────────────────────────────────────────────┘
```

## 📊 Resultados

| Modelo | Tipo | Accuracy | Precision | Recall | F1 |
|--------|------|----------|-----------|--------|-----|
| Naive Bayes (Multinomial) | Clásico | 97.32% | 97.41% | 97.32% | 97.33% |
| Logistic Regression | Clásico | 98.21% | 98.29% | 98.21% | 98.21% |
| **Transformer (ELECTRA-small)** | **Neuronal** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

Set de test: 224 noticias, 7 categorías.

El Transformer (neuronal) supera a los modelos clásicos, cumpliendo el requisito de comparación clásico vs. neuronal. Entre los clásicos, Logistic Regression supera a Naive Bayes en este set.

## 🚀 Reproducción

### Requisitos
- Python 3.10+
- GPU opcional (acelera el fine-tuning del Transformer, pero corre en CPU)

### Pasos

```bash
# 1. Instalar dependencias
make setup

# 2. Generar dataset sintético (1500 noticias)
make data

# 3. Entrenar modelos clásicos (rápido, ~10s)
make train-classic

# 4. Entrenar Transformer (requiere GPU recomendada, ~5-10 min)
make train

# 5. Evaluar y generar métricas + matrices de confusión
make evaluate

# 6. Exportar modelos a JSON + ONNX (para la web)
make export

# 7. Probar localmente
make serve
# → http://localhost:8090
```

### Despliegue en GitHub Pages (github.io)

**Repositorio:** [SimonLexRS/NLP](https://github.com/SimonLexRS/NLP)  
**Sitio en vivo:** [https://simonlexrs.github.io/NLP/](https://simonlexrs.github.io/NLP/)

El workflow [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) publica la carpeta `web/` (incluye `.nojekyll` y el ONNX de ELECTRA ~14 MB).

1. En el repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push a `main` dispara el deploy automáticamente.
3. Tras el workflow *Deploy to GitHub Pages*, el sitio queda en `https://simonlexrs.github.io/NLP/`.

**Notas:**
- Sirve siempre por **HTTPS** (caché del navegador para modelos ONNX disponible).
- No abras el `index.html` con `file://`: la Cache API no funciona y transformers.js mostrará avisos.
- Local: `make serve` → `http://localhost:8090` (contexto seguro).
- Las rutas de assets son relativas a la página; funcionan en project pages (`/NLP/`).

## 📁 Estructura

```
Appv2/
├── backend/              # Pipeline Python (entrenamiento, no se despliega)
│   ├── data/             # Dataset sintético + generador
│   ├── pipeline/         # Módulos NLP por semana
│   ├── train/            # Entrenamiento + evaluación
│   ├── export/           # Exportación a JSON/ONNX
│   └── models/           # Artefactos entrenados (.pkl, .pt, .onnx)
├── web/                  # Sitio estático (se publica a GitHub Pages)
│   ├── index.html        # SPA
│   ├── css/              # Estilos
│   ├── js/               # Módulos de inferencia (ES modules)
│   ├── assets/           # Modelos ONNX + pesos JSON + métricas
│   └── examples/         # Plantilla CSV
├── docs/                 # Presentación + informe
├── notebooks/            # Notebook de comparación
├── .github/workflows/    # CI/CD GitHub Pages
└── Makefile              # Automatización
```

## 🔧 Pipeline NLP (por semana del módulo)

| Semana | Técnica | Implementación |
|--------|---------|----------------|
| 1 | Normalización y tokenización | `preprocess.py` / `preprocess.js` (spaCy + NLTK) |
| 2 | TF-IDF + Naive Bayes | `vectorize.py`, `models_classic.py` / `vectorize.js`, `naive_bayes.js` |
| 3 | LDA + Análisis de sentimientos | `topics.py`, `sentiment.py` / `lda.js`, `sentiment_lexicon.js` |
| 4 | Transformers | `transformer.py` (ELECTRA fine-tune) / `transformer.js` (ONNX) |
| 5 | Arquitectura + Pipelines | Comparación clásico vs. neuronal sobre el mismo test |

## 📝 Cumplimiento de la rúbrica

| # | Requisito | Estado |
|---|-----------|--------|
| 1 | Pipeline de preprocesamiento reutilizable | ✅ `pipeline/preprocess.py` + `preprocess.js` |
| 2 | Modelado de temas (LDA) | ✅ `pipeline/topics.py` + `lda.js` |
| 3 | Análisis de sentimientos | ✅ RoBERTuito ONNX + léxico |
| 4 | Comparación clásico vs. neuronal | ✅ NB + LogReg vs. ELECTRA, métricas en `metrics.json` |
| 5 | Interfaz de uso funcional | ✅ Web app en GitHub Pages |
| 6 | Presentación con demo en vivo | ✅ `docs/presentacion.md` |
| 7 | Informe + repositorio | ✅ `docs/informe.md` + repo git |

## 👤 Autor

**Simon Alex Rodriguez** — [LinkedIn](https://www.linkedin.com/in/srodriguezxs/) · [GitHub](https://github.com/SimonLexRS)

Docente del módulo: [Kenji Kawaida](https://www.linkedin.com/in/kenji-kawaida/) · UCB San Pablo

## 📜 Licencia

Proyecto académico — Maestría en Inteligencia Artificial, Universidad Católica Boliviana "San Pablo", 2026.
