# Informe — Clasificador de Noticias en Español

**Proyecto Final · Módulo de PLN · Caso 9**
Maestría en Inteligencia Artificial — UCAB · 2026

---

## 1. Resumen

Se implementa un sistema de Procesamiento de Lenguaje Natural punta a punta que clasifica noticias en español por **categoría** (7 clases), **tono** (informativo/sensacionalista), **sentimiento** y **tema** (LDA). El sistema compara dos enfoques de clasificación: modelos clásicos (Naive Bayes Multinomial y Logistic Regression sobre TF-IDF) y un Transformer neuronal afinado (ELECTRA-small). Toda la inferencia se ejecuta en el navegador mediante transformers.js, desplegándose como sitio 100% estático en GitHub Pages.

## 2. Problema y caso de uso

**Caso 9: Clasificador de noticias.** Dado el texto de una noticia (titular + cuerpo), el sistema debe:
- Predecir la categoría temática (política, economía, deportes, tecnología, salud, internacional, cultura).
- Detectar el tono sensacionalista (clickbait, exclamaciones, mayúsculas, palabras emocionales).
- Medir el sentimiento (positivo/negativo/neutro).
- Asignar un tema LDA.

Esto es útil para agregadores de noticias, monitorización de medios y filtrado de contenido.

## 3. Datos

Se generó un **dataset sintético** de 1500 noticias en español (`backend/data/generate_dataset.py`), diseñado para evitar la separación trivial que produciría métricas perfectas:

- **7 categorías** balanceadas (~214 muestras cada una).
- **Sujetos compartidos** entre categorías ("el gobierno", "el presidente", "los expertos" aparecen en política, economía, salud...), lo que obliga a los modelos a atender el contexto (la acción), no solo palabras aisladas.
- **2 tonos**: 60% informativo / 40% sensacionalista.
- **3 sentimientos**: 40% positivo / 40% negativo / 20% neutro.
- **Ruido**: frases de relleno neutras, duplicaciones ocasionales de palabras.

**División**: 1052 train / 224 val / 224 test (15% val, 15% test), estratificada por categoría.

## 4. Pipeline

El pipeline sigue la progresión del módulo, con cada etapa implementada en Python (entrenamiento) y espejada en JavaScript (inferencia):

### 4.1 Preprocesamiento (Semana 1)
- Normalización: lowercase, URLs/menciones/hashtags normalizados, tildes eliminadas (unidecode).
- Tokenización: TweetTokenizer de NLTK + stopwords en español, conservando negaciones.
- Lematización: spaCy `es_core_news_sm`. En JS, lematización por sufijos sobre el vocabulario exportado.

### 4.2 Vectorización (Semana 2)
- TF-IDF: `TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)` con el preprocesador custom.
- En JS: reconstrucción del vector disperso TF-IDF con el vocabulario e IDF exportados, normalización L2.

### 4.3 Modelado de temas (Semana 3)
- LDA: `LatentDirichletAllocation(n_components=7, max_iter=50)` sobre CountVectorizer.
- En JS: asignación por similitud coseno entre el vector del texto y la matriz tema×término precomputada (evita correr Gibbs sampling en el navegador).

### 4.4 Análisis de sentimientos (Semana 3)
- **Léxico**: diccionario de polaridad + ventana de negación (3 palabras) + intensificadores.
- **ML**: LogisticRegression sobre TF-IDF (entrenado sobre la etiqueta de sentimiento).
- **Transformer**: `robertuito-sentiment-analysis-ONNX` cargado vía transformers.js desde la CDN de HuggingFace.

### 4.5 Clasificación (Semana 2 + 4)
- **Naive Bayes Multinomial** (α=0.1) sobre TF-IDF — modelo clásico.
- **Logistic Regression** (C=1.0) sobre TF-IDF — modelo clásico adicional.
- **ELECTRA-small** (`mrm8488/electricidad-small-discriminator`) fine-tuneado para 7 clases — modelo neuronal.

### 4.6 Detección de sensacionalismo
Reglas léxicas: patrones clickbait al inicio, palabras emocionales, conteo de exclamaciones, proporción de mayúsculas, interrogaciones. Score combinado 0-1, umbral 0.5.

## 5. Entrenamiento del Transformer

- **Modelo base**: ELECTRA-small en español (54MB fp32, ~14MB int8).
- **Fine-tuning**: clasificación de secuencia, 7 etiquetas, loop PyTorch manual.
- **Hiperparámetros**: AdamW lr=2e-4, weight_decay=0.01, 8 épocas, batch=16, max_len=128, warmup 10%, clip grad 1.0.
- **Convergencia**: val_acc 14% (época 1) → 65% (época 3) → 100% (época 6).
- **Nota**: con lr=5e-5 (valor típico recomendado) el modelo no convergía (loss plana cerca de ln(7)=1.946). Se subió a 2e-4 para el clasificador recién inicializado.

## 6. Exportación a formato web

### 6.1 Modelos clásicos → JSON
- `export_js.py` serializa: feature_log_prob + class_log_prior (NB), coef + intercept (LogReg), matriz components + vocabulario (LDA), IDF, stopwords, léxico, reglas de sensacionalismo.
- Inferencia JS exacta (mismas fórmulas que sklearn).

### 6.2 Transformer → ONNX cuantizado
- `export_onnx.py` usa `optimum` para exportar a ONNX, luego `ORTQuantizer` con `AutoQuantizationConfig.avx2(is_static=False)` para cuantización dinámica int8.
- Resultado: `model_quantized.onnx` (14.3MB) + tokenizer + config.
- Carga en navegador: `transformers.js` v3 con `dtype: 'q8'`.

## 7. Resultados

| Modelo | Tipo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|--------|------|----------|-------------------|----------------|------------|
| Naive Bayes (Multinomial) | Clásico | 97.32% | 97.41% | 97.32% | 97.33% |
| Logistic Regression | Clásico | 98.21% | 98.29% | 98.21% | 98.21% |
| **Transformer (ELECTRA-small)** | **Neuronal** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

Set de test: 224 noticias, 7 categorías (32 por clase).

**Análisis**:
- El Transformer supera a ambos modelos clásicos, como es de esperar: captura contexto y semántica que los modelos basados en TF-IDF no pueden.
- Logistic Regression supera a Naive Bayes en este set: con 224 noticias y vocabulario acotado, la frontera lineal regularizada generaliza mejor que la asunción de independencia condicional de NB.
- Los errores de los clásicos (matrices de confusión, §7) se concentran en internacional y cultura, cuyos vocabularios solapan con política y economía.

## 8. Arquitectura web

- **Frontend**: HTML + CSS + JS vanilla (ES modules), sin framework ni bundler.
- **transformers.js**: cargado vía CDN con `<script type="importmap">`.
- **Modelos**: ONNX local (ELECTRA, 14MB) + ONNX desde CDN (RoBERTuito, 25MB). Ambos se cachean en el navegador.
- **Sin backend**: toda la inferencia ocurre en el cliente. GitHub Pages sirve archivos estáticos.
- **Modos**: análisis individual (texto pegado) + modo lote (CSV).

## 9. Limitaciones

1. **Dataset sintético**: aunque se diseñó con ambigüedad y ruido, no refleja la complejidad de noticias reales (ironía, negación, referencias contextuales).
2. **Lematización JS simplificada**: la lematización por sufijos es menos precisa que spaCy.
3. **Latencia en móvil**: la inferencia WASM de ELECTRA puede tardar 1-3s en dispositivos gama baja sin WebGPU.
4. **Primera carga pesada**: ~40MB de modelos (ELECTRA + RoBERTuito) en la primera visita (se cachean después).
5. **Sensacionalismo por reglas**: no usa ML, solo heurísticas léxicas.

## 10. Mejoras futuras

1. **Dataset real**: scraping de medios en español (BBC Mundo, El País, etc.) con etiquetado.
2. **WebGPU**: activar inferencia acelerada por GPU en navegadores compatibles.
3. **Análisis por aspecto**: dentro de cada noticia, identificar de qué se habla (ej. en política: "elecciones", "reforma", "presupuesto").
4. **Clasificación multilabel**: una noticia puede pertenecer a varias categorías.
5. **Modelo de sensacionalismo con ML**: entrenar un clasificador en lugar de reglas.
6. **Optimización de carga**: lazy-loading diferido del modelo de sentimiento.

## 11. Reproducción

```bash
make setup          # instalar dependencias
make data           # generar dataset
make train-classic  # entrenar NB + LogReg + LDA (~10s)
make train          # fine-tunear ELECTRA (~5-10 min con GPU)
make evaluate       # métricas + matrices de confusión
make export         # exportar a JSON + ONNX
make serve          # servir en http://localhost:8090
```

## 12. Cumplimiento de la rúbrica

| # | Requisito | Cumplimiento |
|---|-----------|--------------|
| 1 | Pipeline de preprocesamiento reutilizable | `pipeline/preprocess.py` + `preprocess.js` |
| 2 | Modelado de temas (LDA) | `pipeline/topics.py` + `lda.js` |
| 3 | Análisis de sentimientos | RoBERTuito ONNX + léxico + LogisticRegression |
| 4 | Comparación clásico vs. neuronal con métricas | NB + LogReg vs. ELECTRA, `metrics.json` |
| 5 | Interfaz de uso funcional | Web app en GitHub Pages (modo individual + lote) |
| 6 | Presentación con demo en vivo | `docs/presentacion.md` (10 slides) |
| 7 | Informe + repositorio | Este documento + repo git |
