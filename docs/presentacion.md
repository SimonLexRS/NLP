# Presentación — Clasificador de Noticias en Español

**Proyecto Final · Módulo de PLN · Caso 9**
Duración: 8-10 min + preguntas

---

## Slide 1: Portada

**Clasificador de Noticias en Español**
Sistema PLN punta a punta con interfaz web

- Caso 9: Clasificador de noticias
- Maestría en Inteligencia Artificial — UCAB
- Módulo de PLN · 2026

---

## Slide 2: El problema

**¿Qué resuelve?**

Dada una noticia en español (titular + cuerpo), el sistema predice automáticamente:
- **Categoría**: política, economía, deportes, tecnología, salud, internacional, cultura
- **Tono**: informativo vs. sensacionalista (detección de clickbait)
- **Sentimiento**: positivo / negativo / neutro
- **Tema**: asignación por LDA

**¿Para quién?**
- Agregadores de noticias que necesitan categorizar contenido automáticamente
- Medios que quieren monitorizar el tono de sus publicaciones
- Lectores que quieren filtrar contenido sensacionalista

---

## Slide 3: Los datos

**Dataset sintético** de 600 noticias en español, generado por código (`backend/data/generate_dataset.py`).

- **7 categorías** balanceadas (~85 c/u)
- **2 tonos**: 60% informativo / 40% sensacionalista
- **3 sentimientos**: 40% positivo / 40% negativo / 20% neutro
- **Split**: 432 train / 84 val / 84 test (estratificado por categoría)

**Diseño para realismo**: sujetos compartidos entre categorías (ej. "el gobierno" aparece en política, economía, salud...), obligando a los modelos a entender el contexto (acción), no solo palabras clave. Ruido léxico añadido (duplicaciones, relleno neutro).

---

## Slide 4: El pipeline

```
Texto → Preprocesamiento → Vectorización → Modelado
         │                    │              │
         ├─ Normalizar         ├─ TF-IDF      ├─ LDA (temas)
         ├─ Tokenizar          │  (uni+bi)    ├─ Sentimiento
         ├─ Stopwords          │              │   (léxico + RoBERTuito)
         └─ Lematizar          │              ├─ NB + LogReg (clásicos)
                                  │              └─ ELECTRA (neuronal)
                                  └─ Sensacionalismo (reglas)
```

Cada paso corresponde a una semana del módulo:
1. **Semana 1**: Preprocesamiento (spaCy + NLTK)
2. **Semana 2**: TF-IDF + Naive Bayes
3. **Semana 3**: LDA + Análisis de sentimientos
4. **Semana 4**: Transformer (ELECTRA fine-tune)
5. **Semana 5**: Arquitectura + comparación

---

## Slide 5: Preprocesamiento (Semana 1)

`backend/pipeline/preprocess.py` → `web/js/preprocess.js`

1. **Normalizar**: lowercase, URLs→`<url>`, menciones→`<user>`, quitar tildes (unidecode)
2. **Tokenizar**: TweetTokenizer (NLTK) + stopwords español (conservando negaciones: no, ni, nunca, nada, sin...)
3. **Lematizar**: spaCy `es_core_news_sm` (NER y parser desactivados para velocidad)

En JS: lematización por sufijos (ligera) + diccionario de stopwords exportado.

---

## Slide 6: Representación + modelos clásicos (Semana 2-3)

**TF-IDF**: `TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)` con el preprocesador custom.

**Naive Bayes Multinomial** (α=0.1):
- score(clase) = log P(clase) + Σ tfidfᵢ · log P(palabraᵢ | clase)
- En JS: mismos pesos exportados a JSON, inferencia exacta.

**Logistic Regression** (C=1.0, max_iter=1000):
- score(clase) = bias + Σ tfidfᵢ · coef[clase, i]
- Softmax → probabilidad.

**LDA**: 7 temas sobre CountVectorizer. En JS: asignación por similitud coseno contra la matriz tema×término precomputada (sin Gibbs sampling en navegador).

---

## Slide 7: Transformer afinado (Semana 4)

**Modelo**: `mrm8488/electricidad-small-discriminator` (ELECTRA-small en español, ~14MB cuantizado)

**Fine-tuning**: clasificación de 7 categorías
- AdamW lr=2e-4, 8 épocas, batch=16, max_len=128
- Loop PyTorch manual (consistente con el módulo)
- Convergencia: val_acc 100% en época 6

**Exportación a ONNX**: `optimum-cli` + cuantización int8 dinámica → 14.6MB

**Inferencia en navegador**: `transformers.js` (@huggingface/transformers v3) carga el modelo ONNX con `dtype: 'q8'`.

**Sentimiento**: `robertuito-sentiment-analysis-ONNX` (desde CDN de HuggingFace, ~25MB).

---

## Slide 8: Comparación de modelos

| Modelo | Tipo | Accuracy | F1 |
|--------|------|----------|-----|
| Naive Bayes | Clásico | 97.62% | 97.61% |
| Logistic Regression | Clásico | 92.86% | 92.51% |
| **Transformer (ELECTRA)** | **Neuronal** | **100.00%** | **100.00%** |

**Análisis**:
- El Transformer supera a los clásicos (esperado: captura contexto y semántica).
- NB > LogReg porque las palabras distintivas por categoría son fuertes señales léxicas (NB las aprovecha bien con TF-IDF).
- Los errores de los clásicos ocurren en noticias donde el sujeto es ambiguo y la acción no es suficientemente distintiva.

*(Mostrar matrices de confusión)*

---

## Slide 9: DEMO en vivo

**Abrir**: `https://<usuario>.github.io/<repo>/` (o localhost)

**Demo 1 — Noticia informativa**:
> "El banco central reviso las tasas de interes. La informacion fue confirmada por fuentes oficiales."
→ Categoría: economía | Tono: informativo | Sentimiento: neutro

**Demo 2 — Noticia sensacionalista**:
> "NO CREERAS lo que paso. El gobierno anuncio un plan de estímulo. ESCANDALO total !!"
→ Categoría: economía | Tono: sensacionalista | Sentimiento: negativo

**Demo 3 — Caso difícil** (sujeto ambiguo):
> "El presidente sufrio una lesion grave en el ambito de futbol."
→ Mostrar cómo los 3 modelos pueden diferir.

**Demo 4 — Modo lote**: subir CSV y ver distribución.

---

## Slide 10: Desafíos y conclusiones

**Desafíos encontrados**:
- **Dataset sintético trivial**: primera versión daba métricas de 1.0. Solución: compartir sujetos entre categorías + ruido léxico.
- **Transformer no convergía** con lr=5e-5 (loss plana). Solución: subir a lr=2e-4.
- **Lematización en JS**: spaCy no corre en navegador. Solución: lematización por sufijos + diccionario del vocabulario.
- **Tamaño del modelo**: ONNX fp32 = 53MB. Solución: cuantización int8 → 14.6MB.
- **API de cuantización**: optimum cambió `static` → `is_static`. Documentado en el código.

**Conclusiones**:
- Arquitectura 100% estática viable para PLN en producción (GitHub Pages, sin backend).
- transformers.js permite usar modelos Transformer afinados directamente en el navegador.
- La comparación clásico vs. neuronal muestra claramente la ventaja del Transformer.

**Limitaciones**:
- Dataset sintético (no noticias reales).
- Latencia en móvil (~1-3s sin WebGPU).
- Lematización JS simplificada.

**Mejoras futuras**:
- Dataset real (scraping de medios en español).
- WebGPU para acelerar inferencia.
- Análisis por aspecto (de qué se habla dentro de la noticia).
- Clasificación multilabel (una noticia puede tener varias categorías).
