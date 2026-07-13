"""Pipeline NLP modular para el Clasificador de Noticias (Caso 9).

Cada modulo corresponde a una semana del modulo de PLN:
  - preprocess:     Semana 1 - Normalizacion y tokenizacion
  - vectorize:      Semana 2 - TF-IDF y representacion vectorial
  - topics:         Semana 3 - LDA (modelado de temas)
  - sentiment:      Semana 3 - Analisis de sentimientos (lexico + ML)
  - models_classic: Semana 2 - Naive Bayes + LogisticRegression
  - transformer:    Semana 4 - Transformer afinado (ELECTRA-small)
  - sensationalism: Reglas de tono sensacionalista (clickbait)
"""
