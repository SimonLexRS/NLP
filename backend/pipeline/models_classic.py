"""Modelos clasicos (Semana 2: Clasificador Naive Bayes).

Implementa:
  - MultinomialNB sobre TF-IDF (via sklearn.pipeline.make_pipeline)
  - BernoulliNB sobre BoW binario
  - LogisticRegression sobre TF-IDF (modelo adicional para enriquecer la
    comparacion clasico vs neuronal)

Persistencia con joblib. Exportacion de pesos a JSON para inferencia en JS.
"""
import json
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .vectorize import crear_tfidf, crear_bow


# ---------------------------------------------------------------------------
# Constructores de pipelines.
# ---------------------------------------------------------------------------

def crear_nb_multinomial(alpha=0.1, max_features=3000):
    """MultinomialNB sobre TF-IDF."""
    return make_pipeline(crear_tfidf(max_features=max_features), MultinomialNB(alpha=alpha))


def crear_nb_bernoulli(alpha=0.1, max_features=3000):
    """BernoulliNB sobre BoW binario."""
    return make_pipeline(crear_bow(max_features=max_features, binary=True), BernoulliNB(alpha=alpha))


def crear_logreg(max_iter=1000, C=1.0, max_features=3000):
    """LogisticRegression sobre TF-IDF."""
    return make_pipeline(
        crear_tfidf(max_features=max_features),
        LogisticRegression(max_iter=max_iter, C=C, random_state=42),
    )


# ---------------------------------------------------------------------------
# Persistencia.
# ---------------------------------------------------------------------------

def guardar_modelo(modelo, ruta):
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta):
    return joblib.load(ruta)


# ---------------------------------------------------------------------------
# Exportacion a JSON (para inferencia en JavaScript).
# ---------------------------------------------------------------------------

def _extraer_tfidf_y_clasificador(pipe):
    """De un pipeline sklearn [TfidfVectorizer, Clasificador] extrae ambas piezas."""
    vec = pipe.named_steps.get("tfidfvectorizer") or pipe.named_steps.get("countvectorizer")
    clf = pipe.steps[-1][1]
    return vec, clf


def _vocab_a_str(vocab):
    """Convierte las claves int64 de numpy a str para JSON."""
    return {str(k): int(v) for k, v in vocab.items()}


def exportar_nb_multinomial(pipe, etiquetas, ruta):
    """Exporta los parametros de MultinomialNB a JSON.

    MultinomialNB usa:
      - feature_log_prob_: log P(palabra | clase), shape (n_clases, n_features)
      - class_log_prior_:  log P(clase), shape (n_clases,)

    En JS la inferencia es: score(clase) = class_log_prior + sum(tfidf_i * feature_log_prob[clase, i])
    para las palabras presentes. argmax -> clase.
    """
    vec, clf = _extraer_tfidf_y_clasificador(pipe)
    # IMPORTANTE: labels debe seguir el orden de clf.classes_ (orden alfabetico
    # de sklearn), que es el mismo orden de feature_log_prob y class_log_prior.
    # No usar el orden del parametro 'etiquetas' (que viene del generador).
    classes_list = [str(c) for c in clf.classes_.tolist()]
    data = {
        "type": "multinomial_nb",
        "classes": classes_list,
        "labels": classes_list,  # orden coherente con los pesos
        "feature_log_prob": clf.feature_log_prob_.tolist(),
        "class_log_prior": clf.class_log_prior_.tolist(),
        "vocabulary": _vocab_a_str(vec.vocabulary_),
        "idf": vec.idf_.tolist() if hasattr(vec, "idf_") else None,
        "ngram_range": list(vec.ngram_range),
        "alpha": float(clf.alpha),
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"MultinomialNB exportado ({len(clf.classes_)} clases, {len(vec.vocabulary_)} features): {ruta}")


def exportar_logreg(pipe, etiquetas, ruta):
    """Exporta los parametros de LogisticRegression a JSON.

    LogisticRegression: score(clase) = bias[clase] + sum(tfidf_i * coef[clase, i])
    softmax -> probabilidad.
    """
    vec, clf = _extraer_tfidf_y_clasificador(pipe)
    classes_list = [str(c) for c in clf.classes_.tolist()]
    data = {
        "type": "logistic_regression",
        "classes": classes_list,
        "labels": classes_list,  # orden coherente con los pesos
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "vocabulary": _vocab_a_str(vec.vocabulary_),
        "idf": vec.idf_.tolist() if hasattr(vec, "idf_") else None,
        "ngram_range": list(vec.ngram_range),
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"LogisticRegression exportado ({len(clf.classes_)} clases, {len(vec.vocabulary_)} features): {ruta}")


if __name__ == "__main__":
    textos = ["buen producto", "mala calidad", "envio rapido", "pesimo servicio"]
    etiquetas = ["positivo", "negativo", "positivo", "negativo"]
    pipe = crear_nb_multinomial()
    pipe.fit(textos, etiquetas)
    print("Prediccion:", pipe.predict(["no funciona mal"]))
