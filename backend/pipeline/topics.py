"""Modelado de temas con LDA (Semana 3: Embedding y Modelos de Temas LDA).

Usa LatentDirichletAllocation de sklearn sobre CountVectorizer. Se exporta la
matriz tema x termino para que en JavaScript se pueda asignar el tema de un
texto nuevo por similitud coseno (sin correr Gibbs sampling en el navegador).
"""
import json
import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .preprocess import preproceso_texto


def crear_lda(n_components=7, max_iter=50, random_state=42):
    """LDA con learning_method batch (mas estable para corpus pequeno)."""
    return LatentDirichletAllocation(
        n_components=n_components,
        learning_method="batch",
        max_iter=max_iter,
        random_state=random_state,
        evaluate_every=5,
        verbose=0,
    )


def crear_vec_lda(max_features=2000):
    """CountVectorizer para LDA (sin TF-IDF, solo conteos)."""
    return CountVectorizer(
        max_features=max_features,
        preprocessor=preproceso_texto,
    )


def entrenar_lda(textos, n_components=7, max_iter=50, random_state=42):
    """Entrena LDA + su vectorizador. Devuelve (lda, vec_lda)."""
    vec = crear_vec_lda()
    X = vec.fit_transform(textos)
    lda = crear_lda(n_components=n_components, max_iter=max_iter, random_state=random_state)
    lda.fit(X)
    return lda, vec


def extraer_temas(lda, vec_lda, n_palabras=10):
    """Devuelve lista de temas con sus top palabras (para inspeccion)."""
    vocab_inv = {i: t for t, i in vec_lda.vocabulary_.items()}
    temas = []
    for comp in lda.components_:
        top_idx = comp.argsort()[-n_palabras:][::-1]
        palabras = [(vocab_inv[i], float(comp[i])) for i in top_idx]
        temas.append(palabras)
    return temas


def predecir_tema(lda, vec_lda, texto):
    """Predice la distribucion de temas para un texto (vector denso)."""
    X = vec_lda.transform([texto])
    dist = lda.transform(X)[0]
    return dist.tolist()


def guardar_lda(lda, vec_lda, ruta_lda, ruta_vec):
    joblib.dump(lda, ruta_lda)
    joblib.dump(vec_lda, ruta_vec)


def cargar_lda(ruta_lda, ruta_vec):
    return joblib.load(ruta_lda), joblib.load(ruta_vec)


def exportar_temas_json(lda, vec_lda, ruta, n_palabras=15):
    """Exporta la matriz tema x termino y el vocabulario a JSON.

    Formato:
      {
        "topics": [
          {"id": 0, "top_words": [["palabra", peso], ...]},
          ...
        ],
        "components": [[w0, w1, ...], ...],  # matriz tema x termino
        "vocabulary": {"palabra": indice, ...},  # vocabulario LDA
        "n_topics": 7
      }

    En JS se puede asignar el tema de un texto vectorizandolo sobre el mismo
    vocabulario y calculando similitud coseno contra cada fila de ``components``.
    """
    vocab = vec_lda.vocabulary_
    vocab_inv = {i: t for t, i in vocab.items()}
    components = lda.components_.tolist()
    temas = extraer_temas(lda, vec_lda, n_palabras=n_palabras)
    data = {
        "n_topics": lda.n_components,
        "vocabulary": {str(k): int(v) for k, v in vocab.items()},
        "vocabulary_inv": {str(i): vocab_inv[i] for i in range(len(vocab_inv))},
        "components": components,
        "topics": [
            {"id": i, "top_words": [[w, p] for w, p in tema]}
            for i, tema in enumerate(temas)
        ],
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Temas LDA exportados ({lda.n_components} temas, {len(vocab)} terminos): {ruta}")


if __name__ == "__main__":
    textos = [
        "el gobierno anuncio nuevas medidas economicas",
        "el seleccionado gano el partido de futbol",
        "nueva vacuna contra el virus aprobada",
        "la bolsa cayo por la inflacion",
        "el festival de cine premió a la pelicula",
    ]
    lda, vec = entrenar_lda(textos, n_components=3, max_iter=10)
    temas = extraer_temas(lda, vec, n_palabras=5)
    for i, t in enumerate(temas):
        print(f"Tema {i}:", [w for w, _ in t])
