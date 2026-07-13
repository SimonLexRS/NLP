"""Vectorizacion (Semana 2: TF-IDF y representacion vectorial).

Crea y persiste vectorizadores sklearn:
  - crear_tfidf: TfidfVectorizer (unigramas+bigramas) con el preprocesador custom.
  - crear_bow:   CountVectorizer (bolsa de palabras).

Persistencia con joblib.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .preprocess import preproceso_texto


def crear_tfidf(max_features=3000, ngram_range=(1, 2)):
    """TfidfVectorizer con preprocesador custom (normaliza+tokeniza+lematiza).

    El preprocesador devuelve una cadena de tokens separados por espacios,
    por lo que el token_pattern por defecto de sklearn los separa correctamente.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        preprocessor=preproceso_texto,
        sublinear_tf=True,
    )


def crear_bow(max_features=3000, binary=False):
    """CountVectorizer (BoW). binary=True -> Bernoulli."""
    return CountVectorizer(
        max_features=max_features,
        binary=binary,
        preprocessor=preproceso_texto,
    )


def guardar_vectorizador(vec, ruta):
    joblib.dump(vec, ruta)


def cargar_vectorizador(ruta):
    return joblib.load(ruta)


def exportar_vocab_tfidf(vec, ruta):
    """Exporta el vocabulario (termino -> indice) y los IDF weights a JSON.

    Esto permite reconstruir el vectorizador TF-IDF en JavaScript para inferencia.
    """
    import json

    vocab = vec.vocabulary_  # dict termino -> indice
    idf = vec.idf_.tolist() if hasattr(vec, "idf_") and vec.idf_ is not None else None
    data = {
        "vocabulary": vocab,
        "idf": idf,
        "ngram_range": list(vec.ngram_range),
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Vocab TF-IDF exportado ({len(vocab)} terminos): {ruta}")


if __name__ == "__main__":
    v = crear_tfidf(max_features=100)
    X = v.fit_transform(["buen producto", "mala calidad", "envio rapido"])
    print("Shape:", X.shape)
    print("Vocab size:", len(v.vocabulary_))
