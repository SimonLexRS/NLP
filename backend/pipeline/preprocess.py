"""Pipeline de preprocesamiento (Semana 1: Normalizacion y tokenizacion).

Operaciones:
  1. normalizar_texto: lower, URLs -> <url>, menciones -> <user>, unidecode,
     puntuacion controlada.
  2. tokenizar: TweetTokenizer (NLTK) + stopwords en espanol, conservando
     negaciones (no, ni, nunca, nada, sin, tampoco, jamas).
  3. lematizar: spaCy es_core_news_sm (NER y parser desactivados para velocidad).

El modulo es reutilizable: ``preproceso_texto`` es el preprocesador que se
inyecta en los vectorizadores de sklearn (TF-IDF / BoW).
"""
import re
import os
import json
from functools import lru_cache
from pathlib import Path

# Importaciones opcionales con carga perezosa para no romper el import
# del modulo si una dependencia no esta instalada en el entorno de export.

try:
    from unidecode import unidecode as _unidecode
except Exception:  # pragma: no cover
    _unidecode = None

try:
    import nltk
    from nltk.tokenize import TweetTokenizer
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:  # pragma: no cover
    nltk = None
    TweetTokenizer = None
    nltk_stopwords = None

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None


# Negaciones que SIEMPRE se conservan (no son stopwords para nosotros).
NEGACIONES = {"no", "ni", "nunca", "nada", "sin", "tampoco", "jamas"}

# Patron de URL suelta (sin protocolo) para normalizacion.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENCION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_PUNT_RE = re.compile(r"[^a-z0-9\s<>]")


def _asegurar_recursos_nltk():
    """Descarga silenciosa de recursos NLTK si no estan presentes."""
    if nltk is None:
        return
    for recurso, ruta in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(ruta)
        except LookupError:
            try:
                nltk.download(recurso, quiet=True)
            except Exception:
                pass


@lru_cache(maxsize=1)
def _stopwords_es():
    """Conjunto de stopwords en espanol, conservando negaciones."""
    if nltk_stopwords is None:
        # Fallback minimo si NLTK no esta disponible.
        return _STOPWORDS_FALLBACK
    _asegurar_recursos_nltk()
    try:
        sw = set(nltk_stopwords.words("spanish"))
    except Exception:
        sw = set(_STOPWORDS_FALLBACK)
    sw = {w for w in sw if w not in NEGACIONES}
    return sw


_STOPWORDS_FALLBACK = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero",
    "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mi", "antes", "algunos",
    "que", "unos", "yo", "otro", "otras", "otra", "el", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar",
    "estas", "algunas", "algo", "nosotros", "mi", "mis", "tu", "te", "ti", "tu",
    "tus", "ellas", "nosotras", "vosostros", "vosostras", "os", "mio", "mia",
}


@lru_cache(maxsize=1)
def _nlp():
    """Carga perezosa del modelo spaCy (sin parser ni NER para ir mas rapido)."""
    if spacy is None:
        return None
    try:
        return spacy.load("es_core_news_sm", disable=["ner", "parser"])
    except Exception:
        try:
            spacy.cli.download("es_core_news_sm", False, False, "--quiet")
            return spacy.load("es_core_news_sm", disable=["ner", "parser"])
        except Exception:
            return None


@lru_cache(maxsize=1)
def _tokenizer():
    if TweetTokenizer is None:
        return None
    return TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)


def normalizar_texto(texto: str) -> str:
    """Normaliza el texto: minusculas, URLs, menciones, hashtags y tildes."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = _URL_RE.sub(" url ", texto)
    texto = _MENCION_RE.sub(" user ", texto)
    # Conservar la palabra del hashtag sin el simbolo.
    texto = _HASHTAG_RE.sub(r"\1", texto)
    if _unidecode is not None:
        texto = _unidecode(texto)
    # Normalizar espacios.
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto_normalizado: str):
    """Tokeniza y filtra stopwords (conservando negaciones)."""
    tok = _tokenizer()
    if tok is not None:
        tokens = tok.tokenize(texto_normalizado)
    else:
        tokens = texto_normalizado.split()
    sw = _stopwords_es()
    return [t for t in tokens if t and t not in sw and len(t) > 1]


def lematizar(tokens):
    """Lematiza una lista de tokens usando spaCy.

    Devuelve la lista de lemas. Si spaCy no esta disponible, devuelve los
    tokens sin cambios (fallback aceptable: las palabras raras no suelen
    estar en el vocabulario TF-IDF y no afectan a NB).
    """
    nlp = _nlp()
    if nlp is None or not tokens:
        return list(tokens)
    doc = nlp(" ".join(tokens))
    return [t.lemma_.lower() for t in doc if t.lemma_]


def preproceso(texto: str):
    """Pipeline completo: normalizar -> tokenizar -> lematizar. Devuelve lista de tokens."""
    norm = normalizar_texto(texto)
    toks = tokenizar(norm)
    return lematizar(toks)


def preproceso_texto(texto: str) -> str:
    """Variante que devuelve un string (usada como preprocessor de sklearn)."""
    return " ".join(preproceso(texto))


def exportar_stopwords(ruta: str):
    """Exporta la lista de stopwords a JSON (para usar en JS)."""
    sw = sorted(_stopwords_es())
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(sw, f, ensure_ascii=False, indent=2)
    print(f"Stopwords exportadas ({len(sw)}): {ruta}")


if __name__ == "__main__":
    # Test rapido.
    ej = "El @presidente anuncio en www.gob.bo que NO habrá nuevas medidas!!"
    print("Original:", ej)
    print("Normalizado:", normalizar_texto(ej))
    print("Tokens:", tokenizar(normalizar_texto(ej)))
    print("Preproceso:", preproceso(ej))
