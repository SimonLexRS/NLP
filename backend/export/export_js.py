"""Exporta los pesos de los modelos clasicos a JSON para inferencia en JavaScript.

Genera en web/assets/:
  - nb_weights.json        (MultinomialNB: feature_log_prob, class_log_prior, vocab, idf)
  - logreg_weights.json    (LogisticRegression: coef, intercept, vocab, idf)
  - lda_topics.json        (LDA: matriz tema x termino, vocabulario, top palabras)
  - tfidf_vocab.json       (vocabulario TF-IDF comun + idf)
  - stopwords.json         (stopwords en espanol)
  - lexicon.json           (lexico de sentimiento + negaciones + intensificadores)
  - sensationalism_rules.json (patrones clickbait + palabras emocionales)
  - sample_news.json       (ejemplos de noticias para demo)
"""
import sys
import os
import json
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from pipeline import preprocess, vectorize, topics, models_classic, sentiment, sensationalism
from pipeline.models_classic import cargar_modelo
from data.generate_dataset import cargar_dataset, CATEGORIAS

MODELS_DIR = BACKEND_DIR / "models"
WEB_ASSETS = BACKEND_DIR.parent / "web" / "assets"
DATA_DIR = BACKEND_DIR / "data"


def exportar_todo():
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPORTANDO PESOS A JSON (web/assets/)")
    print("=" * 60)

    # --- Naive Bayes ---
    print("\n[1/7] Naive Bayes Multinomial...")
    nb = cargar_modelo(MODELS_DIR / "nb_multinomial.pkl")
    models_classic.exportar_nb_multinomial(nb, CATEGORIAS, WEB_ASSETS / "nb_weights.json")

    # --- Logistic Regression ---
    print("[2/7] Logistic Regression...")
    logreg = cargar_modelo(MODELS_DIR / "logreg.pkl")
    models_classic.exportar_logreg(logreg, CATEGORIAS, WEB_ASSETS / "logreg_weights.json")

    # --- LDA ---
    print("[3/7] LDA temas...")
    lda, vec_lda = topics.cargar_lda(MODELS_DIR / "lda.pkl", MODELS_DIR / "lda_vec.pkl")
    topics.exportar_temas_json(lda, vec_lda, WEB_ASSETS / "lda_topics.json", n_palabras=15)

    # --- Stopwords ---
    print("[4/7] Stopwords...")
    preprocess.exportar_stopwords(WEB_ASSETS / "stopwords.json")

    # --- Lexico de sentimiento ---
    print("[5/7] Lexico de sentimiento...")
    sentiment.exportar_lexico(WEB_ASSETS / "lexicon.json")

    # --- Reglas de sensacionalismo ---
    print("[6/7] Reglas de sensacionalismo...")
    sensationalism.exportar_reglas(WEB_ASSETS / "sensationalism_rules.json")

    # --- Ejemplos de noticias para demo ---
    print("[7/7] Ejemplos de noticias...")
    full = cargar_dataset(DATA_DIR / "full.json")
    # Seleccionar 1 ejemplo informativo y 1 sensacionalista por categoria.
    ejemplos = []
    for cat in CATEGORIAS:
        info = next((n for n in full if n["categoria"] == cat and n["tono"] == "informativo"), None)
        sens = next((n for n in full if n["categoria"] == cat and n["tono"] == "sensacionalista"), None)
        if info:
            ejemplos.append(info)
        if sens:
            ejemplos.append(sens)
    with open(WEB_ASSETS / "sample_news.json", "w", encoding="utf-8") as f:
        json.dump(ejemplos, f, ensure_ascii=False, indent=2)
    print(f"  {len(ejemplos)} ejemplos exportados -> web/assets/sample_news.json")

    # --- Plantilla CSV para modo lote ---
    examples_dir = BACKEND_DIR.parent / "web" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    csv_path = examples_dir / "batch_template.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,texto\n")
        for i, n in enumerate(ejemplos[:8]):
            # Escapar comas y comillas en CSV.
            texto = n["texto"].replace('"', '""')
            f.write(f'{i},"{texto}"\n')
    print(f"  Plantilla CSV -> web/examples/batch_template.csv")

    print("\n" + "=" * 60)
    print("EXPORTACION COMPLETADA")
    print("=" * 60)
    print(f"Archivos en: {WEB_ASSETS}")
    for f in sorted(WEB_ASSETS.glob("*.json")):
        size = f.stat().st_size
        print(f"  {f.name:<30} {size:>8,} bytes")


if __name__ == "__main__":
    exportar_todo()
