"""Orquesta el entrenamiento de todo el pipeline.

Flujo:
  1. Carga el dataset sintetico (data/*.json).
  2. Entrena TF-IDF + LDA + Naive Bayes + LogisticRegression (modelos clasicos).
  3. Fine-tunea el Transformer (ELECTRA-small) sobre el set de train.
  4. Guarda todos los artefactos en backend/models/.

Uso:
  python backend/train/train_all.py [--no-transformer] [--epochs N]

El flag --no-transformer permite entrenar solo los modelos clasicos (rapido,
sin GPU) para iterar en el pipeline. El Transformer requiere descargar el
modelo base de HuggingFace (~54MB) y tarda varios minutos en CPU.
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path

# Asegurar que backend/ esta en el path.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from pipeline import preprocess, vectorize, topics, models_classic, sentiment, sensationalism
from data.generate_dataset import cargar_dataset, CATEGORIAS

MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"


def cargar_splits():
    train = cargar_dataset(DATA_DIR / "train.json")
    val = cargar_dataset(DATA_DIR / "val.json")
    test = cargar_dataset(DATA_DIR / "test.json")
    return train, val, test


def entrenar_clasicos(train, val, test):
    """Entrena TF-IDF, LDA, NB y LogReg. Devuelve diccionario con artefactos."""
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELOS CLASICOS")
    print("=" * 60)

    textos_train = [n["texto"] for n in train]
    cats_train = [n["categoria"] for n in train]

    # --- TF-IDF + Naive Bayes Multinomial ---
    print("\n[1/4] Naive Bayes Multinomial (TF-IDF)...")
    t0 = time.time()
    nb_pipe = models_classic.crear_nb_multinomial(alpha=0.1, max_features=3000)
    nb_pipe.fit(textos_train, cats_train)
    print(f"      OK ({time.time() - t0:.1f}s) -> backend/models/nb_multinomial.pkl")
    models_classic.guardar_modelo(nb_pipe, MODELS_DIR / "nb_multinomial.pkl")

    # --- Logistic Regression ---
    print("[2/4] Logistic Regression (TF-IDF)...")
    t0 = time.time()
    logreg_pipe = models_classic.crear_logreg(max_iter=1000, C=1.0, max_features=3000)
    logreg_pipe.fit(textos_train, cats_train)
    print(f"      OK ({time.time() - t0:.1f}s) -> backend/models/logreg.pkl")
    models_classic.guardar_modelo(logreg_pipe, MODELS_DIR / "logreg.pkl")

    # --- LDA ---
    print("[3/4] LDA (7 temas)...")
    t0 = time.time()
    textos_full = textos_train + [n["texto"] for n in val]
    lda, vec_lda = topics.entrenar_lda(textos_full, n_components=7, max_iter=50, random_state=42)
    print(f"      OK ({time.time() - t0:.1f}s) -> backend/models/lda.pkl + lda_vec.pkl")
    topics.guardar_lda(lda, vec_lda, MODELS_DIR / "lda.pkl", MODELS_DIR / "lda_vec.pkl")

    # --- Sentimiento ML (sobre sentimiento, no categoria) ---
    print("[4/4] Sentimiento ML (LogReg)...")
    t0 = time.time()
    sents_train = [n["sentimiento"] for n in train]
    sent_pipe = sentiment.entrenar_sentimiento_ml(textos_train, sents_train, max_features=3000)
    print(f"      OK ({time.time() - t0:.1f}s) -> backend/models/sentiment_ml.pkl")
    models_classic.guardar_modelo(sent_pipe, MODELS_DIR / "sentiment_ml.pkl")

    return {
        "nb": nb_pipe,
        "logreg": logreg_pipe,
        "lda": lda,
        "vec_lda": vec_lda,
        "sentiment_ml": sent_pipe,
    }


def entrenar_transformer(train, val, epochs=5, batch_size=16):
    """Fine-tunea ELECTRA-small. Requiere HuggingFace transformers + torch."""
    print("\n" + "=" * 60)
    print("FINE-TUNEANDO TRANSFORMER (ELECTRA-small)")
    print("=" * 60)
    from pipeline import transformer

    textos_train = [n["texto"] for n in train]
    cats_train = [n["categoria"] for n in train]
    textos_val = [n["texto"] for n in val]
    cats_val = [n["categoria"] for n in val]

    save_dir = MODELS_DIR / "transformer"
    history = transformer.entrenar_transformer(
        textos_train,
        cats_train,
        val_texts=textos_val,
        val_labels=cats_val,
        epochs=epochs,
        batch_size=batch_size,
        save_dir=str(save_dir),
    )
    # Guardar historial.
    with open(MODELS_DIR / "transformer_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Transformer guardado en {save_dir}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Entrena todo el pipeline de noticias.")
    parser.add_argument("--no-transformer", action="store_true", help="Saltar fine-tuning del transformer")
    parser.add_argument("--epochs", type=int, default=5, help="Epocas del transformer")
    parser.add_argument("--batch", type=int, default=16, help="Batch size del transformer")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset...")
    train, val, test = cargar_splits()
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    artefactos = entrenar_clasicos(train, val, test)

    if not args.no_transformer:
        entrenar_transformer(train, val, epochs=args.epochs, batch_size=args.batch)
    else:
        print("\n[--no-transformer] Saltando fine-tuning del transformer.")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Artefactos en: {MODELS_DIR}")


if __name__ == "__main__":
    main()
