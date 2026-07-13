"""Evalua los modelos sobre el set de test y genera metricas + matrices de confusion.

Compara:
  - Naive Bayes Multinomial (clasico)
  - Logistic Regression (clasico)
  - Transformer ELECTRA-small afinado (neuronal)

Metricas: accuracy, precision, recall, F1 (macro y por clase).
Matrices de confusion: guardadas como PNG en backend/models/.
Tabla comparativa: backend/models/metrics.json (tambien copiada a web/assets/).
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from data.generate_dataset import cargar_dataset, CATEGORIAS
from pipeline.models_classic import cargar_modelo

MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
WEB_ASSETS = BACKEND_DIR.parent / "web" / "assets"


def cargar_test():
    test = cargar_dataset(DATA_DIR / "test.json")
    textos = [n["texto"] for n in test]
    cats = [n["categoria"] for n in test]
    return textos, cats


def evaluar_clasico(nombre, pipe, textos, y_true, labels):
    """Evalua un modelo clasico (pipeline sklearn)."""
    y_pred = pipe.predict(textos)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "nombre": nombre,
        "tipo": "clasico",
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            labels[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "support": int(sup_per[i]),
            }
            for i in range(len(labels))
        },
        "confusion_matrix": cm.tolist(),
        "y_pred": [str(p) for p in y_pred],
    }


def evaluar_transformer(textos, y_true, labels, model_dir=None):
    """Evalua el transformer fine-tuneado."""
    from pipeline.transformer import cargar_modelo, predecir, ID2LABEL

    if model_dir is None:
        model_dir = MODELS_DIR / "transformer"
    if not Path(model_dir).exists():
        print(f"[evaluate] No se encontro el transformer en {model_dir}, saltando.")
        return None

    print("  Cargando transformer...")
    model, tok = cargar_modelo(model_path=str(model_dir))
    print("  Prediciendo...")
    resultados = predecir(model, tok, textos)
    y_pred = [r["label"] for r in resultados]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "nombre": "Transformer (ELECTRA-small)",
        "tipo": "neuronal",
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            labels[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "support": int(sup_per[i]),
            }
            for i in range(len(labels))
        },
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
    }


def guardar_matriz_confusion(cm, labels, nombre, ruta):
    """Guarda la matriz de confusion como PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [warn] matplotlib no disponible, no se guardara PNG de {nombre}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Matriz de Confusion - {nombre}")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")

    # Anotar celdas.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matriz guardada: {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Evalua modelos sobre el set de test.")
    parser.add_argument("--no-transformer", action="store_true", help="Saltar transformer")
    args = parser.parse_args()

    print("Cargando set de test...")
    textos, y_true = cargar_test()
    labels = CATEGORIAS
    print(f"  Test: {len(textos)} muestras | {len(labels)} categorias")

    resultados = []

    # --- Naive Bayes ---
    print("\n[1/3] Evaluando Naive Bayes Multinomial...")
    nb = cargar_modelo(MODELS_DIR / "nb_multinomial.pkl")
    r_nb = evaluar_clasico("Naive Bayes (Multinomial)", nb, textos, y_true, labels)
    print(f"  Accuracy: {r_nb['accuracy']:.4f} | F1: {r_nb['f1_macro']:.4f}")
    guardar_matriz_confusion(
        np.array(r_nb["confusion_matrix"]),
        labels,
        "Naive Bayes",
        MODELS_DIR / "confusion_nb.png",
    )
    resultados.append(r_nb)

    # --- Logistic Regression ---
    print("\n[2/3] Evaluando Logistic Regression...")
    logreg = cargar_modelo(MODELS_DIR / "logreg.pkl")
    r_lr = evaluar_clasico("Logistic Regression", logreg, textos, y_true, labels)
    print(f"  Accuracy: {r_lr['accuracy']:.4f} | F1: {r_lr['f1_macro']:.4f}")
    guardar_matriz_confusion(
        np.array(r_lr["confusion_matrix"]),
        labels,
        "Logistic Regression",
        MODELS_DIR / "confusion_logreg.png",
    )
    resultados.append(r_lr)

    # --- Transformer ---
    if not args.no_transformer:
        print("\n[3/3] Evaluando Transformer (ELECTRA-small)...")
        r_tr = evaluar_transformer(textos, y_true, labels)
        if r_tr is not None:
            print(f"  Accuracy: {r_tr['accuracy']:.4f} | F1: {r_tr['f1_macro']:.4f}")
            guardar_matriz_confusion(
                np.array(r_tr["confusion_matrix"]),
                labels,
                "Transformer (ELECTRA)",
                MODELS_DIR / "confusion_transformer.png",
            )
            resultados.append(r_tr)
    else:
        print("\n[3/3] [--no-transformer] Saltando transformer.")

    # --- Tabla comparativa ---
    tabla = {
        "labels": labels,
        "n_test": len(textos),
        "modelos": [
            {
                "nombre": r["nombre"],
                "tipo": r["tipo"],
                "accuracy": r["accuracy"],
                "precision_macro": r["precision_macro"],
                "recall_macro": r["recall_macro"],
                "f1_macro": r["f1_macro"],
            }
            for r in resultados
        ],
        "detalle": resultados,
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(tabla, f, ensure_ascii=False, indent=2)
    print(f"\nMetricas guardadas: {metrics_path}")

    # Copiar a web/assets/ para que la interfaz las muestre.
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    web_metrics = WEB_ASSETS / "metrics.json"
    with open(web_metrics, "w", encoding="utf-8") as f:
        # Version sin y_pred (mas ligera para la web).
        web_tabla = {
            "labels": labels,
            "n_test": len(textos),
            "modelos": tabla["modelos"],
            "detalle": [
                {
                    "nombre": r["nombre"],
                    "tipo": r["tipo"],
                    "accuracy": r["accuracy"],
                    "precision_macro": r["precision_macro"],
                    "recall_macro": r["recall_macro"],
                    "f1_macro": r["f1_macro"],
                    "per_class": r["per_class"],
                    "confusion_matrix": r["confusion_matrix"],
                }
                for r in resultados
            ],
        }
        json.dump(web_tabla, f, ensure_ascii=False, indent=2)
    print(f"Metricas (web): {web_metrics}")

    # Copiar matrices de confusion a web/assets/.
    for src_name, dst_name in [
        ("confusion_nb.png", "confusion_nb.png"),
        ("confusion_logreg.png", "confusion_logreg.png"),
        ("confusion_transformer.png", "confusion_transformer.png"),
    ]:
        src = MODELS_DIR / src_name
        if src.exists():
            import shutil

            shutil.copy(src, WEB_ASSETS / dst_name)
            print(f"Matriz copiada a web: {dst_name}")

    # Resumen.
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO")
    print("=" * 60)
    print(f"{'Modelo':<32} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 60)
    for m in tabla["modelos"]:
        print(
            f"{m['nombre']:<32} {m['accuracy']:>8.4f} {m['precision_macro']:>8.4f} "
            f"{m['recall_macro']:>8.4f} {m['f1_macro']:>8.4f}"
        )


if __name__ == "__main__":
    main()
