"""Verifica el contrato entre el backend y los assets JSON que consume la web.

Comprueba, sin depender de sklearn ni de ninguna libreria externa, que los
archivos exportados a web/assets/ son consistentes con el dataset y con lo que
esperan los modulos JS (naive_bayes.js, logreg.js, lda.js, app.js, ...).

Uso:
  python3 backend/train/check_assets.py

Salida: imprime cada comprobacion y termina con exit code 1 si alguna falla.
"""
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from data.generate_dataset import CATEGORIAS  # noqa: E402

WEB_ASSETS = BACKEND_DIR.parent / "web" / "assets"
DATA_DIR = BACKEND_DIR / "data"

# Archivos que la web necesita para arrancar (app.js los carga con Promise.all).
OBLIGATORIOS = [
    "nb_weights.json",
    "logreg_weights.json",
    "lda_topics.json",
    "stopwords.json",
    "lexicon.json",
    "sensationalism_rules.json",
    "sample_news.json",
    "metrics.json",
    "notebooks/model_comparison.ipynb",
]

# Claves que consumen naive_bayes.js / logreg.js / lda.js / sentiment_lexicon.js /
# sensationalism.js.
CLAVES = {
    "nb_weights.json": ["type", "labels", "feature_log_prob", "class_log_prior", "vocabulary", "idf", "ngram_range"],
    "logreg_weights.json": ["type", "labels", "coef", "intercept", "vocabulary", "idf", "ngram_range"],
    "lda_topics.json": ["n_topics", "vocabulary", "components", "topics"],
    "lexicon.json": ["lexicon", "negaciones", "intensificadores", "umbral_positivo", "umbral_negativo"],
    "sensationalism_rules.json": ["clickbait_patterns", "palabras_emocionales", "umbral", "ventana_clickbait", "pesos", "caps"],
    "metrics.json": ["labels", "n_test", "modelos", "detalle"],
}

fallos = []


def check(condicion, mensaje):
    if condicion:
        print(f"  [ok]   {mensaje}")
    else:
        print(f"  [FAIL] {mensaje}")
        fallos.append(mensaje)


def cargar(nombre):
    with open(WEB_ASSETS / nombre, encoding="utf-8") as f:
        return json.load(f)


def check_pesos(nombre, clave_clf, clave_bias):
    """Comprobaciones comunes para nb_weights.json y logreg_weights.json."""
    data = cargar(nombre)
    for clave in CLAVES[nombre]:
        check(clave in data, f"{nombre}: clave '{clave}' presente")

    labels = data.get("labels", [])
    check(len(labels) == len(CATEGORIAS), f"{nombre}: {len(CATEGORIAS)} labels (hay {len(labels)})")
    check(set(labels) == set(CATEGORIAS), f"{nombre}: labels == CATEGORIAS del dataset")
    check(labels == sorted(labels), f"{nombre}: labels en orden alfabetico (orden de classes_ de sklearn)")

    vocab = data.get("vocabulary") or {}
    check(len(vocab) > 0, f"{nombre}: vocabulario no vacio ({len(vocab)} terminos)")
    check(all(isinstance(v, int) for v in vocab.values()), f"{nombre}: indices de vocabulario enteros")
    check(sorted(vocab.values()) == list(range(len(vocab))), f"{nombre}: indices de vocabulario contiguos 0..n-1")

    idf = data.get("idf")
    check(isinstance(idf, list) and len(idf) == len(vocab), f"{nombre}: len(idf) == len(vocabulary)")

    matriz = data.get(clave_clf) or []
    check(len(matriz) == len(labels), f"{nombre}: {clave_clf} tiene una fila por label ({len(labels)})")
    check(all(len(fila) == len(vocab) for fila in matriz), f"{nombre}: {clave_clf} tiene una columna por feature")
    bias = data.get(clave_bias)
    check(isinstance(bias, list) and len(bias) == len(labels), f"{nombre}: {clave_bias} tiene un valor por label")

    ngram = data.get("ngram_range")
    check(isinstance(ngram, list) and len(ngram) == 2, f"{nombre}: ngram_range = {ngram}")


def check_lda(data):
    for clave in CLAVES["lda_topics.json"]:
        check(clave in data, f"lda_topics.json: clave '{clave}' presente")
    n_topics = data.get("n_topics")
    vocab = data.get("vocabulary") or {}
    components = data.get("components") or []
    check(n_topics == len(CATEGORIAS), f"lda_topics.json: n_topics == {len(CATEGORIAS)} (hay {n_topics})")
    check(len(components) == n_topics, f"lda_topics.json: components tiene {n_topics} filas")
    check(all(len(fila) == len(vocab) for fila in components), "lda_topics.json: components tiene una columna por termino del vocabulario")
    check(len(data.get("topics", [])) == n_topics, "lda_topics.json: lista 'topics' con un elemento por tema")
    for tema in data.get("topics", []):
        if not tema.get("top_words"):
            fallos.append(f"lda_topics.json: tema {tema.get('id')} sin top_words")
            print(f"  [FAIL] lda_topics.json: tema {tema.get('id')} sin top_words")


def check_metrics(n_test):
    data = cargar("metrics.json")
    for clave in CLAVES["metrics.json"]:
        check(clave in data, f"metrics.json: clave '{clave}' presente")
    check(data.get("n_test") == n_test, f"metrics.json: n_test == {n_test} (hay {data.get('n_test')})")
    check(data.get("labels") == CATEGORIAS, "metrics.json: labels en el orden canonico de CATEGORIAS")
    modelos = data.get("modelos") or []
    check(len(modelos) >= 2, f"metrics.json: compara >= 2 modelos (hay {len(modelos)})")
    check(
        any(m.get("tipo") == "clasico" for m in modelos) and any(m.get("tipo") == "neuronal" for m in modelos),
        "metrics.json: hay al menos un modelo clasico y uno neuronal",
    )
    for m in modelos:
        for metrica in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            valor = m.get(metrica)
            check(isinstance(valor, (int, float)) and 0.0 <= valor <= 1.0, f"metrics.json: {m.get('nombre')}.{metrica} en [0,1] ({valor})")
    for d in data.get("detalle", []):
        per_class = d.get("per_class") or {}
        check(set(per_class) == set(CATEGORIAS), f"metrics.json: per_class de '{d.get('nombre')}' cubre las 7 categorias")
        cm = d.get("confusion_matrix") or []
        check(len(cm) == len(CATEGORIAS) and all(len(f) == len(CATEGORIAS) for f in cm), f"metrics.json: matriz de confusion de '{d.get('nombre')}' es {len(CATEGORIAS)}x{len(CATEGORIAS)}")


def check_sample_news():
    data = cargar("sample_news.json")
    check(len(data) >= 2 * len(CATEGORIAS), f"sample_news.json: >= {2 * len(CATEGORIAS)} ejemplos (hay {len(data)})")
    por_cat = {}
    for n in data:
        por_cat.setdefault(n.get("categoria"), []).append(n.get("tono"))
    check(set(por_cat) == set(CATEGORIAS), "sample_news.json: cubre las 7 categorias")
    for cat, tonos in sorted(por_cat.items()):
        check("informativo" in tonos and "sensacionalista" in tonos, f"sample_news.json: {cat} tiene ejemplo informativo y sensacionalista")


def check_notebook():
    """El visor (notebook_viewer.js) y la descarga consumen el .ipynb."""
    ruta = WEB_ASSETS / "notebooks" / "model_comparison.ipynb"
    if not ruta.exists():
        return  # Ya se avisa en el bloque de obligatorios.
    try:
        data = cargar("notebooks/model_comparison.ipynb")
        check("cells" in data and "nbformat" in data, "model_comparison.ipynb es un notebook valido (cells + nbformat)")
        check(len(data.get("cells", [])) > 0, f"model_comparison.ipynb tiene celdas ({len(data.get('cells', []))})")
    except (ValueError, OSError) as e:
        check(False, f"model_comparison.ipynb no es JSON legible: {e}")


def main():
    print("=" * 60)
    print("VERIFICANDO CONTRATO backend -> web/assets/")
    print("=" * 60)

    if not WEB_ASSETS.exists():
        print(f"[FAIL] No existe {WEB_ASSETS}. Ejecuta: make export")
        return 1

    # 1. Archivos obligatorios.
    print("\n[1/6] Archivos obligatorios")
    for nombre in OBLIGATORIOS:
        check((WEB_ASSETS / nombre).exists(), f"{nombre} existe")

    # 2. Dataset de referencia.
    print("\n[2/6] Dataset")
    test_path = DATA_DIR / "test.json"
    check(test_path.exists(), "backend/data/test.json existe")
    n_test = 0
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            n_test = len(json.load(f))
        check(n_test > 0, f"backend/data/test.json no vacio ({n_test} noticias)")

    # 3. Pesos de modelos clasicos.
    print("\n[3/6] Pesos (nb_weights / logreg_weights)")
    if (WEB_ASSETS / "nb_weights.json").exists():
        check_pesos("nb_weights.json", "feature_log_prob", "class_log_prior")
    if (WEB_ASSETS / "logreg_weights.json").exists():
        check_pesos("logreg_weights.json", "coef", "intercept")

    # 4. LDA.
    print("\n[4/6] Temas LDA")
    if (WEB_ASSETS / "lda_topics.json").exists():
        check_lda(cargar("lda_topics.json"))

    # 5. Metricas.
    print("\n[5/6] Metricas")
    if (WEB_ASSETS / "metrics.json").exists():
        check_metrics(n_test)

    # 6. Ejemplos de demo.
    print("\n[6/6] Ejemplos de noticias")
    if (WEB_ASSETS / "sample_news.json").exists():
        check_sample_news()

    # 7. Notebook del visor.
    print("\n[7/7] Notebook del visor")
    check_notebook()

    print("\n" + "=" * 60)
    if fallos:
        print(f"CONTRATO INCORRECTO: {len(fallos)} comprobaciones fallaron")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print("CONTRATO OK: web/assets/ es consistente con el backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())