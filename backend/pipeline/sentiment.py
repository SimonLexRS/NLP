"""Analisis de sentimientos (Semana 3: Analisis de Sentimientos).

Dos enfoques:
  1. sentimiento_lexico: diccionario + ventana de negacion + intensificadores.
  2. entrenar_sentimiento_ml: LogisticRegression sobre TF-IDF (sklearn.pipeline).

En la version web, el sentimiento se calcula en el navegador con el modelo
``robertuito-sentiment-analysis-ONNX`` via transformers.js. Este modulo se
mantiene para el pipeline Python de entrenamiento/evaluacion y para exportar
el lexico (util como fallback y para inspeccion).
"""
import json

# Importacion perezosa de sklearn: solo la usa entrenar_sentimiento_ml, asi el
# lexico y su exportacion funcionan sin sklearn instalado.

# Léxico de polaridad en espanol (palabra -> polaridad en [-1, 1]).
LEX = {
    # Positivas
    "bueno": 0.8, "buena": 0.8, "excelente": 1.0, "genial": 0.9, "increible": 0.9,
    "fantastico": 0.9, "maravilloso": 0.9, "perfecto": 1.0, "positivo": 0.7,
    "optimismo": 0.7, "optimista": 0.7, "avance": 0.6, "ganar": 0.7, "gano": 0.7,
    "crecimiento": 0.7, "exitoso": 0.8, "exito": 0.8, "celebran": 0.7, "celebro": 0.7,
    "apoyo": 0.6, "beneficio": 0.6, "recuperacion": 0.6, "aclamado": 0.7,
    "histórico": 0.6, "historico": 0.6, "homeneje": 0.5, "homenaje": 0.5,
    # Negativas
    "malo": -0.8, "mala": -0.8, "pesimo": -1.0, "terrible": -0.9, "horrible": -0.9,
    "negativo": -0.7, "critica": -0.5, "critico": -0.5, "cuestionaron": -0.6,
    "rechazo": -0.7, "preocupacion": -0.6, "adversos": -0.6, "graves": -0.7,
    "catastrofico": -0.9, "alarmante": -0.7, "devastador": -0.8, "escandalo": -0.7,
    "escandalizado": -0.6, "conmocionado": -0.6, "lesion": -0.6, "falla": -0.6,
    "inflacion": -0.4, "violencia": -0.7, "sanciones": -0.5, "estafa": -0.8,
    "decepcion": -0.7,
    # Neutras / contexto
    "normal": 0.0, "estandar": 0.0, "informacion": 0.0, "confirmada": 0.1,
}

# Negaciones que invierten la polaridad dentro de una ventana de 3 palabras.
NEGACIONES = {"no", "ni", "nunca", "nada", "sin", "tampoco", "jamas"}

# Intensificadores que multiplican la polaridad.
INTENSIFICADORES = {
    "muy": 1.5,
    "super": 1.5,
    "demasiado": 1.3,
    "realmente": 1.2,
    "totalmente": 1.4,
    "absolutamente": 1.5,
    "extremadamente": 1.6,
}


def _normalizar_palabra(p: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(p.lower())
    except Exception:
        return p.lower().translate(str.maketrans("áéíóúñü", "aeiounu"))


def sentimiento_lexico(texto: str) -> dict:
    """Analiza el sentimiento de un texto con el enfoque lexico.

    Reglas:
      - Ventana de negacion de 3 palabras: si una negacion aparece hasta 3
        palabras antes de una palabra de polaridad, se invierte el signo.
      - Intensificadores: multiplican la polaridad de la palabra siguiente.
    """
    if not isinstance(texto, str) or not texto.strip():
        return {"label": "neutro", "score": 0.0, "polaridad": 0.0}

    palabras = [_normalizar_palabra(p) for p in texto.split()]
    polaridad = 0.0
    n_palabras_lex = 0

    for i, p in enumerate(palabras):
        if p in LEX:
            pol = LEX[p]
            # Verificar negacion en ventana de 3 palabras hacia atras.
            for j in range(max(0, i - 3), i):
                if palabras[j] in NEGACIONES:
                    pol = -pol
                    break
            # Verificar intensificador en la palabra inmediatamente anterior.
            if i > 0 and palabras[i - 1] in INTENSIFICADORES:
                pol *= INTENSIFICADORES[palabras[i - 1]]
            polaridad += pol
            n_palabras_lex += 1

    # Normalizar por numero de palabras con polaridad (o por longitud total).
    if n_palabras_lex > 0:
        polaridad_norm = polaridad / n_palabras_lex
    else:
        polaridad_norm = 0.0

    # Etiqueta segun umbral.
    if polaridad_norm > 0.15:
        label = "positivo"
    elif polaridad_norm < -0.15:
        label = "negativo"
    else:
        label = "neutro"

    # Score en [0, 1] para la intensidad (valor absoluto acotado).
    score = min(abs(polaridad_norm), 1.0)

    return {
        "label": label,
        "score": round(score, 4),
        "polaridad": round(polaridad_norm, 4),
    }


def entrenar_sentimiento_ml(textos, etiquetas, max_features=3000):
    """LogisticRegression sobre TF-IDF para sentimiento (3 clases)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from .vectorize import crear_tfidf

    pipe = make_pipeline(
        crear_tfidf(max_features=max_features),
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    )
    pipe.fit(textos, etiquetas)
    return pipe


def exportar_lexico(ruta: str):
    """Exporta el lexico + negaciones + intensificadores a JSON (para JS)."""
    data = {
        "lexicon": {k: v for k, v in LEX.items()},
        "negaciones": sorted(NEGACIONES),
        "intensificadores": INTENSIFICADORES,
        "umbral_positivo": 0.15,
        "umbral_negativo": -0.15,
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Lexico de sentimiento exportado ({len(LEX)} entradas): {ruta}")


if __name__ == "__main__":
    ej1 = "El gobierno celebro el crecimiento economico, muy positivo."
    ej2 = "Criticos cuestionaron la falta de planificacion, graves consecuencias."
    ej3 = "El banco central reviso las tasas de interes."
    print("Pos:", sentimiento_lexico(ej1))
    print("Neg:", sentimiento_lexico(ej2))
    print("Neu:", sentimiento_lexico(ej3))
