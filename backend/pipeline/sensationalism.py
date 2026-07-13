"""Deteccion de tono sensacionalista (Semana 3: Analisis de Sentimientos + reglas).

Heuristica basada en reglas lexicas sobre el texto crudo:
  - Patrones clickbait al inicio ("NO CREERAS", "ESCANDALO", "Te sorprendera"...).
  - Palabras emocionales (conmocionado, impactante, bomba, catastrofico...).
  - Conteo de signos de exclamacion.
  - Proporcion de mayusculas.
  - Signos de interrogacion sensacionalistas.

Devuelve un score 0-1 y una etiqueta (informativo / sensacionalista).
El umbral por defecto es 0.5.
"""
import json
import re

CLICKBAIT_PATTERNS = [
    r"no creeras",
    r"no te creeras",
    r"escandalo",
    r"te sorprendera",
    r"te sorprendera saber",
    r"exclusivo",
    r"impactante",
    r"nadie esperaba",
    r"secreto mejor guardado",
    r"urgente",
    r"increible",
    r"no vas a creer",
    r"lo que nadie te dijo",
    r"la verdad oculta",
    r"filtran",
    r"explota",
    r"bomba",
]

PALABRAS_EMOCIONALES = [
    "conmocionado",
    "escandalizado",
    "estremecedor",
    "impactante",
    "increible",
    "bomba",
    "escandalo",
    "catastrofico",
    "historico",
    "sin precedentes",
    "explosivo",
    "alarmante",
    "devastador",
    "monumental",
    "escalofriante",
    "viralo",
    "viral",
]

_EXCLAM_RE = re.compile(r"[!¡]+")
_MAYUS_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{2,}\b")
_INTERROG_RE = re.compile(r"[?¿]+")


def _normalizar_para_match(texto: str) -> str:
    """Quita tildes y pasa a minusculas para matching robusto."""
    try:
        from unidecode import unidecode
        return unidecode(texto.lower())
    except Exception:
        # Fallback manual de tildes comunes en espanol.
        return texto.lower().translate(str.maketrans("áéíóúñü", "aeiounu"))


def analizar_sensacionalismo(texto: str) -> dict:
    """Analiza el tono sensacionalista de un texto.

    Devuelve:
      {
        "score": float,        # 0.0 - 1.0
        "label": str,          # "sensacionalista" o "informativo"
        "signals": dict,       # desglose de senales detectadas
      }
    """
    if not isinstance(texto, str) or not texto.strip():
        return {"score": 0.0, "label": "informativo", "signals": {}}

    texto_norm = _normalizar_para_match(texto)
    palabras = texto_norm.split()
    n_palabras = max(len(palabras), 1)

    # --- Senal 1: patron clickbait al inicio (peso alto). ---
    inicio_norm = texto_norm[:80]
    clickbait_hits = [p for p in CLICKBAIT_PATTERNS if re.search(p, inicio_norm)]
    clickbait_score = min(len(clickbait_hits) * 0.35, 0.6)

    # --- Senal 2: palabras emocionales (peso medio). ---
    emocional_hits = [p for p in PALABRAS_EMOCIONALES if p in texto_norm]
    emocional_score = min(len(emocional_hits) * 0.15, 0.4)

    # --- Senal 3: exclamaciones (peso bajo-medio). ---
    n_exclam = len(_EXCLAM_RE.findall(texto))
    exclam_score = min(n_exclam * 0.1, 0.3)

    # --- Senal 4: proporcion de palabras en MAYUSCULAS (peso medio). ---
    mayus_palabras = _MAYUS_RE.findall(texto)
    prop_mayus = len(mayus_palabras) / n_palabras
    mayus_score = min(prop_mayus * 1.5, 0.3)

    # --- Senal 5: interrogaciones sensacionalistas (peso bajo). ---
    n_interrog = len(_INTERROG_RE.findall(texto))
    interrog_score = min(n_interrog * 0.05, 0.15)

    # Score final combinado (capado en 1.0).
    score = min(clickbait_score + emocional_score + exclam_score + mayus_score + interrog_score, 1.0)
    label = "sensacionalista" if score >= 0.5 else "informativo"

    return {
        "score": round(score, 4),
        "label": label,
        "signals": {
            "clickbait_hits": clickbait_hits,
            "emocional_hits": emocional_hits,
            "n_exclamaciones": n_exclam,
            "prop_mayusculas": round(prop_mayus, 4),
            "n_interrogaciones": n_interrog,
        },
    }


def exportar_reglas(ruta: str):
    """Exporta los patrones y palabras a JSON para usar en JS."""
    data = {
        "clickbait_patterns": CLICKBAIT_PATTERNS,
        "palabras_emocionales": PALABRAS_EMOCIONALES,
        "umbral": 0.5,
        "pesos": {
            "clickbait": 0.35,
            "emocional": 0.15,
            "exclamacion": 0.1,
            "mayusculas": 1.5,  # multiplicador de la proporcion
            "interrogacion": 0.05,
        },
        "caps": {
            "clickbait": 0.6,
            "emocional": 0.4,
            "exclamacion": 0.3,
            "mayusculas": 0.3,
            "interrogacion": 0.15,
        },
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Reglas de sensacionalismo exportadas: {ruta}")


if __name__ == "__main__":
    ej1 = "NO CREERAS lo que paso. El gobierno anuncio medidas. !"
    ej2 = "El banco central reviso las tasas de interes. La informacion fue confirmada."
    print("Sensacionalista:", analizar_sensacionalismo(ej1))
    print("Informativo:", analizar_sensacionalismo(ej2))
