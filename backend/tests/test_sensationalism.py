"""Tests de sensationalism.py (reglas clickbait, solo stdlib)."""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import sensationalism


class AnalizarSensacionalismoTest(unittest.TestCase):
    def test_clickbait_claro(self):
        texto = "¡ÚLTIMA HORA! Escándalo increíble: nadie esperaba esto, bomba informativa!!!"
        r = sensationalism.analizar_sensacionalismo(texto)
        self.assertEqual(r["label"], "sensacionalista")
        self.assertGreaterEqual(r["score"], 0.5)
        self.assertTrue(r["signals"]["clickbait_hits"])
        self.assertGreater(r["signals"]["n_exclamaciones"], 0)

    def test_informativo(self):
        r = sensationalism.analizar_sensacionalismo(
            "El banco central reviso las tasas de interes. La informacion fue confirmada."
        )
        self.assertEqual(r["label"], "informativo")
        self.assertLess(r["score"], 0.5)

    def test_vacio_o_no_string(self):
        for entrada in ("", "   ", None, 42):
            r = sensationalism.analizar_sensacionalismo(entrada)
            self.assertEqual(r["label"], "informativo")
            self.assertEqual(r["score"], 0.0)
            self.assertEqual(r["signals"], {})

    def test_score_acotado(self):
        texto = ("EXCLUSIVO URGENTE escandalo bomba increible impactante!!! " * 10)
        r = sensationalism.analizar_sensacionalismo(texto)
        self.assertLessEqual(r["score"], 1.0)

    def test_signals_desglosadas(self):
        r = sensationalism.analizar_sensacionalismo("El GOBIERNO anuncio MEDIDAS nuevas hoy")
        for clave in (
            "clickbait_hits",
            "emocional_hits",
            "n_exclamaciones",
            "prop_mayusculas",
            "n_interrogaciones",
        ):
            self.assertIn(clave, r["signals"])
        self.assertGreater(r["signals"]["prop_mayusculas"], 0)

    def test_interrogacion_suma(self):
        con = sensationalism.analizar_sensacionalismo("Sabias que esto paso??")
        self.assertGreater(con["signals"]["n_interrogaciones"], 0)
        self.assertGreater(con["score"], 0)


class ZonaClickbaitTest(unittest.TestCase):
    def test_patron_al_inicio_cuenta_mas_que_al_final(self):
        inicio = "Escándalo: " + "texto neutro sin senales " * 30
        final = "texto neutro sin senales " * 30 + " escandalo"
        score_inicio = sensationalism.analizar_sensacionalismo(inicio)["score"]
        score_final = sensationalism.analizar_sensacionalismo(final)["score"]
        self.assertGreater(score_inicio, score_final)


class ExportarReglasTest(unittest.TestCase):
    def test_exporta_json_valido(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "rules.json"
            sensationalism.exportar_reglas(ruta)
            data = json.loads(ruta.read_text(encoding="utf-8"))
        self.assertEqual(data["clickbait_patterns"], sensationalism.CLICKBAIT_PATTERNS)
        self.assertEqual(data["palabras_emocionales"], sensationalism.PALABRAS_EMOCIONALES)
        self.assertEqual(data["umbral"], 0.5)
        self.assertEqual(data["ventana_clickbait"], sensationalism.VENTANA_CLICKBAIT)
        self.assertIn("caps", data)


if __name__ == "__main__":
    unittest.main()