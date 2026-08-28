"""Tests de sentiment.py (enfoque lexico, sin necesidad de sklearn)."""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import sentiment


class SentimientoLexicoTest(unittest.TestCase):
    def test_positivo(self):
        r = sentiment.sentimiento_lexico("El gobierno celebro el crecimiento economico, muy positivo.")
        self.assertEqual(r["label"], "positivo")
        self.assertGreater(r["polaridad"], 0)
        self.assertGreaterEqual(r["score"], 0)

    def test_negativo(self):
        r = sentiment.sentimiento_lexico("Criticos cuestionaron la falta de planificacion, graves consecuencias.")
        self.assertEqual(r["label"], "negativo")
        self.assertLess(r["polaridad"], 0)

    def test_neutro(self):
        r = sentiment.sentimiento_lexico("El banco central reviso las tasas de interes.")
        self.assertEqual(r["label"], "neutro")
        self.assertEqual(r["polaridad"], 0.0)
        self.assertEqual(r["score"], 0.0)

    def test_negacion_invierte_polaridad(self):
        positivo = sentiment.sentimiento_lexico("bueno")["polaridad"]
        negado = sentiment.sentimiento_lexico("no es bueno")["polaridad"]
        self.assertGreater(positivo, 0)
        self.assertLess(negado, 0)
        # La magnitud se conserva, solo cambia el signo.
        self.assertAlmostEqual(abs(negado), positivo, places=4)

    def test_negacion_fuera_de_ventana(self):
        # Mas de 3 palabras de distancia: la negacion no debe aplicar.
        r = sentiment.sentimiento_lexico("no de la de la bueno")
        self.assertEqual(r["label"], "positivo")

    def test_intensificador_multiplica(self):
        base = sentiment.sentimiento_lexico("excelente resultado")["polaridad"]
        intens = sentiment.sentimiento_lexico("muy excelente resultado")["polaridad"]
        self.assertAlmostEqual(intens, base * 1.5, places=4)

    def test_texto_vacio_o_no_string(self):
        for entrada in ("", "   ", None, 42):
            r = sentiment.sentimiento_lexico(entrada)
            self.assertEqual(r["label"], "neutro")
            self.assertEqual(r["score"], 0.0)

    def test_tildes_normalizadas(self):
        # "histórico" y "historico" deben dar la misma polaridad.
        con_tilde = sentiment.sentimiento_lexico("historico")
        self.assertIn("historico", sentiment.LEX)
        self.assertEqual(sentiment.sentimiento_lexico("HISTÓRICO")["polaridad"], con_tilde["polaridad"])


class ExportarLexicoTest(unittest.TestCase):
    def test_exporta_json_valido(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "lexicon.json"
            sentiment.exportar_lexico(ruta)
            data = json.loads(ruta.read_text(encoding="utf-8"))
        self.assertIn("lexicon", data)
        self.assertIn("negaciones", data)
        self.assertIn("intensificadores", data)
        self.assertEqual(set(data["negaciones"]), set(sentiment.NEGACIONES))
        self.assertEqual(data["lexicon"], sentiment.LEX)


if __name__ == "__main__":
    unittest.main()