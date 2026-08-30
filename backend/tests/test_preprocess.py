"""Tests de preprocess.py (Semana 1: normalizacion y tokenizacion).

Las aserciones se escriben para pasar tanto con NLTK/spaCy instalados como con
sus fallbacks, asi la suite corre en cualquier entorno (incluido uno limpio).
"""
import unittest

from pipeline import preprocess


class NormalizarTextoTest(unittest.TestCase):
    def test_minusculas_y_espacios(self):
        self.assertEqual(
            preprocess.normalizar_texto("  HOLA   Mundo  "),
            "hola mundo",
        )

    def test_url_reemplazada(self):
        norm = preprocess.normalizar_texto("Mira esto: https://ejemplo.com/a?b=1 fin")
        self.assertIn("url", norm)
        self.assertNotIn("https", norm)

    def test_mencion_reemplazada(self):
        norm = preprocess.normalizar_texto("segui a @usuario por la noticia")
        self.assertIn("user", norm)
        self.assertNotIn("@usuario", norm)

    def test_hashtag_conserva_palabra(self):
        norm = preprocess.normalizar_texto("gran #deporte hoy")
        self.assertIn("deporte", norm)
        self.assertNotIn("#", norm)

    def test_tildes_eliminadas(self):
        # Con o sin unidecode, el fallback manual quita las tildes comunes.
        self.assertEqual(preprocess.normalizar_texto("habrá información"), "habra informacion")

    def test_no_string_devuelve_vacio(self):
        self.assertEqual(preprocess.normalizar_texto(None), "")
        self.assertEqual(preprocess.normalizar_texto(123), "")


class TokenizarTest(unittest.TestCase):
    def test_filtra_stopwords_y_cortas(self):
        toks = preprocess.tokenizar(preprocess.normalizar_texto("El gobierno de la ciudad"))
        self.assertNotIn("el", toks)
        self.assertNotIn("de", toks)
        self.assertNotIn("la", toks)
        self.assertIn("gobierno", toks)

    def test_conserva_negaciones(self):
        toks = preprocess.tokenizar(preprocess.normalizar_texto("no hay ninguna novedad"))
        self.assertIn("no", toks)
        self.assertIn("nunca", preprocess.tokenizar(preprocess.normalizar_texto("nunca paso nada")))

    def test_texto_vacio(self):
        self.assertEqual(preprocess.tokenizar(""), [])

    def test_reduce_repeticiones(self):
        # TweetTokenizer (reduce_len=True) colapsa repeticiones; sin NLTK el
        # fallback las conserva, asi solo se exige la version con NLTK.
        toks = preprocess.tokenizar("graciaaas")
        if preprocess.TweetTokenizer is None:
            self.assertIn("graciaaas", toks)
        else:
            self.assertTrue(all("aaaa" not in t for t in toks))


class PreprocesoTest(unittest.TestCase):
    def test_preproceso_texto_devuelve_string(self):
        out = preprocess.preproceso_texto("El @presidente anuncio en www.gob.bo que NO habrá medidas!")
        self.assertIsInstance(out, str)
        self.assertNotIn("presidente", out)  # la mencion se reemplaza por user
        self.assertIn("no", out)  # las negaciones se conservan

    def test_lematizar_sin_tokens(self):
        self.assertEqual(preprocess.lematizar([]), [])


if __name__ == "__main__":
    unittest.main()