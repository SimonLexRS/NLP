"""Tests del dataset sintetico (splits y etiquetas). Solo stdlib."""
import json
import unittest
from pathlib import Path

from data.generate_dataset import CATEGORIAS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CAMPOS = {"texto", "categoria", "tono", "sentimiento", "id"}
TONOS = {"informativo", "sensacionalista"}
SENTIMIENTOS = {"positivo", "negativo", "neutro"}


def cargar(nombre):
    with open(DATA_DIR / nombre, encoding="utf-8") as f:
        return json.load(f)


class SplitsTest(unittest.TestCase):
    def test_splits_existen_y_no_vacios(self):
        for nombre in ("train.json", "val.json", "test.json"):
            with self.subTest(split=nombre):
                datos = cargar(nombre)
                self.assertGreater(len(datos), 0)

    def test_campos_obligatorios(self):
        for nombre in ("train.json", "val.json", "test.json"):
            for noticia in cargar(nombre)[:50]:
                self.assertTrue(CAMPOS.issubset(noticia.keys()), noticia.keys())
                self.assertIsInstance(noticia["texto"], str)
                self.assertGreater(len(noticia["texto"].strip()), 0)

    def test_categorias_validas_y_cubiertas(self):
        for nombre in ("train.json", "val.json", "test.json"):
            cats = {n["categoria"] for n in cargar(nombre)}
            self.assertTrue(cats.issubset(set(CATEGORIAS)))
            self.assertEqual(cats, set(CATEGORIAS), f"{nombre} no cubre las 7 categorias")

    def test_tonos_y_sentimientos_validos(self):
        for nombre in ("train.json", "val.json", "test.json"):
            datos = cargar(nombre)
            self.assertTrue({n["tono"] for n in datos}.issubset(TONOS), nombre)
            self.assertTrue({n["sentimiento"] for n in datos}.issubset(SENTIMIENTOS), nombre)

    def test_ids_unicos_por_split(self):
        for nombre in ("train.json", "val.json", "test.json"):
            ids = [n["id"] for n in cargar(nombre)]
            self.assertEqual(len(ids), len(set(ids)), f"ids duplicados en {nombre}")

    def test_val_y_test_mismo_tamano(self):
        self.assertEqual(len(cargar("val.json")), len(cargar("test.json")))

    def test_categorias_orden_canonico(self):
        # El orden de CATEGORIAS es el que usan metrics.json y las matrices.
        self.assertEqual(len(CATEGORIAS), 7)
        self.assertEqual(len(set(CATEGORIAS)), 7)


if __name__ == "__main__":
    unittest.main()