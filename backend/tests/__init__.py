"""Tests del pipeline backend (solo stdlib; los que necesitan sklearn se saltan).

Ejecucion:
  python3 -m unittest discover -s backend/tests -t backend -v
  # o, si hay pytest disponible:
  pytest backend/tests -q
"""
import sys
from pathlib import Path

# Asegurar que backend/ este en el path para importar los paquetes del pipeline.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))