# Makefile - Clasificador de Noticias (Caso 9) - Proyecto Final PLN
# Automatiza todo el flujo: setup -> datos -> entrenamiento -> exportacion -> web

PYTHON ?= python3
PIP ?= pip

.PHONY: help setup data train train-classic evaluate export export-js export-onnx web serve deploy clean

help: ## Mostrar ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Instalar dependencias Python
	$(PIP) install -r backend/requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"
	$(PYTHON) -c "import spacy; spacy.cli.download('es_core_news_sm', False, False, '--quiet')" 2>/dev/null || true

data: ## Generar dataset sintetico de noticias
	$(PYTHON) backend/data/generate_dataset.py

train: ## Entrenar TODO el pipeline (clasicos + transformer)
	$(PYTHON) backend/train/train_all.py --epochs 8 --batch 16

train-classic: ## Entrenar solo modelos clasicos (rapido, sin GPU)
	$(PYTHON) backend/train/train_all.py --no-transformer

evaluate: ## Evaluar modelos y generar metricas + matrices de confusion
	$(PYTHON) backend/train/evaluate.py

evaluate-classic: ## Evaluar solo modelos clasicos
	$(PYTHON) backend/train/evaluate.py --no-transformer

export: export-js export-onnx ## Exportar todo (JSON + ONNX) a web/assets/

export-js: ## Exportar pesos de modelos clasicos a JSON
	$(PYTHON) backend/export/export_js.py

export-onnx: ## Exportar transformer a ONNX cuantizado
	$(PYTHON) backend/export/export_onnx.py

web: export ## Re-exportar modelos y preparar web/ para despliegue
	@echo "Sitio listo en web/. Ejecuta 'make serve' para probar localmente."

serve: ## Servir sitio web localmente (http://localhost:8090)
	@echo "Sirviendo web/ en http://localhost:8090 ..."
	$(PYTHON) -m http.server 8090 --directory web

deploy: ## Desplegar a GitHub Pages (requiere git push a main)
	@echo "Para desplegar: git push origin main"
	@echo "El workflow de GitHub Actions publicara web/ automaticamente."
	@echo "URL: https://<usuario>.github.io/<repo>/"

clean: ## Limpiar artefactos generados
	rm -rf backend/models/*.pkl backend/models/*.json backend/models/*.png
	rm -rf backend/models/transformer backend/models/transformer_onnx backend/models/transformer_onnx_q8
	rm -rf web/assets/*.json web/assets/*.png web/assets/model_onnx
	@echo "Artefactos limpiados. Ejecuta 'make data train export' para regenerar."
