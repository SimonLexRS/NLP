"""Exporta el transformer fine-tuneado a ONNX (cuantizado) para transformers.js.

Flujo:
  1. Exporta el modelo PyTorch (models/transformer) a ONNX con optimum.
  2. Cuantiza a int8 (q8) para reducir tamano (~15-20MB).
  3. Copia el resultado a web/assets/model_onnx/ para que lo cargue transformers.js.

Uso:
  python backend/export/export_onnx.py

Requisitos: optimum[onnxruntime], onnxruntime. Se instalan con:
  pip install optimum[onnxruntime] onnxruntime
"""
import sys
import os
import json
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

MODELS_DIR = BACKEND_DIR / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"
ONNX_DIR = MODELS_DIR / "transformer_onnx"
WEB_ONNX_DIR = BACKEND_DIR.parent / "web" / "assets" / "model_onnx"


def exportar_onnx():
    print("=" * 60)
    print("EXPORTANDO TRANSFORMER A ONNX")
    print("=" * 60)

    if not TRANSFORMER_DIR.exists():
        print(f"ERROR: No se encontro el modelo en {TRANSFORMER_DIR}")
        print("Ejecuta primero: python backend/train/train_all.py --epochs 8")
        return False

    # --- Paso 1: Exportar a ONNX ---
    print("\n[1/3] Exportando a ONNX con optimum...")
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"ERROR: Falta instalar optimum/onnxruntime: {e}")
        print("  pip install optimum[onnxruntime] onnxruntime")
        return False

    if ONNX_DIR.exists():
        shutil.rmtree(ONNX_DIR)
    ONNX_DIR.mkdir(parents=True)

    # Exportar modelo.
    model = ORTModelForSequenceClassification.from_pretrained(
        str(TRANSFORMER_DIR), export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
    model.save_pretrained(str(ONNX_DIR))
    tokenizer.save_pretrained(str(ONNX_DIR))
    print(f"  ONNX exportado -> {ONNX_DIR}")

    # --- Paso 2: Cuantizar a int8 ---
    print("\n[2/3] Cuantizando a int8 (q8)...")
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))
        # Cuantizacion dinamica (compatible con transformers.js / WASM).
        # API nueva de optimum: avx2(is_static=False, ...) cuantizacion dinamica.
        qconfig = AutoQuantizationConfig.avx2(is_static=False)
        quantized_dir = MODELS_DIR / "transformer_onnx_q8"
        if quantized_dir.exists():
            shutil.rmtree(quantized_dir)
        quantizer.quantize(qconfig, save_dir=str(quantized_dir))
        # Copiar tokenizer y config al dir cuantizado.
        for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.txt",
                       "special_tokens_map.json", "config.json"]:
            src = ONNX_DIR / fname
            if src.exists() and not (quantized_dir / fname).exists():
                shutil.copy(src, quantized_dir / fname)
        print(f"  Cuantizado -> {quantized_dir}")
        src_dir = quantized_dir
    except Exception as e:
        print(f"  [warn] Cuantizacion fallida ({e}), usando ONNX fp32.")
        src_dir = ONNX_DIR

    # --- Paso 3: Copiar a web/assets/model_onnx/ ---
    # transformers.js v3 espera la estructura de HuggingFace:
    #   model_onnx/
    #     config.json
    #     tokenizer.json, ...
    #     onnx/                  <- subcarpeta obligatoria
    #       model_quantized.onnx
    #       model.onnx  (si existe)
    print("\n[3/3] Copiando a web/assets/model_onnx/...")
    if WEB_ONNX_DIR.exists():
        shutil.rmtree(WEB_ONNX_DIR)
    WEB_ONNX_DIR.mkdir(parents=True)
    onnx_subdir = WEB_ONNX_DIR / "onnx"
    onnx_subdir.mkdir(parents=True)

    # Copiar archivos no-ONNX (config, tokenizer, vocab) a la raiz.
    # Copiar archivos .onnx a la subcarpeta onnx/.
    for item in src_dir.iterdir():
        if item.is_file():
            if item.suffix == ".onnx":
                shutil.copy(item, onnx_subdir / item.name)
            else:
                shutil.copy(item, WEB_ONNX_DIR / item.name)
            size = item.stat().st_size
            destino = "onnx/" + item.name if item.suffix == ".onnx" else item.name
            print(f"  {destino:<40} {size:>10,} bytes")

    # Asegurar que config.json tiene id2label/label2id (para transformers.js).
    config_path = WEB_ONNX_DIR / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # transformers.js espera id2label con claves string.
        from pipeline.transformer import ID2LABEL, LABEL2ID
        cfg["id2label"] = {str(i): l for i, l in ID2LABEL.items()}
        cfg["label2id"] = LABEL2ID
        cfg["architectures"] = ["ElectraForSequenceClassification"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"  config.json actualizado con id2label/label2id")

    # Tamano total.
    total = sum(f.stat().st_size for f in WEB_ONNX_DIR.iterdir() if f.is_file())
    print(f"\n  Tamano total: {total / 1024 / 1024:.1f} MB")
    print(f"  ONNX listo en: {WEB_ONNX_DIR}")
    return True


if __name__ == "__main__":
    ok = exportar_onnx()
    if ok:
        print("\nExportacion ONNX completada.")
    else:
        print("\nExportacion ONNX fallo.")
        sys.exit(1)
