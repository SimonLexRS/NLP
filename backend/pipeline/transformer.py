"""Transformer afinado (Semana 4: Transformers).

Fine-tuning de ``mrm8488/electricidad-small-discriminator`` (ELECTRA-small en
espanol) para clasificacion de categoria de noticias (7 clases).

Loop de entrenamiento manual en PyTorch (consistente con Appv1): AdamW,
CrossEntropyLoss, 3-5 epocas. Se guarda el modelo fine-tuneado y luego se
exporta a ONNX en ``export/export_onnx.py``.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "mrm8488/electricidad-small-discriminator"
MAX_LEN = 512
DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 16
DEFAULT_LR = 2e-4

# Etiquetas de categoria (deben coincidir con generate_dataset.CATEGORIAS).
LABELS = [
    "politica",
    "economia",
    "deportes",
    "tecnologia",
    "salud",
    "internacional",
    "cultura",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


class NoticiasDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_len=MAX_LEN):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.textos[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.etiquetas[idx], dtype=torch.long),
        }


def cargar_modelo(num_labels=len(LABELS), model_path=None):
    """Carga el modelo (pre-entrenado o fine-tuneado) + tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME if model_path is None else model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME if model_path is None else model_path,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tok


def entrenar_transformer(
    train_texts,
    train_labels,  # lista de strings (nombres de categoria)
    val_texts=None,
    val_labels=None,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH,
    lr=DEFAULT_LR,
    max_len=MAX_LEN,
    device=None,
    save_dir=None,
):
    """Fine-tunea ELECTRA-small sobre el dataset de noticias.

    train_labels: lista de strings (nombres de categoria) -> se mapean a IDs.
    save_dir: si se pasa, guarda el modelo y tokenizer ahi al final.
    Devuelve el historial de perdida por epoca.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[transformer] device={device} epochs={epochs} batch={batch_size} lr={lr}")
    model, tok = cargar_modelo()
    model.to(device)

    # Mapear etiquetas string -> int.
    train_y = [LABEL2ID[l] for l in train_labels]
    train_ds = NoticiasDataset(train_texts, train_y, tok, max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_dl = None
    if val_texts is not None and val_labels is not None:
        val_y = [LABEL2ID[l] for l in val_labels]
        val_ds = NoticiasDataset(val_texts, val_y, tok, max_len)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            n += len(labels)

        avg_loss = total_loss / max(n, 1)
        msg = f"  Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}"

        # Validacion.
        if val_dl is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_dl:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = out.logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)
            val_acc = correct / max(total, 1)
            msg += f" - val_acc: {val_acc:.4f}"
            history.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": val_acc})
        else:
            history.append({"epoch": epoch + 1, "loss": avg_loss})

        print(msg)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model.to("cpu")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        # Asegurar id2label/label2id en config.
        config_path = os.path.join(save_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["id2label"] = {str(i): l for i, l in ID2LABEL.items()}
            cfg["label2id"] = LABEL2ID
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"[transformer] modelo guardado en {save_dir}")

    return history


def predecir(model, tokenizer, textos, device=None, max_len=MAX_LEN):
    """Predice la categoria de una lista de textos.

    Devuelve lista de dicts: {label, scores: {cat: prob}, confidence}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    resultados = []
    with torch.no_grad():
        for texto in textos:
            enc = tokenizer(
                texto,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
            idx = int(probs.argmax())
            resultados.append(
                {
                    "label": ID2LABEL[idx],
                    "confidence": float(probs[idx]),
                    "scores": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
                }
            )
    return resultados


if __name__ == "__main__":
    # Smoke test minimo (requiere descarga del modelo base).
    print("Modulo transformer listo. Labels:", LABELS)
