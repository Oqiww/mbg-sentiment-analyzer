"""
inference.py
─────────────────────────────────────────────────────────────────────────────
Model loading and sentiment inference for MBG Sentiment Analyzer.

Model: IndoBERT-base-p1 fine-tuned as BertForSequenceClassification (3 classes)
- config_mbg.json
- model_mbg.safetensors
- tokenizer_mbg.json / tokenizer_config_mbg.json

Label mapping (from config_mbg.json id2label):
  0 -> Positive
  1 -> Neutral
  2 -> Negative

Tokenizer settings:
  - BertTokenizer (do_lower_case: False)
  - max_length: 256 (MAX_LEN_INDOBERT in notebook)
  - padding: max_length
  - truncation: True
  - return_tensors: pt
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from src.preprocessing import clean_stream_b

logger = logging.getLogger(__name__)

# Label mapping as per config_mbg.json id2label
LABEL_MAP: Dict[int, str] = {0: "Positive", 1: "Neutral", 2: "Negative"}

# Max sequence length for IndoBERT-base-p1 (Stream B)
MAX_LEN_INDOBERT = 256

# Minimum text length (after stripping) to consider valid
MIN_TEXT_LENGTH = 1

# Maximum character length to accept (prevent abuse / extremely slow inference)
MAX_INPUT_CHARS = 1000


def load_slang_dict(slang_path: str) -> Dict[str, str]:
    """Load the curated slang dictionary from JSON file."""
    with open(slang_path, encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(tokenizer_dir: str) -> AutoTokenizer:
    """
    Load tokenizer from local directory using AutoTokenizer.
    Files needed: tokenizer.json (fast tokenizer), tokenizer_config.json
    AutoTokenizer returns BertTokenizer for IndoBERT-base-p1.
    """
    logger.info(f"Loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    return tokenizer


def load_model(model_dir: str, device: torch.device) -> BertForSequenceClassification:
    """
    Load BertForSequenceClassification from local directory.
    Files needed: config_mbg.json, model_mbg.safetensors
    """
    logger.info(f"Loading model from: {model_dir}")
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on device: {device}")
    return model


def tokenize_text(text: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenize a single text using the same strategy as during training:
      - truncation=True
      - max_length=256
      - padding="max_length"
      - return_tensors="pt"
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN_INDOBERT,
        padding="max_length",
        return_tensors="pt",
    )
    return encoding


def predict_sentiment(
    raw_text: str,
    tokenizer: AutoTokenizer,
    model: BertForSequenceClassification,
    slang_dict: Dict[str, str],
    device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run full inference pipeline on a raw input text.

    Returns:
        label       : Predicted sentiment label ("Positive", "Neutral", "Negative")
        confidence  : Confidence score for the predicted class (0.0 - 1.0)
        probs_dict  : Dict mapping each label to its probability
    """
    # --- Preprocessing (Stream B, exactly as training) ---
    processed_text = clean_stream_b(raw_text, slang_dict)

    # --- Tokenization ---
    encoding = tokenize_text(processed_text, tokenizer)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # --- Model Inference ---
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # --- Label Mapping ---
    pred_idx = int(torch.argmax(torch.tensor(probs)).item())
    label = LABEL_MAP[pred_idx]
    confidence = probs[pred_idx]

    probs_dict = {
        "Positive": probs[0],
        "Neutral": probs[1],
        "Negative": probs[2],
    }

    return label, confidence, probs_dict


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validate user input text.

    Returns:
        (is_valid, error_message)
        is_valid=True means the text is safe to process.
    """
    if not isinstance(text, str):
        return False, "Input harus berupa teks."

    stripped = text.strip()

    if len(stripped) == 0:
        return False, "Silakan masukkan komentar terlebih dahulu."

    if len(stripped) > MAX_INPUT_CHARS:
        return (
            False,
            f"Komentar terlalu panjang. Maksimum {MAX_INPUT_CHARS} karakter "
            f"(saat ini: {len(stripped)} karakter).",
        )

    return True, ""
