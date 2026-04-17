"""
FastAPI backend — RoBERTa Multi-Label Emotion Classifier
SemEval-2018 Task 1, Subtask E-c
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import re
import os
import json

# ── Try to import demoji / wordninja gracefully ────────────────────
try:
    import demoji
    demoji.download_codes()
    DEMOJI_AVAILABLE = True
except Exception:
    DEMOJI_AVAILABLE = False

try:
    import wordninja
    WORDNINJA_AVAILABLE = True
except Exception:
    WORDNINJA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/best_roberta_model.pt")
MAX_LEN    = 128

EMOTION_LABELS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust"
]

# Adaptive thresholds tuned on SemEval-2018 dev set
# Replace these with your actual values from the notebook output
# (Stage 7 → final_results.json → adaptive_thresholds)
ADAPTIVE_THRESHOLDS = {
    "anger":        0.53,
    "anticipation": 0.49,
    "disgust":      0.42,
    "fear":         0.93,
    "joy":          0.33,
    "love":         0.89,
    "optimism":     0.66,
    "pessimism":    0.73,
    "sadness":      0.62,
    "surprise":     0.82,
    "trust":        0.58,
}

# Emotion metadata for UI display
EMOTION_META = {
    "anger":        {"emoji": "😠", "color": "#E24B4A", "description": "Feeling angry or irritated"},
    "anticipation": {"emoji": "🤩", "color": "#BA7517", "description": "Looking forward to something"},
    "disgust":      {"emoji": "🤢", "color": "#3B6D11", "description": "Strong dislike or revulsion"},
    "fear":         {"emoji": "😨", "color": "#534AB7", "description": "Feeling scared or anxious"},
    "joy":          {"emoji": "😊", "color": "#1D9E75", "description": "Feeling happy or elated"},
    "love":         {"emoji": "❤️",  "color": "#D4537E", "description": "Feeling affection or deep care"},
    "optimism":     {"emoji": "🌟", "color": "#185FA5", "description": "Hopeful about the future"},
    "pessimism":    {"emoji": "😔", "color": "#5F5E5A", "description": "Expecting the worst outcome"},
    "sadness":      {"emoji": "😢", "color": "#378ADD", "description": "Feeling sad or sorrowful"},
    "surprise":     {"emoji": "😲", "color": "#993C1D", "description": "Feeling unexpected shock"},
    "trust":        {"emoji": "🤝", "color": "#0F6E56", "description": "Feeling safe or confident"},
}

# ─────────────────────────────────────────────────────────────────────
# Model definition (must match training architecture exactly)
# ─────────────────────────────────────────────────────────────────────
class RoBERTaEmotionClassifier(nn.Module):
    def __init__(self, num_labels=11, dropout_p=0.1):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained("roberta-base")
        self.dropout    = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


# ─────────────────────────────────────────────────────────────────────
# Preprocessing (mirrors the Colab notebook Stage 3)
# ─────────────────────────────────────────────────────────────────────
def segment_hashtag(match):
    tag = match.group(1)
    if WORDNINJA_AVAILABLE:
        return " ".join(wordninja.split(tag)).lower()
    return tag.lower()


def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if DEMOJI_AVAILABLE:
        text = demoji.replace_with_desc(text, sep=" ")
        text = re.sub(r":", " ", text)
        text = re.sub(r"_", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", segment_hashtag, text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"([!?.])\\1+", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ─────────────────────────────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] Device: {DEVICE}")

print("[startup] Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

print("[startup] Loading model weights...")
model = RoBERTaEmotionClassifier(num_labels=len(EMOTION_LABELS))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print(f"[startup] Model ready on {DEVICE}")


# ─────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion Detection API",
    description="Multi-label emotion classification using RoBERTa fine-tuned on SemEval-2018",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    threshold_mode: str = "adaptive"   # "adaptive" | "fixed"
    fixed_threshold: float = 0.5


class EmotionResult(BaseModel):
    label: str
    probability: float
    detected: bool
    emoji: str
    color: str
    description: str


class PredictResponse(BaseModel):
    original_text: str
    cleaned_text: str
    emotions: List[EmotionResult]
    detected_emotions: List[str]
    dominant_emotion: str | None
    confidence_score: float


class BatchRequest(BaseModel):
    texts: List[str]
    threshold_mode: str = "adaptive"


# ─────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────
def run_inference(text: str, threshold_mode: str = "adaptive", fixed_threshold: float = 0.5) -> dict:
    cleaned = clean_tweet(text)
    if not cleaned:
        raise ValueError("Text is empty after preprocessing")

    enc = tokenizer(
        cleaned,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    emotions = []
    for i, label in enumerate(EMOTION_LABELS):
        if threshold_mode == "adaptive":
            threshold = ADAPTIVE_THRESHOLDS[label]
        else:
            threshold = fixed_threshold

        detected = bool(probs[i] > threshold)
        meta     = EMOTION_META[label]
        emotions.append({
            "label":       label,
            "probability": round(float(probs[i]), 4),
            "detected":    detected,
            "emoji":       meta["emoji"],
            "color":       meta["color"],
            "description": meta["description"],
        })

    emotions.sort(key=lambda x: x["probability"], reverse=True)
    detected_emotions = [e["label"] for e in emotions if e["detected"]]
    dominant = emotions[0]["label"] if emotions else None
    confidence = float(max(probs)) if len(probs) > 0 else 0.0

    return {
        "original_text":    text,
        "cleaned_text":     cleaned,
        "emotions":         emotions,
        "detected_emotions": detected_emotions,
        "dominant_emotion": dominant,
        "confidence_score": round(confidence, 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Emotion Detection API",
        "model": "RoBERTa-base (SemEval-2018)",
        "labels": EMOTION_LABELS,
        "endpoints": ["/predict", "/predict/batch", "/thresholds", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": True}


@app.get("/thresholds")
def get_thresholds():
    return {"adaptive_thresholds": ADAPTIVE_THRESHOLDS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")
    try:
        result = run_inference(req.text, req.threshold_mode, req.fixed_threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")
    if len(req.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch")
    results = []
    for text in req.texts:
        try:
            result = run_inference(text, req.threshold_mode)
            results.append({"text": text, "result": result, "error": None})
        except Exception as e:
            results.append({"text": text, "result": None, "error": str(e)})
    return {"results": results, "total": len(results)}
