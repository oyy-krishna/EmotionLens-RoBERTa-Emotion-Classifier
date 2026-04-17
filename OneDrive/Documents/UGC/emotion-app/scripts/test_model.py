"""
test_model.py — Standalone model evaluation script
Run this first to verify your trained model is working correctly.

Usage:
    python scripts/test_model.py

Requires:
    - model/best_roberta_model.pt
    - pip install transformers torch demoji wordninja scikit-learn
"""

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import re
import os
import sys
import json
from pathlib import Path

try:
    import demoji
    demoji.download_codes()
    DEMOJI_AVAILABLE = True
except ImportError:
    DEMOJI_AVAILABLE = False
    print("⚠️  demoji not installed — emoji conversion disabled")

try:
    import wordninja
    WORDNINJA_AVAILABLE = True
except ImportError:
    WORDNINJA_AVAILABLE = False
    print("⚠️  wordninja not installed — hashtag segmentation disabled")

# ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "model" / "best_roberta_model.pt"

EMOTION_LABELS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust"
]

ADAPTIVE_THRESHOLDS = {
    "anger": 0.45, "anticipation": 0.35, "disgust": 0.40,
    "fear": 0.38, "joy": 0.48, "love": 0.32,
    "optimism": 0.42, "pessimism": 0.36, "sadness": 0.45,
    "surprise": 0.30, "trust": 0.28,
}


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
        return self.classifier(self.dropout(cls_output))


def segment_hashtag(match):
    tag = match.group(1)
    if WORDNINJA_AVAILABLE:
        return " ".join(wordninja.split(tag)).lower()
    return tag.lower()


def clean_tweet(text):
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
    return re.sub(r"\s+", " ", text).strip().lower()


def load_model():
    if not MODEL_PATH.exists():
        print(f"❌  Model file not found: {MODEL_PATH}")
        print(f"    Make sure best_roberta_model.pt is in the model/ folder.")
        sys.exit(1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅  Device     : {DEVICE}")

    print("    Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("    Loading model weights...")
    model = RoBERTaEmotionClassifier(num_labels=len(EMOTION_LABELS))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"✅  Model ready ({sum(p.numel() for p in model.parameters()):,} params)\n")
    return model, tokenizer, DEVICE


def predict(text, model, tokenizer, device, verbose=True):
    cleaned = clean_tweet(text)
    enc = tokenizer(
        cleaned, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    results = {}
    detected = []
    for i, label in enumerate(EMOTION_LABELS):
        p = float(probs[i])
        results[label] = p
        if p > ADAPTIVE_THRESHOLDS[label]:
            detected.append((label, p))

    detected.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"  Input   : {text}")
        print(f"  Cleaned : {cleaned}")
        print(f"  Detected: {[f'{l} ({p:.2f})' for l, p in detected] or ['none']}")
        top = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top-5 P : {[(l, f'{p:.3f}') for l, p in top]}")

    return results, detected


# ─────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "text": "I can't believe I got the job! Best day of my life 🎉🎉",
        "expected": ["joy", "optimism"],
    },
    {
        "text": "So exhausted and sad... nothing ever gets better 😭",
        "expected": ["sadness", "pessimism"],
    },
    {
        "text": "This makes me SO angry!! How dare they do this #outraged 😡",
        "expected": ["anger", "disgust"],
    },
    {
        "text": "I'm scared about tomorrow's results, not sure what to expect",
        "expected": ["fear", "anticipation"],
    },
    {
        "text": "Feeling so loved and grateful for everyone around me ❤️",
        "expected": ["love", "joy"],
    },
    {
        "text": "Wow I did NOT see that coming... completely shocked 😲",
        "expected": ["surprise"],
    },
    {
        "text": "I trust you completely, you've never let me down 🤝",
        "expected": ["trust"],
    },
    {
        "text": "@user check out this link http://t.co/xyz #SoHappy today!!!!!!",
        "expected": ["joy"],
        "note": "Tests preprocessing (URL, mention, hashtag, punctuation)"
    },
    {
        "text": "Mixed feelings today — happy about the news but also worried 😕",
        "expected": ["joy", "fear"],
        "note": "Multi-label case"
    },
    {
        "text": "The weather is nice today.",
        "expected": [],
        "note": "Low-emotion neutral sentence"
    },
]


def run_tests(model, tokenizer, device):
    print("=" * 65)
    print("RUNNING TEST SUITE")
    print("=" * 65)

    passed = 0
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}]{' — ' + tc.get('note','')}")
        results, detected = predict(tc["text"], model, tokenizer, device, verbose=True)

        detected_labels = [l for l, _ in detected]
        expected        = tc["expected"]

        # Soft pass: at least half of expected labels are detected
        if not expected:
            passed += 1
            print(f"  Result  : ✅  (no expected emotion — model says: {detected_labels or 'none'})")
        else:
            hits = sum(1 for e in expected if e in detected_labels)
            if hits >= max(1, len(expected) // 2):
                passed += 1
                print(f"  Result  : ✅  (expected {expected}, got {detected_labels})")
            else:
                print(f"  Result  : ⚠️  (expected {expected}, got {detected_labels})")

    print(f"\n{'='*65}")
    print(f"TEST RESULTS: {passed}/{len(TEST_CASES)} passed")
    print(f"{'='*65}\n")
    return passed


def run_speed_test(model, tokenizer, device, n=50):
    import time
    print(f"Speed test ({n} inferences)...")
    texts = ["This is a test tweet with some emotions 😊 #happy @user"] * n
    start = time.time()
    for t in texts:
        predict(t, model, tokenizer, device, verbose=False)
    elapsed = time.time() - start
    print(f"  Total   : {elapsed:.2f}s")
    print(f"  Per item: {elapsed/n*1000:.1f}ms")
    print(f"  Throughput: {n/elapsed:.1f} tweets/sec\n")


def interactive_mode(model, tokenizer, device):
    print("=" * 65)
    print("INTERACTIVE MODE — type a tweet, press Enter (q to quit)")
    print("=" * 65)
    while True:
        text = input("\nTweet > ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue
        predict(text, model, tokenizer, device, verbose=True)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 RoBERTa Emotion Classifier — Model Test")
    print("   SemEval-2018 Task 1 Subtask E-c\n")

    model, tokenizer, device = load_model()

    # Run tests
    run_tests(model, tokenizer, device)

    # Speed test
    run_speed_test(model, tokenizer, device)

    # Optional interactive mode
    ans = input("Run interactive mode? (y/n): ").strip().lower()
    if ans == "y":
        interactive_mode(model, tokenizer, device)
