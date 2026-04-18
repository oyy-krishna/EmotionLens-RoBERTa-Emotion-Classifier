# EmotionLens — RoBERTa Emotion Detection Web App
### SemEval-2018 Task 1, Subtask E-c · Bennett University

**Authors:** Krishan (E23CSEU1282) · Chirag Anand (E23CSEU1276) · Niyati Jain (E23CSEU1284)

---

## 🌐 Live Demo

> Try it instantly — no setup needed

**Web App →** [https://emotion-lens-roberta.vercel.app/](https://emotion-lens-roberta.vercel.app/)

**API Docs →** [https://oyykrishna-emotionlens-api.hf.space/docs](https://oyykrishna-emotionlens-api.hf.space/docs)

---

## 📋 What this project does

EmotionLens detects multiple emotions simultaneously from short social media text using a fine-tuned RoBERTa-base model. Unlike simple positive/negative sentiment analysis, it classifies text across **11 emotion labels** at once:

`anger` · `anticipation` · `disgust` · `fear` · `joy` · `love` · `optimism` · `pessimism` · `sadness` · `surprise` · `trust`

---

## 🚀 Run locally — Quick Start

You have two options depending on whether you want to use our pre-trained model or bring your own.

### Option A — Use our pre-trained model (recommended)

Our trained model is hosted on Hugging Face and downloads automatically on first run. You don't need to download anything manually.

**Step 1 — Clone the repo**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/emotion-app.git
cd emotion-app
```

**Step 2 — Start the backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

On first run, this will automatically download our model (~400 MB) from:
[https://huggingface.co/oyykrishna/roberta-semeval2018-emotions](https://huggingface.co/oyykrishna/roberta-semeval2018-emotions)

The model is cached locally after the first download — subsequent startups are instant.

**Step 3 — Start the frontend** (open a second terminal)
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** — the app is live.

---

### Option B — Use your own trained model

If you have your own `.pt` model file fine-tuned on SemEval-2018, you can plug it straight into the UI.

**Step 1 — Clone and install (same as above)**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/emotion-app.git
cd emotion-app
cd backend && pip install -r requirements.txt
```

**Step 2 — Edit `backend/main.py`**

Find these two lines near the top and update them:

```python
# Option B-1: Load from your own Hugging Face repo
HF_MODEL_REPO     = "your-hf-username/your-model-repo"
HF_MODEL_FILENAME = "your_model_file.pt"
```

Or if your model file is stored locally on your machine, replace the entire `hf_hub_download` block with:

```python
# Option B-2: Load from a local file path
model_path = "/absolute/path/to/your/model.pt"
```

**Step 3 — Update thresholds (optional but recommended)**

If your model has different per-label thresholds, update `ADAPTIVE_THRESHOLDS` in `backend/main.py`:

```python
ADAPTIVE_THRESHOLDS = {
    "anger":        0.45,   # ← replace with your values
    "anticipation": 0.35,
    "disgust":      0.40,
    "fear":         0.38,
    "joy":          0.48,
    "love":         0.32,
    "optimism":     0.42,
    "pessimism":    0.36,
    "sadness":      0.45,
    "surprise":     0.30,
    "trust":        0.28,
}
```

Your thresholds were printed at the end of **Stage 7** in the Colab notebook and saved in:
`/content/outputs/final_results.json` → `adaptive_thresholds`

**Step 4 — Start the app**
```bash
# Terminal 1
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2
cd frontend && npm install && npm run dev
```

---

## 🔧 Requirements

| Tool | Minimum version |
|------|----------------|
| Python | 3.9+ |
| Node.js | 18+ |
| pip | any recent |
| RAM | 4 GB (8 GB recommended) |
| Disk | ~2 GB free (for model cache) |

> **No GPU required.** The model runs on CPU by default. Inference takes ~1–2 seconds per tweet on CPU.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and label list |
| GET | `/health` | Health check + device info |
| GET | `/thresholds` | View current adaptive thresholds |
| POST | `/predict` | Classify a single tweet |
| POST | `/predict/batch` | Classify up to 50 tweets at once |
| GET | `/docs` | Interactive Swagger UI |

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!! 🎉", "threshold_mode": "adaptive"}'
```

### Example response

```json
{
  "original_text": "I am so happy today!! 🎉",
  "cleaned_text": "i am so happy today! smiling face with open mouth and smiling eyes",
  "detected_emotions": ["joy", "optimism"],
  "dominant_emotion": "joy",
  "confidence_score": 0.9341,
  "emotions": [
    { "label": "joy",      "probability": 0.9341, "detected": true  },
    { "label": "optimism", "probability": 0.7823, "detected": true  },
    { "label": "sadness",  "probability": 0.0412, "detected": false },
    ...
  ]
}
```

---

## ✨ Features

- **Single tweet analysis** — per-label probability bar chart + individual emotion cards
- **Batch analysis** — paste up to 50 tweets (one per line), download results as CSV
- **Adaptive thresholds** — per-emotion tuned τ for better detection of rare emotions like surprise and trust
- **Full preprocessing pipeline** — emoji → text, hashtag segmentation, mention masking
- **Preprocessed input visible** — shows what the model actually sees alongside the original tweet
- **Analysis history** — last 8 analysed tweets shown in the sidebar

---

## 📁 Project Structure

```
emotion-app/
├── backend/
│   ├── main.py            ← FastAPI server (model loading, inference, API routes)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── EmotionCard.jsx
│   │       ├── ResultsChart.jsx
│   │       └── BatchAnalyzer.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── test_model.py      ← standalone model test (no server needed)
├── Dockerfile             ← used by Hugging Face Spaces deployment
└── README.md
```

---

## 📊 Model Performance (Test Set)

| Model | Macro-F1 | Micro-F1 | Jaccard |
|-------|----------|----------|---------|
| SVM (TF-IDF) | 42.1% | — | 38.5% |
| BERT-base | 53.8% | — | — |
| RoBERTa-base (fixed τ = 0.5) | ~56% | ~67% | ~52% |
| **RoBERTa-base + Adaptive τ (ours)** | **58.4%** | **68.9%** | **54.7%** |

Model trained on SemEval-2018 Task 1, Subtask E-c (6,838 train / 3,259 test tweets).

---

## 📦 Pre-trained Model

Our fine-tuned model is publicly available on Hugging Face:

🤗 [oyykrishna/roberta-semeval2018-emotions](https://huggingface.co/oyykrishna/roberta-semeval2018-emotions)

- Architecture: `roberta-base` + linear classification head
- Loss: BCEWithLogitsLoss (class-weighted for imbalance handling)
- Optimizer: AdamW (lr = 2e-5, weight decay = 0.01)
- Trained for 5 epochs with linear warmup scheduler

---

## 🛠️ Troubleshooting

**Model download is slow or fails**
The model is ~400 MB. If download fails, check your internet connection and retry. The download only happens once — it's cached after that.

**Port 8000 already in use**
```bash
uvicorn main:app --reload --port 8001
```
Then update `VITE_API_URL` in `frontend/.env` to `http://localhost:8001`.

**Windows path errors**
Always wrap paths containing spaces in quotes:
```bash
cd "C:\Users\Your Name\Documents\emotion-app"
```

**`ModuleNotFoundError` on startup**
Make sure you're installing dependencies inside the `backend/` folder:
```bash
cd backend
pip install -r requirements.txt
```

---

## 📄 Citation

If you use this project in your research, please cite:

```
Krishan, Chirag Anand, Niyati Jain. "Predicting Multiple Human Emotions
from Short Text or Social Media Posts." Bennett University, 2024.
Dataset: SemEval-2018 Task 1, Subtask E-c.
```
