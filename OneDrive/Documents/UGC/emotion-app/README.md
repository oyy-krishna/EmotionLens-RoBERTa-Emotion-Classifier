# EmotionLens — RoBERTa Emotion Detection Web App
## SemEval-2018 Task 1, Subtask E-c · Bennett University

---

## Project Structure

```
emotion-app/
├── model/
│   └── best_roberta_model.pt     ← PUT YOUR MODEL FILE HERE
├── backend/
│   ├── main.py                   ← FastAPI server
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
│   └── test_model.py             ← Run this first to verify the model
└── README.md
```

---

## Step 0 — Place your model file

Copy `best_roberta_model.pt` into the `model/` folder:

```
emotion-app/model/best_roberta_model.pt
```

---

## Step 1 — Update adaptive thresholds

Open `backend/main.py` and find `ADAPTIVE_THRESHOLDS` (around line 45).
Replace the default values with your actual thresholds from the Colab notebook.

They were printed at the end of **Stage 7** and saved in:
`/content/outputs/final_results.json` → `adaptive_thresholds`

---

## Step 2 — Test the model (no server needed)

```bash
# From the emotion-app/ root
pip install transformers torch demoji wordninja scikit-learn
python scripts/test_model.py
```

This runs 10 test cases and an interactive mode to verify the model works
before starting the web app.

---

## Step 3 — Run the backend

Open a terminal in Cursor (Ctrl+` or Cmd+`):

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API will be live at: http://localhost:8000
Interactive API docs: http://localhost:8000/docs

---

## Step 4 — Run the frontend

Open a **second terminal** in Cursor:

```bash
cd frontend
npm install
npm run dev
```

App will open at: http://localhost:3000

---

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | /                 | API info                           |
| GET    | /health           | Health check                       |
| GET    | /thresholds       | View adaptive thresholds           |
| POST   | /predict          | Classify a single tweet            |
| POST   | /predict/batch    | Classify up to 50 tweets at once   |
| GET    | /docs             | Interactive Swagger UI             |

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!! 🎉", "threshold_mode": "adaptive"}'
```

---

## Features

- **Single tweet analysis** — per-label probability bar chart + emotion cards
- **Batch analysis** — paste up to 50 tweets, download results as CSV
- **Adaptive thresholds** — per-emotion tuned τ for better rare emotion recall
- **Preprocessing visible** — shows cleaned tweet alongside original
- **Analysis history** — last 8 tweets shown in sidebar

---

## Paper results (for reference)

| Model                         | Macro-F1 | Micro-F1 | Jaccard |
|-------------------------------|----------|----------|---------|
| SVM (TF-IDF)                  | 42.1%    | —        | 38.5%   |
| BERT-base                     | 53.8%    | —        | —       |
| RoBERTa-base (fixed τ=0.5)    | ~56%     | ~67%     | ~52%    |
| RoBERTa-base + Adaptive τ     | 58.4%    | 68.9%    | 54.7%   |
