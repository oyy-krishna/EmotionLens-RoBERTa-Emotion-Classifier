import { useState, useRef } from "react"
import EmotionCard from "./components/EmotionCard"
import BatchAnalyzer from "./components/BatchAnalyzer"
import ResultsChart from "./components/ResultsChart"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

const TABS = ["single", "batch"]

export default function App() {
  const [tab, setTab]           = useState("single")
  const [text, setText]         = useState("")
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [history, setHistory]   = useState([])
  const [threshMode, setThreshMode] = useState("adaptive")
  const textareaRef = useRef(null)

  const charCount  = text.length
  const overLimit  = charCount > 2000

  async function analyze() {
    if (!text.trim() || overLimit) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, threshold_mode: threshMode }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "API error")
      }
      const data = await res.json()
      setResult(data)
      setHistory(prev => [{ text: text.slice(0, 80), emotions: data.detected_emotions, ts: Date.now() }, ...prev].slice(0, 8))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze()
  }

  const examples = [
    "I can't believe I got the job offer!! Best day ever 🎉",
    "So exhausted and hopeless... nothing ever changes 😭",
    "This is absolutely disgusting behavior, I'm furious 😡",
    "Scared but also excited about tomorrow's presentation!",
  ]

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <span className="brand-icon">🔬</span>
            <div>
              <h1 className="brand-title">EmotionLens</h1>
              <p className="brand-sub">RoBERTa · SemEval-2018 · 11 Emotions</p>
            </div>
          </div>
          <div className="header-badges">
            <span className="badge badge-blue">RoBERTa-base</span>
            <span className="badge badge-green">Multi-label</span>
            <span className="badge badge-purple">SemEval-2018</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Tab switcher */}
        <div className="tab-bar">
          {TABS.map(t => (
            <button key={t} className={`tab-btn ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t === "single" ? "📝 Single Tweet" : "📋 Batch Analysis"}
            </button>
          ))}
        </div>

        {tab === "single" && (
          <div className="single-layout">
            {/* Left: input */}
            <div className="input-panel">
              <div className="input-card">
                <div className="input-header">
                  <span className="input-label">Tweet or short text</span>
                  <span className={`char-count ${overLimit ? "over" : ""}`}>{charCount}/2000</span>
                </div>
                <textarea
                  ref={textareaRef}
                  className={`tweet-input ${overLimit ? "error" : ""}`}
                  placeholder="Type or paste a tweet here... (Ctrl+Enter to analyse)"
                  value={text}
                  onChange={e => setText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={5}
                />

                {/* Example pills */}
                <div className="examples-row">
                  <span className="examples-label">Try:</span>
                  {examples.map((ex, i) => (
                    <button key={i} className="example-pill" onClick={() => setText(ex)}>
                      {ex.slice(0, 32)}…
                    </button>
                  ))}
                </div>

                {/* Threshold toggle */}
                <div className="controls-row">
                  <div className="thresh-toggle">
                    <span className="ctrl-label">Threshold mode</span>
                    <div className="toggle-group">
                      {["adaptive", "fixed"].map(m => (
                        <button
                          key={m}
                          className={`toggle-btn ${threshMode === m ? "active" : ""}`}
                          onClick={() => setThreshMode(m)}
                        >
                          {m === "adaptive" ? "Adaptive (per-label)" : "Fixed (τ = 0.5)"}
                        </button>
                      ))}
                    </div>
                  </div>
                  <button
                    className="analyse-btn"
                    onClick={analyze}
                    disabled={loading || !text.trim() || overLimit}
                  >
                    {loading ? <span className="spinner" /> : "Analyse →"}
                  </button>
                </div>

                {error && <div className="error-banner">⚠️ {error}</div>}
              </div>

              {/* History */}
              {history.length > 0 && (
                <div className="history-card">
                  <p className="section-label">Recent analyses</p>
                  {history.map((h, i) => (
                    <div key={h.ts} className="history-row" onClick={() => setText(examples.find(e => e.startsWith(h.text.slice(0, 20))) || h.text)}>
                      <span className="history-text">{h.text}{h.text.length >= 80 ? "…" : ""}</span>
                      <div className="history-tags">
                        {(h.emotions || []).slice(0, 3).map(e => (
                          <span key={e} className="history-tag">{e}</span>
                        ))}
                        {h.emotions.length === 0 && <span className="history-tag muted">none</span>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right: results */}
            <div className="results-panel">
              {!result && !loading && (
                <div className="empty-state">
                  <div className="empty-icon">🧠</div>
                  <p className="empty-title">Ready to analyse</p>
                  <p className="empty-sub">Enter a tweet on the left and click Analyse</p>
                </div>
              )}

              {loading && (
                <div className="empty-state">
                  <div className="pulse-ring" />
                  <p className="empty-title">Analysing…</p>
                  <p className="empty-sub">Running RoBERTa inference</p>
                </div>
              )}

              {result && !loading && (
                <>
                  {/* Summary strip */}
                  <div className="summary-strip">
                    <div className="summary-item">
                      <span className="summary-label">Detected emotions</span>
                      <span className="summary-value">{result.detected_emotions.length || "none"}</span>
                    </div>
                    <div className="summary-item">
                      <span className="summary-label">Dominant</span>
                      <span className="summary-value dominant">{result.dominant_emotion || "—"}</span>
                    </div>
                    <div className="summary-item">
                      <span className="summary-label">Top confidence</span>
                      <span className="summary-value">{(result.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="summary-item">
                      <span className="summary-label">Threshold</span>
                      <span className="summary-value">{threshMode}</span>
                    </div>
                  </div>

                  {/* Detected badges */}
                  {result.detected_emotions.length > 0 && (
                    <div className="detected-row">
                      {result.detected_emotions.map(em => {
                        const e = result.emotions.find(x => x.label === em)
                        return (
                          <span key={em} className="detected-badge"
                            style={{ background: e?.color + "22", color: e?.color, border: `1px solid ${e?.color}55` }}>
                            {e?.emoji} {em} <span className="badge-prob">{(e?.probability * 100).toFixed(0)}%</span>
                          </span>
                        )
                      })}
                    </div>
                  )}

                  {/* Chart */}
                  <ResultsChart emotions={result.emotions} />

                  {/* Emotion cards grid */}
                  <div className="emotion-grid">
                    {result.emotions.map(e => (
                      <EmotionCard key={e.label} emotion={e} />
                    ))}
                  </div>

                  {/* Preprocessed text */}
                  {result.cleaned_text !== result.original_text.toLowerCase() && (
                    <div className="cleaned-text-card">
                      <p className="section-label">Preprocessed input</p>
                      <p className="cleaned-text">{result.cleaned_text}</p>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {tab === "batch" && <BatchAnalyzer apiUrl={API_URL} />}
      </main>

      <footer className="app-footer">
        Research project · Bennett University ·
        RoBERTa-base fine-tuned on SemEval-2018 Task 1 (Subtask E-c) ·
        Macro-F1: 58.4% · Jaccard: 54.7%
      </footer>
    </div>
  )
}
