import { useState } from "react"

export default function BatchAnalyzer({ apiUrl }) {
  const [input, setInput]     = useState("")
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const lines = input.split("\n").map(l => l.trim()).filter(Boolean)

  async function runBatch() {
    if (!lines.length) return
    if (lines.length > 50) { setError("Max 50 tweets at once"); return }
    setLoading(true); setError(null); setResults([])
    try {
      const res = await fetch(`${apiUrl}/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines, threshold_mode: "adaptive" }),
      })
      if (!res.ok) throw new Error("Batch API error")
      const data = await res.json()
      setResults(data.results)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function downloadCSV() {
    const header = ["tweet", "detected_emotions", "dominant_emotion", "confidence", ...Array.from({length:11},(_,i)=>["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"][i])]
    const rows   = results.filter(r => r.result).map(r => {
      const res = r.result
      const probCols = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"].map(l => {
        const e = res.emotions.find(x => x.label === l)
        return e ? e.probability.toFixed(4) : "0"
      })
      return [
        `"${r.text.replace(/"/g,'""')}"`,
        `"${res.detected_emotions.join("|")}"`,
        res.dominant_emotion || "",
        res.confidence_score,
        ...probCols
      ].join(",")
    })
    const csv = [header.join(","), ...rows].join("\n")
    const a = document.createElement("a")
    a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(csv)
    a.download = "emotion_predictions.csv"
    a.click()
  }

  return (
    <div className="batch-layout">
      <div className="batch-input-card">
        <div className="input-header">
          <span className="input-label">Paste tweets — one per line (max 50)</span>
          <span className={"char-count " + (lines.length > 50 ? "over" : "")}>
          {lines.length}/50 tweets
          </span>
        </div>
        <textarea
          className="tweet-input batch"
          placeholder={"I love this so much!\nThis is so frustrating...\nNot sure how to feel about this."}
          value={input}
          onChange={e => setInput(e.target.value)}
          rows={10}
        />
        <div className="controls-row">
          {error && <span className="error-banner">{error}</span>}
          <button className="analyse-btn" onClick={runBatch} disabled={loading || !lines.length}>
          Analyse
          </button>
        </div>
      </div>

      {results.length > 0 && (
        <div className="batch-results">
          <div className="batch-results-header">
            <p className="section-label">{results.length} results</p>
            <button className="download-btn" onClick={downloadCSV}>⬇ Download CSV</button>
          </div>
          <div className="batch-table-wrap">
            <table className="batch-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Tweet</th>
                  <th>Detected Emotions</th>
                  <th>Dominant</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className={r.error ? "row-error" : ""}>
                    <td className="td-num">{i + 1}</td>
                    <td className="td-tweet">{r.text.slice(0, 80)}{r.text.length > 80 ? "…" : ""}</td>
                    <td className="td-emotions">
                      {r.error ? (
                        <span className="error-tag">Error</span>
                      ) : r.result.detected_emotions.length === 0 ? (
                        <span className="none-tag">none</span>
                      ) : (
                        r.result.detected_emotions.map(em => {
                          const colors = {anger:"#E24B4A",anticipation:"#BA7517",disgust:"#3B6D11",fear:"#534AB7",joy:"#1D9E75",love:"#D4537E",optimism:"#185FA5",pessimism:"#5F5E5A",sadness:"#378ADD",surprise:"#993C1D",trust:"#0F6E56"}
                          return (
                            <span key={em} className="batch-tag"
                              style={{ background: (colors[em]||"#888") + "22", color: colors[em]||"#888" }}>
                              {em}
                            </span>
                          )
                        })
                      )}
                    </td>
                    <td className="td-dominant">{r.result?.dominant_emotion || "—"}</td>
                    <td className="td-conf">{r.result ? (r.result.confidence_score * 100).toFixed(1) + "%" : "—"}</td>                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
