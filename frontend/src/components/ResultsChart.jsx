import { useEffect, useRef } from "react"

export default function ResultsChart({ emotions }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !emotions?.length) return
    const ctx    = canvas.getContext("2d")
    const dpr    = window.devicePixelRatio || 1
    const W      = canvas.offsetWidth
    const H      = 180

    canvas.width  = W * dpr
    canvas.height = H * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, W, H)

    const sorted  = [...emotions].sort((a, b) => b.probability - a.probability)
    const n       = sorted.length
    const barH    = Math.floor((H - 24) / n) - 4
    const labelW  = 100
    const pctW    = 44
    const trackW  = W - labelW - pctW - 16

    sorted.forEach((em, i) => {
      const y    = 4 + i * (barH + 4)
      const fill = Math.round(em.probability * trackW)
      const color = em.detected ? em.color : "#B4B2A9"

      // Label
      ctx.fillStyle = em.detected ? "#2C2C2A" : "#888780"
      ctx.font      = `${em.detected ? 500 : 400} 12px system-ui, sans-serif`
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText(`${em.emoji} ${em.label}`, 0, y + barH / 2)

      // Track
      ctx.fillStyle = "#F1EFE8"
      ctx.beginPath()
      ctx.roundRect(labelW, y, trackW, barH, 3)
      ctx.fill()

      // Fill
      if (fill > 0) {
        ctx.fillStyle = color
        ctx.globalAlpha = em.detected ? 1 : 0.55
        ctx.beginPath()
        ctx.roundRect(labelW, y, fill, barH, 3)
        ctx.fill()
        ctx.globalAlpha = 1
      }

      // Pct label
      ctx.fillStyle = em.detected ? "#2C2C2A" : "#888780"
      ctx.font      = "11px system-ui, sans-serif"
      ctx.textAlign = "right"
      ctx.fillText(`${Math.round(em.probability * 100)}%`, W, y + barH / 2)
    })
  }, [emotions])

  return (
    <div className="chart-wrap">
      <p className="section-label">Probability distribution</p>
      <canvas ref={canvasRef} style={{ width: "100%", height: 180, display: "block" }} />
    </div>
  )
}
