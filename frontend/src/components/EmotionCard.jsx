export default function EmotionCard({ emotion }) {
  const pct = Math.round(emotion.probability * 100)
  return (
    <div className={`emotion-card ${emotion.detected ? "detected" : ""}`}
      style={emotion.detected ? { borderColor: emotion.color + "88" } : {}}>
      <div className="ec-top">
        <span className="ec-emoji">{emotion.emoji}</span>
        <div className="ec-info">
          <span className="ec-label">{emotion.label}</span>
          <span className="ec-pct" style={emotion.detected ? { color: emotion.color } : {}}>{pct}%</span>
        </div>
        {emotion.detected && (
          <span className="ec-dot" style={{ background: emotion.color }} />
        )}
      </div>
      <div className="ec-bar-track">
        <div
          className="ec-bar-fill"
          style={{
            width: `${pct}%`,
            background: emotion.detected ? emotion.color : "#D3D1C7",
          }}
        />
      </div>
      <p className="ec-desc">{emotion.description}</p>
    </div>
  )
}
