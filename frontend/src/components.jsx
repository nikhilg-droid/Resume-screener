export const LABEL_NAMES = {
  SKILL: 'Skill',
  DEGREE: 'Degree',
  COLLEGE: 'College',
  COMPANY: 'Company',
  YEARS_EXP: 'Experience',
}

export function ErrorBox({ error }) {
  if (!error) return null
  return <div className="error" role="alert">{String(error.message || error)}</div>
}

export function ScoreBar({ score }) {
  const tone = score >= 70 ? 'good' : score >= 40 ? 'mid' : 'low'
  return (
    <div className="score" title={`${score}% of required skills found`}>
      <div className="score-track"><div className={`score-fill ${tone}`} style={{ width: `${score}%` }} /></div>
      <span className="score-num">{score}%</span>
    </div>
  )
}

export function Chips({ items, kind = '' }) {
  if (!items?.length) return <span className="muted">—</span>
  return (
    <span className="chips">
      {items.map((s) => <span key={s} className={`chip ${kind}`}>{s}</span>)}
    </span>
  )
}

// Shows the resume text with each entity the model found highlighted in its label colour.
export function HighlightedResume({ text, entities }) {
  const parts = []
  let pos = 0
  for (const [i, e] of [...entities].sort((a, b) => a.start - b.start).entries()) {
    if (e.start < pos) continue // overlapping spans: keep the first
    if (e.start > pos) parts.push(text.slice(pos, e.start))
    parts.push(
      <mark key={i} className={`ent ent-${e.label}`} title={LABEL_NAMES[e.label] || e.label}>
        {text.slice(e.start, e.end)}
      </mark>,
    )
    pos = e.end
  }
  parts.push(text.slice(pos))
  return (
    <>
      <div className="legend">
        {Object.entries(LABEL_NAMES).map(([k, v]) => <mark key={k} className={`ent ent-${k}`}>{v}</mark>)}
      </div>
      <pre className="resume">{parts}</pre>
    </>
  )
}
