import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api.js'
import { Chips, ErrorBox, HighlightedResume, LABEL_NAMES, ScoreBar } from '../components.jsx'

export default function CandidatePage() {
  const { candidateId } = useParams()
  const [c, setC] = useState(null)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    api.getCandidate(candidateId).then(setC).catch(setError)
  }, [candidateId])

  if (error && !c) return <ErrorBox error={error} />
  if (!c) return <p className="muted">Loading…</p>

  async function remove() {
    if (!confirm(`Delete ${c.label}?`)) return
    await api.deleteCandidate(c.id)
    navigate(`/jobs/${c.job_id}`)
  }

  return (
    <>
      <p><Link to={`/jobs/${c.job_id}`}>← Shortlist</Link></p>
      <div className="row space">
        <h1>{c.label}</h1>
        <button className="danger" onClick={remove}>Delete</button>
      </div>

      <div className="grid-2">
        <section className="card">
          <h2>Match</h2>
          <ScoreBar score={c.match.score} />
          <p><strong>Has:</strong> <Chips items={c.match.matched} kind="ok" /></p>
          <p><strong>Missing:</strong> <Chips items={c.match.missing} kind="miss" /></p>
          <h2>Pulled from the resume</h2>
          <dl className="facts">
            {Object.entries(LABEL_NAMES).map(([k, name]) => (
              <div key={k}><dt>{name}</dt><dd><Chips items={c.entity_groups[k]} /></dd></div>
            ))}
          </dl>
        </section>
        <FitReport candidate={c} />
      </div>

      <section className="card">
        <h2>Resume</h2>
        <HighlightedResume text={c.resume_text} entities={c.entities} />
      </section>
    </>
  )
}

function FitReport({ candidate }) {
  const [report, setReport] = useState(candidate.fit_report)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function generate() {
    setBusy(true)
    setError(null)
    try {
      setReport(await api.fitReport(candidate.id))
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card">
      <div className="row space">
        <h2>Fit report</h2>
        <button onClick={generate} disabled={busy}>
          {busy ? 'Asking the LLM…' : report ? 'Refresh' : 'Generate fit report'}
        </button>
      </div>
      <ErrorBox error={error} />
      {!report && !busy && <p className="muted">The LLM reads the job and this resume, then writes a short fit summary and five interview questions for the gaps.</p>}
      {report && (
        <>
          <p>{report.summary}</p>
          <h3>Interview questions</h3>
          <ol>{report.questions.map((q, i) => <li key={i}>{q}</li>)}</ol>
          <p className="muted small">
            {report.model} · {new Date(report.created_at).toLocaleString()}
            {report.cached && ' · served from cache (no new LLM call)'}
          </p>
        </>
      )}
    </section>
  )
}
