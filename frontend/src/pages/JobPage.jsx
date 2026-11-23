import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api.js'
import { Chips, ErrorBox, ScoreBar } from '../components.jsx'
import { SAMPLE_RESUMES } from '../samples.js'

export default function JobPage() {
  const { jobId } = useParams()
  const [job, setJob] = useState(null)
  const [rows, setRows] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    Promise.all([api.getJob(jobId), api.shortlist(jobId)])
      .then(([j, r]) => { setJob(j); setRows(r) })
      .catch(setError)
  }, [jobId])

  useEffect(load, [load])

  if (error && !job) return <ErrorBox error={error} />
  if (!job) return <p className="muted">Loading…</p>

  return (
    <>
      <p><Link to="/">← All jobs</Link></p>
      <h1>{job.title}</h1>
      <div className="grid-2">
        <section className="card">
          <SkillsEditor job={job} onSaved={load} />
          <details>
            <summary>Job description</summary>
            <pre className="resume">{job.description}</pre>
          </details>
        </section>
        <AddCandidate jobId={job.id} onAdded={load} />
      </div>

      <section className="card">
        <h2>Shortlist <span className="muted">({rows.length})</span></h2>
        {rows.length === 0 ? <p className="muted">No candidates yet.</p> : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>#</th><th>Candidate</th><th>Match</th><th>Has</th><th>Missing</th><th>Exp.</th><th>Degree</th></tr>
              </thead>
              <tbody>
                {rows.map((c, i) => (
                  <tr key={c.id}>
                    <td>{i + 1}</td>
                    <td>
                      <Link to={`/candidates/${c.id}`}>{c.label}</Link>
                      {c.source === 'pdf' && <span className="badge">PDF</span>}
                      {c.has_fit_report && <span className="badge">report</span>}
                    </td>
                    <td><ScoreBar score={c.match.score} /></td>
                    <td><Chips items={c.match.matched} kind="ok" /></td>
                    <td><Chips items={c.match.missing} kind="miss" /></td>
                    <td>{c.years_experience != null ? `${c.years_experience} y` : '—'}</td>
                    <td>{c.entity_groups.DEGREE?.join(', ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  )
}

function SkillsEditor({ job, onSaved }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState('')
  const [error, setError] = useState(null)

  async function save() {
    try {
      const skills = value.split(',').map((s) => s.trim()).filter(Boolean)
      await api.updateJob(job.id, { required_skills: skills })
      setEditing(false)
      onSaved()
    } catch (err) {
      setError(err)
    }
  }

  return (
    <div>
      <h2>Required skills</h2>
      {editing ? (
        <div className="form">
          <input value={value} onChange={(e) => setValue(e.target.value)} />
          <div className="row">
            <button onClick={save}>Save</button>
            <button className="secondary" onClick={() => setEditing(false)}>Cancel</button>
          </div>
        </div>
      ) : (
        <p>
          <Chips items={job.required_skills} />{' '}
          <button className="link" onClick={() => { setValue(job.required_skills.join(', ')); setEditing(true) }}>edit</button>
        </p>
      )}
      <ErrorBox error={error} />
    </div>
  )
}

function AddCandidate({ jobId, onAdded }) {
  const [text, setText] = useState('')
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function run(fn) {
    setBusy(true)
    setError(null)
    try {
      await fn()
      setText('')
      setLabel('')
      onAdded()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card">
      <h2>Add a candidate</h2>
      <form className="form" onSubmit={(e) => { e.preventDefault(); run(() => api.addCandidate(jobId, { resume_text: text, label: label || null })) }}>
        <label>Label <span className="muted">(optional, e.g. &ldquo;Resume 12&rdquo; — no real names)</span>
          <input value={label} onChange={(e) => setLabel(e.target.value)} />
        </label>
        <label>Resume text
          <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} required />
        </label>
        <div className="row">
          <button type="submit" disabled={busy}>{busy ? 'Reading…' : 'Screen resume'}</button>
          {SAMPLE_RESUMES.map((s) => (
            <button key={s.label} type="button" className="secondary"
              onClick={() => { setText(s.text); setLabel(s.label) }}>{s.label}</button>
          ))}
        </div>
      </form>
      <div className="upload">
        <label className="file">
          or upload a PDF
          <input type="file" accept="application/pdf,.pdf" disabled={busy}
            onChange={(e) => { const f = e.target.files[0]; e.target.value = ''; if (f) run(() => api.uploadPdf(jobId, f)) }} />
        </label>
      </div>
      <ErrorBox error={error} />
    </section>
  )
}
