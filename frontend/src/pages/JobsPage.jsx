import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api.js'
import { Chips, ErrorBox } from '../components.jsx'
import { SAMPLE_JOB } from '../samples.js'

export default function JobsPage() {
  const [jobs, setJobs] = useState(null)
  const [error, setError] = useState(null)
  const [form, setForm] = useState({ title: '', description: '', skills: '' })
  const [saving, setSaving] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    api.listJobs().then(setJobs).catch(setError)
  }, [])

  async function submit(e) {
    e.preventDefault()
    setSaving(true)
    setError(null)
    try {
      const skills = form.skills.split(',').map((s) => s.trim()).filter(Boolean)
      const job = await api.createJob({
        title: form.title,
        description: form.description,
        // Leave skills empty and the NER model pulls them out of the job description.
        required_skills: skills.length ? skills : null,
      })
      navigate(`/jobs/${job.id}`)
    } catch (err) {
      setError(err)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid-2">
      <section className="card">
        <h2>New job</h2>
        <form onSubmit={submit} className="form">
          <label>Job title
            <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} required />
          </label>
          <label>Job description
            <textarea rows={10} value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })} required />
          </label>
          <label>Required skills <span className="muted">(optional, comma separated — leave blank to extract from the description)</span>
            <input value={form.skills} onChange={(e) => setForm({ ...form, skills: e.target.value })}
              placeholder="SQL, Python, Power BI" />
          </label>
          <div className="row">
            <button type="submit" disabled={saving}>{saving ? 'Creating…' : 'Create job'}</button>
            <button type="button" className="secondary"
              onClick={() => setForm({ ...SAMPLE_JOB, skills: '' })}>Load sample job</button>
          </div>
        </form>
        <ErrorBox error={error} />
      </section>

      <section className="card">
        <h2>Jobs</h2>
        {jobs === null && !error && <p className="muted">Loading…</p>}
        {jobs?.length === 0 && <p className="muted">No jobs yet. Create one to start screening.</p>}
        <ul className="list">
          {jobs?.map((j) => (
            <li key={j.id}>
              <Link to={`/jobs/${j.id}`} className="list-title">{j.title}</Link>
              <span className="muted"> · {j.candidate_count} candidate{j.candidate_count === 1 ? '' : 's'}</span>
              <div><Chips items={j.required_skills} /></div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
