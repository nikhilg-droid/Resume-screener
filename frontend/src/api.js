// All calls go to our own back end. No API keys ever live in the front end.

async function request(path, options = {}) {
  const res = await fetch(`/api${path}`, options)
  if (res.status === 204) return null
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join('; ')
      : body.detail
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return body
}

const json = (method, data) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data),
})

export const api = {
  listJobs: () => request('/jobs'),
  getJob: (id) => request(`/jobs/${id}`),
  createJob: (data) => request('/jobs', json('POST', data)),
  updateJob: (id, data) => request(`/jobs/${id}`, json('PATCH', data)),
  shortlist: (jobId) => request(`/jobs/${jobId}/candidates`),
  addCandidate: (jobId, data) => request(`/jobs/${jobId}/candidates`, json('POST', data)),
  uploadPdf: (jobId, file) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/jobs/${jobId}/candidates/pdf`, { method: 'POST', body: form })
  },
  getCandidate: (id) => request(`/candidates/${id}`),
  deleteCandidate: (id) => request(`/candidates/${id}`, { method: 'DELETE' }),
  fitReport: (id) => request(`/candidates/${id}/fit-report`, { method: 'POST' }),
}
