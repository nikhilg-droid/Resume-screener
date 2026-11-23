import { Link, Route, Routes } from 'react-router-dom'
import JobsPage from './pages/JobsPage.jsx'
import JobPage from './pages/JobPage.jsx'
import CandidatePage from './pages/CandidatePage.jsx'

export default function App() {
  return (
    <>
      <header className="topbar">
        <Link to="/" className="brand">Resume Screener</Link>
        <span className="tagline">DistilBERT NER + Groq fit reports</span>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<JobsPage />} />
          <Route path="/jobs/:jobId" element={<JobPage />} />
          <Route path="/candidates/:candidateId" element={<CandidatePage />} />
          <Route path="*" element={<p>Page not found. <Link to="/">Back to jobs</Link></p>} />
        </Routes>
      </main>
    </>
  )
}
