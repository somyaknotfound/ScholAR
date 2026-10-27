import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const DEFAULT_FILTERS = {
  subject: '',
  doc_type: '',
  source_type: '',
  min_credibility: '',
}

const SUBJECT_OPTIONS = [
  'unknown',
  'Operating Systems',
  'Database Management Systems',
  'Computer Networks',
  'Data Structures',
  'Algorithms',
  'Machine Learning',
  'Artificial Intelligence',
  'Software Engineering',
]

const DOC_TYPE_OPTIONS = [
  'unknown',
  'lecture_notes',
  'student_notes',
  'question_paper',
  'assignment',
  'lab_manual',
  'syllabus',
  'textbook',
  'research_paper',
]

const SOURCE_TYPE_OPTIONS = [
  'unknown',
  'faculty_notes',
  'official_college',
  'official_publisher',
  'textbook',
  'previous_year_paper',
  'journal',
  'conference',
  'arxiv',
  'research_repository',
  'web_resource',
  'student_uploaded',
]

const VERIFIED_STATUS_OPTIONS = ['unknown', 'verified', 'unverified']
const EXAM_TYPE_OPTIONS = ['unknown', 'midsem', 'endsem', 'quiz', 'assignment', 'lab']
const UPLOADED_BY_OPTIONS = ['unknown', 'user', 'student', 'faculty', 'admin']

function cleanPayload(payload) {
  const cleaned = {}
  for (const [key, value] of Object.entries(payload)) {
    if (value === '') {
      cleaned[key] = null
    } else if (key === 'min_credibility' && value !== null) {
      cleaned[key] = Number(value)
    } else {
      cleaned[key] = value
    }
  }
  return cleaned
}

async function apiPost(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cleanPayload(payload)),
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed with status ${response.status}`)
  }
  return response.json()
}

async function apiDelete(path) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Delete failed with status ${response.status}`)
  }
  return response.json()
}

async function apiUpload(payload) {
  const formData = new FormData()
  for (const [key, value] of Object.entries(payload)) {
    if (value !== '' && value !== null && value !== undefined) {
      formData.append(key, value)
    }
  }

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Upload failed with status ${response.status}`)
  }

  return response.json()
}

function CredibilityBadge({ credibility }) {
  if (!credibility) return null
  const label = credibility.label || credibility.credibility_label || 'unknown'
  const score = credibility.score ?? credibility.credibility_score ?? 0
  return (
    <span className={`badge badge-${label}`}>
      {label} · {score}/100
    </span>
  )
}

function MatchBadge({ scores }) {
  const score = scores?.hybrid ?? scores?.vector ?? 0
  const strong = score >= 0.4
  return (
    <span className={`match-badge ${strong ? 'match-strong' : 'match-weak'}`}>
      {strong ? 'strong match' : 'weak match'} · {score.toFixed(3)}
    </span>
  )
}

function SourceCard({ source }) {
  return (
    <article className="source-card">
      <div className="source-topline">
        <span>{source.citation}</span>
        <CredibilityBadge
          credibility={{
            label: source.credibility_label,
            score: source.credibility_score,
          }}
        />
      </div>
      <h4>{source.source_file}</h4>
      <p>
        Pages {source.start_page}-{source.end_page}
      </p>
      <p className="muted">{source.section_heading || 'unknown heading'}</p>
    </article>
  )
}

function FilterBar({ filters, setFilters }) {
  function update(field, value) {
    setFilters((current) => ({ ...current, [field]: value }))
  }

  return (
    <div className="filter-grid">
      <label>
        Subject
        <input
          value={filters.subject}
          onChange={(event) => update('subject', event.target.value)}
          placeholder="Operating Systems"
        />
      </label>
      <label>
        Doc type
        <input
          value={filters.doc_type}
          onChange={(event) => update('doc_type', event.target.value)}
          placeholder="lecture_notes"
        />
      </label>
      <label>
        Source type
        <input
          value={filters.source_type}
          onChange={(event) => update('source_type', event.target.value)}
          placeholder="faculty_notes"
        />
      </label>
      <label>
        Min credibility
        <input
          value={filters.min_credibility}
          onChange={(event) => update('min_credibility', event.target.value)}
          placeholder="80"
          type="number"
          min="0"
          max="100"
        />
      </label>
    </div>
  )
}

function MetadataSelect({ label, value, options, onChange }) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option value={option} key={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  )
}

function SearchPanel() {
  const [query, setQuery] = useState('deadlock prevention')
  const [mode, setMode] = useState('hybrid')
  const [filters, setFilters] = useState({
    ...DEFAULT_FILTERS,
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function runSearch() {
    setLoading(true)
    setError('')
    try {
      const data = await apiPost('/search', {
        query,
        top_k: 3,
        source_file: null,
        mode,
        ...filters,
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Hybrid retrieval</p>
          <h2>Search sourced chunks</h2>
        </div>
        <button onClick={runSearch} disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
      <div className="filter-grid compact-grid">
        <MetadataSelect
          label="Retrieval mode"
          value={mode}
          options={['hybrid', 'hybrid_mmr', 'vector']}
          onChange={setMode}
        />
      </div>
      <FilterBar filters={filters} setFilters={setFilters} />
      {error && <p className="error">{error}</p>}

      {result?.retrieval_confidence && !result.retrieval_confidence.confident && (
        <p className="warning">
          Weak retrieval: {result.retrieval_confidence.reason}. These are only
          the closest available chunks, not reliable matches.
        </p>
      )}

      <div className="result-list">
        {result?.results?.map((item) => (
          <article
            className={`result-card ${
              item.scores.hybrid < 0.4 ? 'weak-result' : ''
            }`}
            key={item.id}
          >
            <div className="result-topline">
              <span>#{item.rank}</span>
              <MatchBadge scores={item.scores} />
            </div>
            <h3>{item.metadata.section_heading || item.metadata.source_file}</h3>
            <p>{item.preview}</p>
            <div className="score-grid">
              <span>hybrid {item.scores.hybrid?.toFixed(3)}</span>
              <span>vector {item.scores.vector?.toFixed(3)}</span>
              <span>bm25 {item.scores.bm25?.toFixed(3)}</span>
              <span>heading {item.scores.heading?.toFixed(3)}</span>
              {item.scores.mmr !== null && item.scores.mmr !== undefined && (
                <span>mmr {item.scores.mmr?.toFixed(3)}</span>
              )}
            </div>
            <p className="source-credibility-line">
              Source credibility:{' '}
              <CredibilityBadge
                credibility={{
                  label: item.metadata.credibility_label,
                  score: item.metadata.credibility_score,
                }}
              />
            </p>
            <p className="muted">
              {item.metadata.source_file} · pages {item.metadata.start_page}-
              {item.metadata.end_page} · {item.metadata.subject}
            </p>
          </article>
        ))}
      </div>
    </section>
  )
}

function AnswerPanel() {
  const [query, setQuery] = useState('explain deadlock prevention')
  const [filters, setFilters] = useState({
    ...DEFAULT_FILTERS,
  })
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function askQuestion() {
    setLoading(true)
    setError('')
    try {
      const data = await apiPost('/answer', {
        query,
        top_k: 3,
        source_file: null,
        mode: 'hybrid_mmr',
        temperature: 0.2,
        max_output_tokens: 1400,
        timeout_seconds: 30,
        include_context: false,
        ...filters,
      })
      setAnswer(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Cited RAG</p>
          <h2>Ask your documents</h2>
        </div>
        <button onClick={askQuestion} disabled={loading}>
          {loading ? 'Answering...' : 'Ask'}
        </button>
      </div>

      <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
      <FilterBar filters={filters} setFilters={setFilters} />
      {error && <p className="error">{error}</p>}

      {answer && (
        <div className="answer-card">
          <div className="answer-meta">
            <CredibilityBadge credibility={answer.output_credibility} />
            <span>
              {answer.provider_used || 'no provider'} · fallback{' '}
              {String(answer.fallback_used)}
            </span>
          </div>
          {answer.primary_error && (
            <p className="warning">Primary provider: {answer.primary_error}</p>
          )}
          <p className="answer-text">{answer.answer}</p>
          <h3>Sources</h3>
          <div className="sources-grid">
            {answer.sources?.map((source) => (
              <SourceCard source={source} key={source.chunk_id} />
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function ChatPanel() {
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('explain deadlock prevention')
  const [filters, setFilters] = useState({
    ...DEFAULT_FILTERS,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastResponse, setLastResponse] = useState(null)

  async function loadSessions() {
    try {
      const data = await apiGet('/sessions')
      setSessions(data.sessions || [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function loadSession(sessionId) {
    const data = await apiGet(`/sessions/${sessionId}`)
    setActiveSession(data.session)
    setMessages(data.messages || [])
  }

  async function newChat() {
    setError('')
    const session = await apiPost('/sessions', { title: 'New chat' })
    setActiveSession(session)
    setMessages([])
    setLastResponse(null)
    await loadSessions()
  }

  async function deleteChat(sessionId) {
    setError('')
    try {
      await apiDelete(`/sessions/${sessionId}`)
      if (activeSession?.id === sessionId) {
        setActiveSession(null)
        setMessages([])
        setLastResponse(null)
      }
      await loadSessions()
    } catch (err) {
      setError(err.message)
    }
  }

  async function sendMessage() {
    if (!query.trim()) return
    setLoading(true)
    setError('')
    setLastResponse(null)

    try {
      let session = activeSession
      if (!session) {
        session = await apiPost('/sessions', { title: 'New chat' })
        setActiveSession(session)
      }

      setMessages((current) => [
        ...current,
        {
          id: `temp-${Date.now()}`,
          role: 'user',
          content: query,
        },
      ])

      const response = await apiPost(`/sessions/${session.id}/answer`, {
        query,
        top_k: 3,
        source_file: null,
        mode: 'hybrid_mmr',
        temperature: 0.2,
        max_output_tokens: 1400,
        timeout_seconds: 30,
        context_messages: 5,
        include_context: false,
        ...filters,
      })

      setLastResponse(response)
      await loadSession(session.id)
      await loadSessions()
      setQuery('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSessions()
  }, [])

  return (
    <section className="panel chat-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Context-aware RAG</p>
          <h2>Chat with memory</h2>
        </div>
        <button onClick={newChat}>New chat</button>
      </div>

      <div className="chat-layout">
        <aside className="session-list">
          <p className="muted">Sessions</p>
          {sessions.length === 0 && (
            <p className="muted">No sessions yet. Start a new chat.</p>
          )}
          {sessions.map((session) => (
            <div
              className={`session-row ${
                activeSession?.id === session.id ? 'selected-session' : ''
              }`}
              key={session.id}
            >
              <button type="button" onClick={() => loadSession(session.id)}>
                {session.title}
              </button>
              <button
                type="button"
                className="delete-session"
                onClick={() => deleteChat(session.id)}
                aria-label={`Delete session ${session.title}`}
              >
                ×
              </button>
            </div>
          ))}
        </aside>

        <div className="chat-main">
          <div className="message-thread">
            {messages.length === 0 && (
              <p className="muted">
                Ask a question, then follow up with things like “give MCQs on
                that” or “explain the second point”.
              </p>
            )}
            {messages.map((message) => (
              <article className={`message-bubble ${message.role}`} key={message.id}>
                <span>{message.role}</span>
                <p>{message.content}</p>
              </article>
            ))}
          </div>

          {lastResponse?.query_rewrite && (
            <p className="warning">
              Retrieval query: {lastResponse.query_rewrite.query}
            </p>
          )}

          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a question or follow up on the previous answer..."
          />
          <FilterBar filters={filters} setFilters={setFilters} />
          {error && <p className="error">{error}</p>}
          <button onClick={sendMessage} disabled={loading}>
            {loading ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </div>
    </section>
  )
}

function QuestionsPanel() {
  const [topic, setTopic] = useState('deadlock prevention')
  const [questionType, setQuestionType] = useState('mcq')
  const [difficulty, setDifficulty] = useState('medium')
  const [filters, setFilters] = useState({
    ...DEFAULT_FILTERS,
  })
  const [questions, setQuestions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function generate() {
    setLoading(true)
    setError('')
    try {
      const data = await apiPost('/questions', {
        topic,
        question_type: questionType,
        count: 2,
        difficulty,
        top_k: 3,
        source_file: null,
        mode: 'hybrid_mmr',
        temperature: 0.35,
        max_output_tokens: 1800,
        timeout_seconds: 30,
        include_context: false,
        ...filters,
      })
      setQuestions(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Study mode</p>
          <h2>Generate practice questions</h2>
        </div>
        <button onClick={generate} disabled={loading}>
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </div>

      <textarea value={topic} onChange={(event) => setTopic(event.target.value)} />
      <div className="inline-controls">
        <div>
          <label>Type</label>
          <div className="segmented-control">
            <button
              type="button"
              className={questionType === 'mcq' ? 'selected' : ''}
              onClick={() => setQuestionType('mcq')}
            >
              MCQ
            </button>
            <button
              type="button"
              className={questionType === 'short' ? 'selected' : ''}
              onClick={() => setQuestionType('short')}
            >
              Short answer
            </button>
          </div>
        </div>
        <div>
          <label>Difficulty</label>
          <div className="segmented-control">
            {['easy', 'medium', 'hard'].map((level) => (
              <button
                type="button"
                key={level}
                className={difficulty === level ? 'selected' : ''}
                onClick={() => setDifficulty(level)}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
      </div>
      <FilterBar filters={filters} setFilters={setFilters} />
      {error && <p className="error">{error}</p>}

      {questions && (
        <div className="answer-card">
          <div className="answer-meta">
            <CredibilityBadge credibility={questions.output_credibility} />
            <span>
              {questions.provider_used || 'no provider'} · fallback{' '}
              {String(questions.fallback_used)}
            </span>
          </div>
          <p className="answer-text">{questions.questions}</p>
          <h3>Sources</h3>
          <div className="sources-grid">
            {questions.sources?.map((source) => (
              <SourceCard source={source} key={source.chunk_id} />
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function UploadPanel({ onUploaded }) {
  const [file, setFile] = useState(null)
  const [form, setForm] = useState({
    subject: 'unknown',
    course_code: 'unknown',
    doc_type: 'unknown',
    source_type: 'unknown',
    verified_status: 'unknown',
    semester: '0',
    year: '0',
    exam_type: 'unknown',
    institution: 'unknown',
    uploaded_by: 'unknown',
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function update(field, value) {
    setForm((current) => ({ ...current, [field]: value }))
  }

  async function uploadDocument() {
    if (!file) {
      setError('Choose a PDF first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await apiUpload({
        file,
        ...form,
      })
      setResult(data)
      onUploaded?.()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Ingestion</p>
          <h2>Upload and index a PDF</h2>
        </div>
        <button onClick={uploadDocument} disabled={loading}>
          {loading ? 'Indexing...' : 'Upload'}
        </button>
      </div>

      <label>
        PDF file
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
      </label>

      <div className="filter-grid upload-grid">
        <MetadataSelect
          label="Subject"
          value={form.subject}
          options={SUBJECT_OPTIONS}
          onChange={(value) => update('subject', value)}
        />
        <label>
          Course code
          <input
            value={form.course_code}
            onChange={(event) => update('course_code', event.target.value)}
            placeholder="CS204"
          />
        </label>
        <MetadataSelect
          label="Doc type"
          value={form.doc_type}
          options={DOC_TYPE_OPTIONS}
          onChange={(value) => update('doc_type', value)}
        />
        <MetadataSelect
          label="Source type"
          value={form.source_type}
          options={SOURCE_TYPE_OPTIONS}
          onChange={(value) => update('source_type', value)}
        />
        <MetadataSelect
          label="Verified status"
          value={form.verified_status}
          options={VERIFIED_STATUS_OPTIONS}
          onChange={(value) => update('verified_status', value)}
        />
        <label>
          Semester
          <input
            type="number"
            value={form.semester}
            onChange={(event) => update('semester', event.target.value)}
          />
        </label>
        <label>
          Year
          <input
            type="number"
            value={form.year}
            onChange={(event) => update('year', event.target.value)}
          />
        </label>
        <MetadataSelect
          label="Exam type"
          value={form.exam_type}
          options={EXAM_TYPE_OPTIONS}
          onChange={(value) => update('exam_type', value)}
        />
        <label>
          Institution
          <input
            value={form.institution}
            onChange={(event) => update('institution', event.target.value)}
            placeholder="unknown"
          />
        </label>
        <MetadataSelect
          label="Uploaded by"
          value={form.uploaded_by}
          options={UPLOADED_BY_OPTIONS}
          onChange={(value) => update('uploaded_by', value)}
        />
      </div>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="answer-card upload-summary">
          <div className="answer-meta">
            <span>Indexed successfully</span>
            <span>{result.indexing?.indexed ?? 0} new chunks</span>
          </div>
          <p>
            <strong>{result.metadata?.original_filename}</strong> saved as{' '}
            {result.source_path}
          </p>
          <div className="score-grid">
            <span>pages {result.indexing?.page_count}</span>
            <span>chunks {result.indexing?.chunks_created}</span>
            <span>indexed {result.indexing?.indexed}</span>
            <span>skipped existing {result.indexing?.skipped_already_present}</span>
            <span>BM25 chunks {result.indexing?.bm25_chunk_count}</span>
          </div>
          <p className="muted">document_id: {result.document_id}</p>
        </div>
      )}
    </section>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('ask')
  const [stats, setStats] = useState(null)

  function refreshStats() {
    fetch(`${API_BASE}/stats`)
      .then((response) => response.json())
      .then(setStats)
      .catch(() => setStats(null))
  }

  useEffect(() => {
    refreshStats()
  }, [])

  return (
    <main className="app-shell">
      <section className="hero-section">
        <div>
          <div className="brand-lockup" aria-label="scholAR">
            <span className="brand-mark">s</span>
            <span className="brand-word">
              schol<span>AR</span>
            </span>
            <small>Augmented Retrieval</small>
          </div>
          <h1>Study from trusted sources, not scattered folders.</h1>
          <p className="hero-copy">
            Search PDFs, ask cited questions, generate practice material, and see
            credibility signals for every answer.
          </p>
        </div>
        <div className="stats-card">
          <span>Indexed collection</span>
          <strong>{stats?.chunk_count ?? '—'} chunks</strong>
          <p>{stats?.pdf_count ?? '—'} PDFs · FastAPI connected</p>
        </div>
      </section>

      <nav className="tabs">
        <button
          className={activeTab === 'ask' ? 'active' : ''}
          onClick={() => setActiveTab('ask')}
        >
          Ask
        </button>
        <button
          className={activeTab === 'search' ? 'active' : ''}
          onClick={() => setActiveTab('search')}
        >
          Search
        </button>
        <button
          className={activeTab === 'chat' ? 'active' : ''}
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button
          className={activeTab === 'practice' ? 'active' : ''}
          onClick={() => setActiveTab('practice')}
        >
          Practice
        </button>
        <button
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          Upload
        </button>
      </nav>

      {activeTab === 'ask' && <AnswerPanel />}
      {activeTab === 'search' && <SearchPanel />}
      {activeTab === 'chat' && <ChatPanel />}
      {activeTab === 'practice' && <QuestionsPanel />}
      {activeTab === 'upload' && <UploadPanel onUploaded={refreshStats} />}
    </main>
  )
}

export default App
