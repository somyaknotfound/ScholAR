# scholAR

![scholAR banner](docs/assets/scholar-banner.svg)

**scholAR** is a full-stack academic retrieval and RAG system for students who have study material scattered across PDFs, class drives, WhatsApp groups, and local folders.

The name stands for **scholastic Augmented Retrieval**: a system that retrieves trustworthy academic context, answers with citations, and keeps credibility visible.

## What it does

scholAR lets a student:

- upload academic PDFs with document metadata;
- search naturally across indexed material;
- ask cited questions over retrieved sources;
- generate short-answer or MCQ practice material;
- continue conversations with persistent chat history;
- see source credibility, retrieval confidence, and answer credibility.

Example queries:

```text
deadlock prevention mutual exclusion hold and wait
banker's algorithm safe state
flow diffusion architecture
what is best-fit memory allocation?
give me MCQs on that
```

## Feature overview

### Document ingestion

- PDF upload through the React frontend
- Metadata capture: subject, course code, document type, source type, verification status, semester/year, institution, uploader type
- File hashing for document identity and duplicate-safe storage
- Page-wise text extraction using `pymupdf4llm`
- Automatic Tesseract OCR fallback for low-text scanned/image-only PDF pages
- Markdown-aware, page-aware chunking
- Cross-page merge pass for continued topics
- Stable document IDs and content-based chunk IDs

### Retrieval

- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB vector index with cosine similarity
- Stored BM25 lexical index
- Hybrid ranking:

```text
60% vector similarity
15% BM25 lexical relevance
5% exact keyword overlap
10% heading match
10% source credibility
```

- Metadata filters for subject, document type, source type, and minimum credibility
- Retrieval confidence gate to avoid answering from weak context

### RAG and study tools

- Cited answers over retrieved chunks
- Gemini primary LLM with Groq fallback
- Short-answer and MCQ generation
- Output credibility scoring based on retrieval strength, source credibility, source variety, and citation usage
- Weak-retrieval refusal instead of hallucinated answers

### Chat memory

- Persistent chat sessions stored in Neon PostgreSQL
- Session/message history with delete support
- Last-N-message context window
- Context-aware query rewriting for follow-up questions
- Guardrails to reject bad query rewrites that drop the previous topic

Example:

```text
User: flow diffusion
Assistant: FlowDiffusion is ...
User: generate a question on it
Retrieval query: flow diffusion generate a question on it
```

## Architecture

```mermaid
flowchart TD
    UI[React frontend] --> API[FastAPI backend]

    API --> Upload[PDF upload + metadata]
    Upload --> Files[Local PDF storage]
    Files --> Extract[Page-wise extraction]
    Extract --> Chunk[Page-aware chunking]
    Chunk --> Metadata[Metadata enrichment]

    Metadata --> Embed[MiniLM embeddings]
    Embed --> Chroma[ChromaDB vector store]
    Metadata --> BM25[BM25 lexical index]

    API --> Search[Hybrid retrieval]
    Chroma --> Search
    BM25 --> Search
    Search --> Gate[Retrieval confidence gate]

    API --> Chat[Chat sessions/messages]
    Chat --> Neon[Neon PostgreSQL]
    Chat --> Rewrite[Contextual query rewriting]
    Rewrite --> Search

    Gate -->|strong| Prompt[Citation-aware prompt]
    Gate -->|weak| Refuse[Refuse / ask for better query]
    Prompt --> Gemini[Gemini primary]
    Gemini -->|fallback| Groq[Groq fallback]
    Gemini --> Answer[Answer / questions with citations]
    Groq --> Answer
```

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React, Vite, CSS |
| API | FastAPI, Pydantic |
| PDF extraction | pymupdf4llm, PyMuPDF, Tesseract OCR |
| Chunking | LangChain text splitters + custom page-aware merge logic |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Lexical retrieval | Custom BM25 index |
| Chat persistence | Neon PostgreSQL, psycopg |
| LLM providers | Gemini, Groq |
| Evaluation | Custom retrieval/RAG sanity checks |

## Project structure

```text
backend/
├── data/
│   ├── pdfs/                    # local/uploaded PDFs, gitignored
│   ├── db/                      # local ChromaDB + BM25 indexes, gitignored
│   └── document_metadata.json   # document-level metadata
└── src/
    ├── extract.py               # PDF -> page-wise markdown text + OCR fallback
    ├── chunking.py              # page-wise text -> retrieval chunks
    ├── embed_index.py           # chunks -> embeddings -> ChromaDB
    ├── bm25_index.py            # stored lexical BM25 index
    ├── search.py                # vector/hybrid retrieval
    ├── llm_provider.py          # Gemini/Groq abstraction and fallback
    ├── rag_answer.py            # cited RAG answer generation
    ├── generate_questions.py    # short-answer and MCQ generation
    ├── output_credibility.py    # generated-output credibility scoring
    ├── contextual_query.py      # follow-up query rewriting
    ├── db.py                    # PostgreSQL connection layer
    ├── chat_store.py            # chat sessions/messages persistence
    ├── api.py                   # FastAPI app
    ├── evaluate_retrieval.py    # retrieval sanity checks
    ├── evaluate_rag.py          # RAG sanity checks
    └── requirements.txt

frontend/
├── src/
│   ├── App.jsx                  # React UI
│   ├── App.css                  # app styling
│   └── index.css
└── package.json
```

## Important files

| File | Purpose |
|---|---|
| `backend/src/api.py` | FastAPI app exposing upload, search, RAG, question generation, and chat endpoints |
| `backend/src/extract.py` | Converts PDFs into page-wise markdown text, with OCR fallback for weak scanned pages |
| `backend/src/chunking.py` | Splits page-wise extracted text into retrieval chunks |
| `backend/src/embed_index.py` | Generates embeddings, stable IDs, document metadata, and stores chunks in ChromaDB |
| `backend/src/bm25_index.py` | Builds and scores the local BM25 lexical index |
| `backend/src/search.py` | Performs vector/hybrid retrieval and retrieval-confidence checks |
| `backend/src/rag_answer.py` | Builds cited RAG prompts and calls the LLM provider layer |
| `backend/src/contextual_query.py` | Rewrites follow-up chat questions into standalone retrieval queries |
| `backend/src/chat_store.py` | Stores chat sessions and messages in Neon PostgreSQL |
| `backend/src/llm_provider.py` | Handles Gemini/Groq provider abstraction, fallback, and timeouts |
| `frontend/src/App.jsx` | Main React UI for Ask, Search, Chat, Practice, and Upload |

## Setup

Install backend dependencies:

```powershell
pip install -r backend/src/requirements.txt
```

For OCR fallback on scanned/image-only PDFs, install the Tesseract OCR executable and make sure it is available in your terminal:

```powershell
tesseract --version
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Fill in `.env`:

```env
LLM_PRIMARY_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=groq

DATABASE_URL=postgresql://username:password@host/database?sslmode=require

GEMINI_API_KEY=your_real_gemini_key
GEMINI_MODEL=gemini-3.5-flash
GEMINI_THINKING_LEVEL=low

GROQ_API_KEY=your_real_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```

The embedding model is downloaded and cached locally on first use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Run locally

Start the backend from the repository root:

```powershell
uvicorn backend.src.api:app --reload
```

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

## Main workflows

### Upload and index a PDF

Use the Upload tab in the frontend.

Backend flow:

```text
save PDF
-> compute file hash
-> generate document_id
-> update metadata
-> extract pages
-> chunk pages
-> embed chunks into Chroma
-> rebuild BM25
```

Uploaded PDFs are stored under:

```text
backend/data/pdfs/uploads/
```

### Search

CLI:

```powershell
python backend/src/search.py "best fit memory allocation" --top-k 5
```

With metadata filters:

```powershell
python backend/src/search.py "deadlock prevention" --subject "Operating Systems" --min-credibility 80
```

Frontend:

```text
Search tab -> enter query -> optional filters -> Search
```

### Ask cited questions

CLI:

```powershell
python backend/src/rag_answer.py "explain deadlock prevention" --top-k 3
```

Frontend:

```text
Ask tab -> enter question -> Ask
```

### Chat with context

Frontend:

```text
Chat tab -> New chat -> ask a question -> ask follow-up questions
```

Backend:

```text
POST /sessions
GET  /sessions
GET  /sessions/{session_id}
DELETE /sessions/{session_id}
POST /sessions/{session_id}/answer
```

### Generate practice questions

CLI:

```powershell
python backend/src/generate_questions.py "deadlock prevention" --type mcq --count 5
```

Frontend:

```text
Practice tab -> choose short answer / MCQ -> Generate
```

## API endpoints

```text
GET    /health
GET    /stats
POST   /upload
POST   /search
POST   /answer
POST   /questions
POST   /sessions
GET    /sessions
GET    /sessions/{session_id}
DELETE /sessions/{session_id}
POST   /sessions/{session_id}/answer
```

Example `/answer` request:

```json
{
  "query": "explain deadlock prevention",
  "top_k": 3,
  "subject": "Operating Systems",
  "doc_type": "lecture_notes"
}
```

Example `/sessions/{session_id}/answer` request:

```json
{
  "query": "give me MCQs on that",
  "top_k": 3,
  "context_messages": 5,
  "mode": "hybrid"
}
```

## Evaluation

Run retrieval checks:

```powershell
python backend/src/evaluate_retrieval.py
```

Run RAG checks:

```powershell
python backend/src/evaluate_rag.py
```

The evaluators check representative retrieval behavior, citation usage, weak-query refusal, provider fallback behavior, and output credibility.

## Key engineering decisions

### Page-aware chunking

Each PDF page is chunked independently first, then adjacent chunks are merged across page boundaries only when the topic appears to continue. This preserves page provenance without fragile string-position reconstruction.

### Stable IDs

Documents use deterministic IDs from source path and file hash. Chunks use stable hashes from document identity, page range, and normalized text. Chroma and BM25 both use the same chunk IDs, which keeps hybrid result merging reliable.

### Hybrid retrieval

Dense vector search handles semantic similarity. BM25, keyword overlap, and heading matching help exact academic terms. Source credibility is a small ranking factor, not a substitute for relevance.

### Credibility separation

scholAR separates:

- **retrieval strength**: how well the chunk matches the query;
- **source credibility**: how trustworthy the underlying document is;
- **output credibility**: how trustworthy the generated answer/question set is.

This prevents a trusted but irrelevant source from being presented as a good match.

### Context-aware chat

Follow-up questions are first rewritten into standalone retrieval queries using recent chat history. The system includes guardrails for invalid rewrites and rewrites that drop the previous topic. The final answer still sees the original user question and recent conversation context.

### Local-first storage split

Different data types are stored where they fit best:

```text
PDF files             -> local filesystem
Embeddings/chunks     -> ChromaDB
Lexical index         -> BM25 pickle
Chat sessions/messages -> Neon PostgreSQL
Secrets               -> .env
```

## Troubleshooting

### Backend starts, but `/sessions` fails

Check that `.env` contains a valid Neon/PostgreSQL connection string:

```env
DATABASE_URL=postgresql://username:password@host/database?sslmode=require
```

Then restart the backend:

```powershell
uvicorn backend.src.api:app --reload
```

### Frontend loads, but API calls fail

Make sure the backend is running at:

```text
http://127.0.0.1:8000
```

and the frontend is running at:

```text
http://localhost:5173
```

### Upload fails

Ensure `python-multipart` is installed:

```powershell
pip install -r backend/src/requirements.txt
```

Only PDF uploads are supported.

### Search or RAG gives weak results

Check whether metadata filters are excluding the correct document. Then rebuild indexes if needed:

```powershell
python backend/src/embed_index.py --rebuild backend/data/pdfs
```

### Scanned PDFs are not producing text

scholAR uses Tesseract only as an extraction fallback for low-text pages. Confirm the OCR engine is installed and visible from the terminal:

```powershell
tesseract --version
```

Then test the included image-only OCR sample:

```powershell
python backend/src/extract.py backend/data/pdfs/OCR_Test_Scanned.pdf
```

### LLM generation fails

Check `.env` for provider keys:

```env
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

If the primary provider fails because of quota, timeout, or key issues, scholAR attempts the configured fallback provider.

## Security

Real API keys and database URLs belong only in `.env` or environment variables.

If a key or database URL is accidentally exposed, rotate it immediately in the provider dashboard and replace the local `.env` value.

Never commit:

```text
.env
backend/data/
frontend/node_modules/
frontend/dist/
```

Safe to commit:

```text
.env.example
source code
README.md
frontend/package-lock.json
```
