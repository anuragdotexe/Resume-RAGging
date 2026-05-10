# Resume-RAGging

Resume-RAGging is a FastAPI application for resume-based retrieval and interview preparation.

The app lets a user:
- upload a resume PDF
- store and switch between previously indexed resumes
- ask natural-language questions against the active resume
- create an interview prep pack from the resume, job description, company context, and interview notes
- optionally run live company research when a Tavily API key is configured

## Current Features

### 1. Resume Upload and Indexing
- Uploads PDF resumes through the UI or API
- Extracts text with PyMuPDF
- Chunks resume content with page and section metadata
- Creates embeddings with `sentence-transformers`
- Stores FAISS indexes and chunk metadata on disk in `storage/`

### 2. Resume Q&A
- Accepts natural-language questions such as `Python and SQL skills`
- Retrieves the most relevant resume chunks
- Re-ranks matches to prefer stronger experience-based evidence
- Returns a cleaner answer with:
  - strongest proof bullets
  - supporting evidence
  - interview-ready talking point

### 3. Interview Prep Pack
- Accepts:
  - company name
  - role title
  - pasted job description
  - optional company research notes
  - optional interview experience notes
- Extracts top JD requirements
- Matches each requirement against the active resume
- Labels each requirement as a `strong`, `partial`, or `gap` match
- Generates personalized interview questions grounded in:
  - resume evidence
  - JD requirements
  - company context
  - interview patterns
- Shows a latency summary so the user understands the cost of local-only versus live-research flows

### 4. Optional Live Company Research
- Supports automatic company research when `TAVILY_API_KEY` is set
- Uses company name and role title to fetch external company/role context
- Falls back cleanly to manual pasted notes when live research is not configured
- Surfaces whether auto research was actually used in the generated prep pack

### 5. Multi-Resume Workspace
- Persists uploaded resumes in `storage/registry.json`
- Lets the user load previously indexed resumes from the UI

## Tech Stack

- FastAPI
- Jinja2 templates
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- NumPy
- Optional Tavily API integration for live research

## Run Locally

From the project root:

```bash
source venv/bin/activate
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

If dependencies are missing:

```bash
source venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --reload
```

## Optional Live Research Setup

To enable automatic company research, set a Tavily API key before starting the app:

```bash
export TAVILY_API_KEY="your_api_key_here"
uvicorn main:app --reload
```

If `TAVILY_API_KEY` is not set, the app still works, but company context and interview patterns must be pasted manually into the UI.

## Project Structure

```text
Resume-RAGing/
├── main.py
├── templates/
│   └── index.html
├── storage/
│   ├── registry.json
│   └── <document-id>/
│       ├── chunks_metadata.json
│       └── faiss_index.bin
├── requirements.txt
└── README.md
```

## API Routes

- `GET /` - main UI
- `POST /upload` - upload a resume PDF
- `POST /ask` - ask a question against the active resume
- `POST /generate-interview-questions` - generate an interview prep pack from resume, JD, and optional company/interview context
- `POST /select-document` - switch active stored resume in the UI

## Notes

- The embedding model may need an initial download from Hugging Face the first time the app runs.
- Resume Q&A is currently more reliable than the company-aware interview-prep flow.
- The interview-prep flow still needs stronger evidence validation, better snippet selection, and more precise separation between company context and interview-pattern research.

