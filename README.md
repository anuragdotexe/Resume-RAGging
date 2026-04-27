# Resume-RAGging

Resume-RAGging is a FastAPI application for resume-based retrieval and interview preparation.

The app lets a user:
- upload a resume PDF
- store and switch between previously indexed resumes
- ask natural-language questions against the active resume
- paste a job description and generate personalized interview questions based on resume-to-JD matches and likely gaps

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
  - strongest proof
  - supporting evidence
  - interview-ready talking point

### 3. Job Description to Interview Questions
- Accepts a pasted job description
- Extracts top JD requirements
- Matches those requirements against the active resume
- Labels each requirement as a `strong`, `partial`, or `gap` match
- Generates personalized interview questions from resume evidence or missing areas

### 4. Multi-Resume Workspace
- Persists uploaded resumes in `storage/registry.json`
- Lets the user load previously indexed resumes from the UI

## Tech Stack

- FastAPI
- Jinja2 templates
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- NumPy

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
- `POST /generate-interview-questions` - generate personalized interview questions from a job description
- `POST /select-document` - switch active stored resume in the UI

## Notes

- The embedding model may need an initial download from Hugging Face the first time the app runs.
- Resume Q&A is currently stronger than the JD-question generation path; the JD flow still needs stricter evidence validation and better snippet selection.

# work in progress
