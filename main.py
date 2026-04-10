from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = SentenceTransformer("all-MiniLM-L6-v2")

document_store = {
    "chunks": [],
    "embeddings": None,
    "filename": None
}

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text_with_metadata(text, filename, chunk_size=500, overlap=100, min_chunk_size=200):
    chunks = []
    start = 0
    text_length = len(text)
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            newline_pos = text.rfind("\n", start, end)
            space_pos = text.rfind(" ", start, end)

            candidate_end = end
            if newline_pos > start + min_chunk_size:
                candidate_end = newline_pos
            elif space_pos > start + min_chunk_size:
                candidate_end = space_pos

            end = candidate_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "source_file": filename,
                "text": chunk_text
            })
            chunk_index += 1

        if end >= text_length:
            break

        next_start = end - overlap
        if next_start < 0:
            next_start = 0

        if next_start > 0:
            newline_pos = text.find("\n", next_start, min(next_start + 50, text_length))
            space_pos = text.find(" ", next_start, min(next_start + 50, text_length))

            if newline_pos != -1:
                next_start = newline_pos + 1
            elif space_pos != -1:
                next_start = space_pos + 1

        start = next_start

    return chunks

def get_local_embeddings(text_list):
    return model.encode(text_list)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filename": None,
            "num_chunks": None,
            "best_match": None,
            "top_matches": []
        }
    )

@app.post("/upload-ui", response_class=HTMLResponse)
async def upload_ui(request: Request, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "filename": None,
                "num_chunks": None,
                "best_match": "Only PDF files are allowed.",
                "top_matches": []
            }
        )

    file_bytes = await file.read()
    extracted_text = extract_text_from_pdf(file_bytes)

    chunks = chunk_text_with_metadata(extracted_text, file.filename)
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = get_local_embeddings(chunk_texts)

    document_store["chunks"] = chunks
    document_store["embeddings"] = embeddings
    document_store["filename"] = file.filename

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "best_match": "Document uploaded successfully.",
            "top_matches": []
        }
    )

@app.post("/ask-ui", response_class=HTMLResponse)
async def ask_ui(request: Request, question: str = Form(...)):
    if not document_store["chunks"] or document_store["embeddings"] is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "filename": None,
                "num_chunks": None,
                "best_match": "No document uploaded yet.",
                "top_matches": []
            }
        )

    question_embedding = get_local_embeddings([question])
    similarities = cosine_similarity(question_embedding, document_store["embeddings"])[0]

    top_indices = np.argsort(similarities)[::-1][:3]

    top_matches = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = document_store["chunks"][idx]
        top_matches.append({
            "rank": rank,
            "score": round(float(similarities[idx]), 4),
            "chunk_text": chunk["text"]
        })

    best_match = top_matches[0]["chunk_text"] if top_matches else ""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filename": document_store["filename"],
            "num_chunks": len(document_store["chunks"]),
            "best_match": best_match,
            "top_matches": top_matches
        }
    )