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


def chunk_text_with_metadata(text, filename, max_chunk_chars=700):
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    section_keywords = {
        "experience",
        "technical skills",
        "skills",
        "projects",
        "education",
        "certifications",
        "summary",
        "work experience",
        "internship",
        "achievements"
    }

    def is_section_header(line):
        line_clean = line.lower().strip()
        return line_clean in section_keywords

    sections = []
    current_section_title = "general"
    current_section_lines = []

    for line in lines:
        if is_section_header(line):
            if current_section_lines:
                sections.append((current_section_title, current_section_lines))
            current_section_title = line
            current_section_lines = [line]
        else:
            current_section_lines.append(line)

    if current_section_lines:
        sections.append((current_section_title, current_section_lines))

    chunks = []
    chunk_index = 0
    char_pointer = 0

    for section_title, section_lines in sections:
        current_lines = []
        current_length = 0
        section_start_char = char_pointer

        for line in section_lines:
            line_length = len(line) + 1

            if current_length + line_length > max_chunk_chars and current_lines:
                chunk_text = "\n".join(current_lines).strip()

                chunks.append({
                    "chunk_index": chunk_index,
                    "start_char": section_start_char,
                    "end_char": section_start_char + current_length,
                    "source_file": filename,
                    "section": section_title,
                    "text": chunk_text
                })
                chunk_index += 1

                overlap_lines = current_lines[-2:] if len(current_lines) >= 2 else current_lines[:]
                current_lines = overlap_lines[:]
                current_length = sum(len(x) + 1 for x in current_lines)
                section_start_char = char_pointer - current_length

            current_lines.append(line)
            current_length += line_length
            char_pointer += line_length

        if current_lines:
            chunk_text = "\n".join(current_lines).strip()

            chunks.append({
                "chunk_index": chunk_index,
                "start_char": section_start_char,
                "end_char": section_start_char + current_length,
                "source_file": filename,
                "section": section_title,
                "text": chunk_text
            })
            chunk_index += 1

    return chunks


def get_local_embeddings(text_list):
    return model.encode(text_list)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
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
            request,
            "index.html",
            {
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
        request,
        "index.html",
        {
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
            request,
            "index.html",
            {
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
            "section": chunk.get("section", "general"),
            "chunk_text": chunk["text"]
        })

    best_match = top_matches[0]["chunk_text"] if top_matches else ""

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "filename": document_store["filename"],
            "num_chunks": len(document_store["chunks"]),
            "best_match": best_match,
            "top_matches": top_matches
        }
    )