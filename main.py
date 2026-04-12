from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import fitz
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import faiss
import json
import os
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = SentenceTransformer("all-MiniLM-L6-v2")

STORAGE_DIR = "storage"
REGISTRY_PATH = os.path.join(STORAGE_DIR, "registry.json")
NO_ANSWER_DISTANCE_THRESHOLD = 1.75

os.makedirs(STORAGE_DIR, exist_ok=True)

document_store = {
    "document_id": None,
    "chunks": [],
    "faiss_index": None,
    "filename": None
}


class UploadResponse(BaseModel):
    status: str
    message: str
    document_id: str | None = None
    filename: str | None = None
    num_chunks: int | None = None


class AskResponse(BaseModel):
    status: str
    question: str
    answer: str
    document_id: str | None = None
    filename: str | None = None
    top_matches: list


def get_document_paths(document_id: str):
    doc_dir = os.path.join(STORAGE_DIR, document_id)
    os.makedirs(doc_dir, exist_ok=True)

    return {
        "doc_dir": doc_dir,
        "faiss_index_path": os.path.join(doc_dir, "faiss_index.bin"),
        "metadata_path": os.path.join(doc_dir, "chunks_metadata.json")
    }


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def register_document(document_id: str, filename: str, num_chunks: int):
    registry = load_registry()

    registry.append({
        "document_id": document_id,
        "filename": filename,
        "num_chunks": num_chunks,
        "uploaded_at": datetime.utcnow().isoformat()
    })

    save_registry(registry)


def get_latest_document_id():
    registry = load_registry()
    if not registry:
        return None
    return registry[-1]["document_id"]


def extract_text_from_pdf(file_bytes):
    pages = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if page_text:
                pages.append({
                    "page_number": page_num,
                    "text": page_text
                })

    return pages


def chunk_text_with_metadata(pages, filename, document_id, max_chunk_chars=700):
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

    chunks = []
    global_chunk_index = 0

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        lines = [line.strip() for line in text.split("\n") if line.strip()]

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

        char_pointer = 0

        for section_title, section_lines in sections:
            current_lines = []
            current_length = 0
            section_start_char = char_pointer

            for line in section_lines:
                line_length = len(line) + 1

                if current_length + line_length > max_chunk_chars and current_lines:
                    chunk_text = "\n".join(current_lines).strip()
                    chunk_id = f"{document_id}_chunk_{global_chunk_index}"

                    chunks.append({
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "chunk_index": global_chunk_index,
                        "page_number": page_number,
                        "start_char": section_start_char,
                        "end_char": section_start_char + current_length,
                        "source_file": filename,
                        "section": section_title,
                        "text": chunk_text
                    })
                    global_chunk_index += 1

                    overlap_lines = current_lines[-2:] if len(current_lines) >= 2 else current_lines[:]
                    current_lines = overlap_lines[:]
                    current_length = sum(len(x) + 1 for x in current_lines)
                    section_start_char = char_pointer - current_length

                current_lines.append(line)
                current_length += line_length
                char_pointer += line_length

            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                chunk_id = f"{document_id}_chunk_{global_chunk_index}"

                chunks.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_index,
                    "page_number": page_number,
                    "start_char": section_start_char,
                    "end_char": section_start_char + current_length,
                    "source_file": filename,
                    "section": section_title,
                    "text": chunk_text
                })
                global_chunk_index += 1

    return chunks


def get_local_embeddings(text_list):
    embeddings = model.encode(text_list)
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):
    if embeddings.size == 0:
        raise ValueError("Embeddings array is empty.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index, path):
    faiss.write_index(index, path)


def load_faiss_index(path):
    if os.path.exists(path):
        return faiss.read_index(path)
    return None


def save_metadata(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_metadata(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_document_into_store(document_id: str):
    paths = get_document_paths(document_id)

    stored_chunks = load_metadata(paths["metadata_path"])
    stored_index = load_faiss_index(paths["faiss_index_path"])

    if not stored_chunks or stored_index is None:
        return False

    document_store["document_id"] = document_id
    document_store["chunks"] = stored_chunks
    document_store["faiss_index"] = stored_index
    document_store["filename"] = stored_chunks[0].get("source_file")

    return True


def refresh_document_store_from_disk():
    latest_document_id = get_latest_document_id()
    if latest_document_id:
        load_document_into_store(latest_document_id)


def get_available_documents():
    registry = load_registry()
    return list(reversed(registry))


def clean_text_for_summary(text):
    cleaned = text.replace("\n", " ").replace("•", "").strip()
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def generate_summary(question, top_matches):
    if not top_matches:
        return "No reliable answer found in the selected document."

    sections = []
    for match in top_matches:
        section = match.get("section", "general")
        if section not in sections:
            sections.append(section)

    section_text = ", ".join(sections)

    top_text = clean_text_for_summary(top_matches[0]["chunk_text"])
    short_text = top_text[:350].strip()

    return (
        f"For '{question}', the most relevant content is from the "
        f"{section_text} section(s). Key evidence shows: {short_text}..."
    )


def process_pdf_upload(file_bytes, filename):
    extracted_pages = extract_text_from_pdf(file_bytes)

    if not extracted_pages:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    document_id = str(uuid.uuid4())
    paths = get_document_paths(document_id)

    chunks = chunk_text_with_metadata(extracted_pages, filename, document_id)
    chunk_texts = [chunk["text"] for chunk in chunks]

    if not chunk_texts:
        raise HTTPException(status_code=400, detail="No valid chunks could be created from the PDF.")

    embeddings = get_local_embeddings(chunk_texts)
    faiss_index = build_faiss_index(embeddings)

    save_faiss_index(faiss_index, paths["faiss_index_path"])
    save_metadata(chunks, paths["metadata_path"])
    register_document(document_id, filename, len(chunks))

    document_store["document_id"] = document_id
    document_store["chunks"] = chunks
    document_store["faiss_index"] = faiss_index
    document_store["filename"] = filename

    return {
        "status": "success",
        "message": "Document uploaded, indexed, and saved successfully.",
        "document_id": document_id,
        "filename": filename,
        "num_chunks": len(chunks)
    }


def process_question(question: str):
    if not document_store["chunks"] or document_store["faiss_index"] is None:
        refresh_document_store_from_disk()

    if not document_store["chunks"] or document_store["faiss_index"] is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    question_embedding = get_local_embeddings([question])

    top_k = min(3, len(document_store["chunks"]))
    distances, indices = document_store["faiss_index"].search(question_embedding, top_k)

    top_matches = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx == -1:
            continue

        distance = float(distances[0][rank - 1])
        chunk = document_store["chunks"][idx]

        top_matches.append({
            "rank": rank,
            "score": round(distance, 4),
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "page_number": chunk.get("page_number"),
            "section": chunk.get("section", "general"),
            "source_file": chunk.get("source_file"),
            "chunk_text": chunk["text"]
        })

    if not top_matches or top_matches[0]["score"] > NO_ANSWER_DISTANCE_THRESHOLD:
        return {
            "status": "success",
            "question": question,
            "answer": "No reliable answer found in the selected document.",
            "document_id": document_store["document_id"],
            "filename": document_store["filename"],
            "top_matches": []
        }

    answer = generate_summary(question, top_matches)

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "document_id": document_store["document_id"],
        "filename": document_store["filename"],
        "top_matches": top_matches
    }


@app.on_event("startup")
def startup_load_storage():
    refresh_document_store_from_disk()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "filename": document_store["filename"],
            "num_chunks": len(document_store["chunks"]) if document_store["chunks"] else None,
            "summary": None,
            "best_match": None,
            "top_matches": [],
            "available_documents": get_available_documents(),
            "active_document_id": document_store["document_id"]
        }
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_api(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = await file.read()
    return process_pdf_upload(file_bytes, file.filename)


@app.post("/ask", response_model=AskResponse)
async def ask_api(question: str = Form(...)):
    return process_question(question)


@app.post("/select-document", response_class=HTMLResponse)
async def select_document(request: Request, document_id: str = Form(...)):
    success = load_document_into_store(document_id)

    if not success:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "filename": document_store["filename"],
                "num_chunks": len(document_store["chunks"]) if document_store["chunks"] else None,
                "summary": None,
                "best_match": "Could not load selected document.",
                "top_matches": [],
                "available_documents": get_available_documents(),
                "active_document_id": document_store["document_id"]
            }
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "filename": document_store["filename"],
            "num_chunks": len(document_store["chunks"]),
            "summary": None,
            "best_match": f"Loaded document: {document_store['filename']}",
            "top_matches": [],
            "available_documents": get_available_documents(),
            "active_document_id": document_store["document_id"]
        }
    )


@app.post("/upload-ui", response_class=HTMLResponse)
async def upload_ui(request: Request, file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        file_bytes = await file.read()
        result = process_pdf_upload(file_bytes, file.filename)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "filename": result["filename"],
                "num_chunks": result["num_chunks"],
                "summary": None,
                "best_match": result["message"],
                "top_matches": [],
                "available_documents": get_available_documents(),
                "active_document_id": document_store["document_id"]
            }
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "filename": document_store["filename"],
                "num_chunks": len(document_store["chunks"]) if document_store["chunks"] else None,
                "summary": None,
                "best_match": e.detail,
                "top_matches": [],
                "available_documents": get_available_documents(),
                "active_document_id": document_store["document_id"]
            }
        )


@app.post("/ask-ui", response_class=HTMLResponse)
async def ask_ui(request: Request, question: str = Form(...)):
    try:
        result = process_question(question)
        best_match = result["top_matches"][0]["chunk_text"] if result["top_matches"] else result["answer"]

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "filename": result["filename"],
                "num_chunks": len(document_store["chunks"]),
                "summary": result["answer"],
                "best_match": best_match,
                "top_matches": result["top_matches"],
                "available_documents": get_available_documents(),
                "active_document_id": document_store["document_id"]
            }
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "filename": document_store["filename"],
                "num_chunks": len(document_store["chunks"]) if document_store["chunks"] else None,
                "summary": None,
                "best_match": e.detail,
                "top_matches": [],
                "available_documents": get_available_documents(),
                "active_document_id": document_store["document_id"]
            }
        )