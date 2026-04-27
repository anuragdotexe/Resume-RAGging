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
import re
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = SentenceTransformer("all-MiniLM-L6-v2")

STORAGE_DIR = "storage"
REGISTRY_PATH = os.path.join(STORAGE_DIR, "registry.json")
NO_ANSWER_DISTANCE_THRESHOLD = 1.75
STRONG_MATCH_DISTANCE_THRESHOLD = 1.05
PARTIAL_MATCH_DISTANCE_THRESHOLD = 1.45
MAX_JD_REQUIREMENTS = 8
MAX_INTERVIEW_QUESTIONS = 6
EXPERIENCE_SECTION_BOOST = 0.22
SKILLS_SECTION_PENALTY = 0.12

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


class InterviewQuestionsResponse(BaseModel):
    status: str
    job_description: str
    document_id: str | None = None
    filename: str | None = None
    questions: list
    matched_requirements: int
    gap_requirements: int


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


def build_page_title(filename: str | None):
    if filename:
        return f"Resume loaded: {filename}"
    return "Upload a resume PDF to begin."


def render_home(request: Request, **overrides):
    context = {
        "filename": document_store["filename"],
        "num_chunks": len(document_store["chunks"]) if document_store["chunks"] else None,
        "summary": None,
        "best_match": None,
        "top_matches": [],
        "available_documents": get_available_documents(),
        "active_document_id": document_store["document_id"],
        "job_description": "",
        "interview_questions": [],
        "interview_summary": None,
        "page_title": build_page_title(document_store["filename"])
    }
    context.update(overrides)
    return templates.TemplateResponse(request, "index.html", context)


def clean_text_for_summary(text):
    cleaned = text.replace("\n", " ").replace("•", "").strip()
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def truncate_text(text, limit=220):
    cleaned = clean_text_for_summary(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def normalize_text(text: str):
    return re.sub(r"\s+", " ", text.strip())


def split_into_bullets(text: str):
    normalized = text.replace("\r", "")
    parts = re.split(r"\n+|•", normalized)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def select_relevant_evidence_lines(question: str, top_matches: list, limit=3):
    question_words = {
        word for word in re.findall(r"[a-zA-Z0-9\+\#]+", question.lower())
        if len(word) > 2
    }

    scored_lines = []
    for match in top_matches:
        for line in split_into_bullets(match["chunk_text"]):
            line_lower = line.lower()
            overlap = sum(1 for word in question_words if word in line_lower)
            score = overlap

            if any(token in line_lower for token in ["reduced", "built", "developed", "led", "automated", "improving"]):
                score += 2
            if any(char.isdigit() for char in line):
                score += 1
            if match.get("section", "").lower() == "experience":
                score += 2
            elif match.get("section", "").lower() == "technical skills":
                score -= 1

            if score > 0:
                scored_lines.append((score, line, match))

    scored_lines.sort(key=lambda item: item[0], reverse=True)

    selected = []
    seen = set()
    for _, line, match in scored_lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "line": line,
            "section": match.get("section", "general"),
            "page_number": match.get("page_number")
        })
        if len(selected) == limit:
            break

    return selected


def classify_question(question: str):
    lowered = question.lower()
    if any(keyword in lowered for keyword in ["skill", "skills", "experience with", "know", "knowledge", "familiar"]):
        return "skills"
    if any(keyword in lowered for keyword in ["why", "impact", "achieve", "result", "outcome"]):
        return "impact"
    return "general"


def adjust_match_score(match: dict):
    adjusted = match["score"]
    section = match.get("section", "").lower()

    if section == "experience":
        adjusted -= EXPERIENCE_SECTION_BOOST
    elif section == "technical skills":
        adjusted += SKILLS_SECTION_PENALTY

    return round(adjusted, 4)


def rerank_top_matches(question: str, top_matches: list):
    question_type = classify_question(question)
    reranked = []

    for match in top_matches:
        adjusted_score = adjust_match_score(match)
        chunk_text_lower = match["chunk_text"].lower()
        if question_type in {"skills", "general"} and any(
            token in chunk_text_lower for token in ["built", "developed", "led", "reduced", "automated"]
        ):
            adjusted_score -= 0.08

        reranked.append({
            **match,
            "score": round(match["score"], 4),
            "adjusted_score": round(adjusted_score, 4)
        })

    reranked.sort(key=lambda item: item["adjusted_score"])

    for index, match in enumerate(reranked, start=1):
        match["rank"] = index

    return reranked


def split_job_description_into_requirements(job_description: str, max_items=MAX_JD_REQUIREMENTS):
    raw_lines = [line.strip() for line in job_description.splitlines() if line.strip()]
    candidates = []
    priority_keywords = {
        "experience", "required", "preferred", "must", "should", "responsible",
        "proficient", "knowledge", "expertise", "skills", "ability", "hands-on"
    }

    for line in raw_lines:
        normalized_line = re.sub(r"^[\-\*\u2022\d\.\)\( ]+", "", line).strip()
        if len(normalized_line) < 20:
            continue

        parts = re.split(r"(?<=[.!?])\s+|;\s+", normalized_line)
        for part in parts:
            sentence = part.strip(" -")
            if len(sentence) < 20:
                continue
            score = len(sentence)
            lowered = sentence.lower()
            score += sum(25 for keyword in priority_keywords if keyword in lowered)
            candidates.append((score, sentence))

    seen = set()
    ranked = []
    for score, sentence in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(sentence)
        if len(ranked) == max_items:
            break

    return ranked


def get_requirement_focus(requirement: str):
    cleaned = requirement.strip()
    words = cleaned.split()
    if len(words) <= 8:
        return cleaned
    return " ".join(words[:8]) + "..."


def build_question_text(requirement: str, resume_chunk: dict | None, match_label: str):
    focus_area = get_requirement_focus(requirement)
    if resume_chunk is None or match_label == "gap":
        return (
            f"This role emphasizes '{focus_area}'. Your resume does not show strong direct evidence for it. "
            f"How would you explain your readiness, learning plan, or adjacent experience in an interview?"
        )

    evidence = truncate_text(resume_chunk["text"], 140)
    section = resume_chunk.get("section", "general")

    if match_label == "strong":
        return (
            f"The job description asks for '{focus_area}'. Your resume mentions {section} experience such as "
            f"'{evidence}'. Can you walk through that example, the impact you created, and how it fits this role?"
        )

    return (
        f"The role expects '{focus_area}'. Your resume shows related evidence in the {section} section: "
        f"'{evidence}'. How would you connect that past work to this requirement during an interview?"
    )


def generate_interview_questions(job_description: str):
    if not document_store["chunks"] or document_store["faiss_index"] is None:
        refresh_document_store_from_disk()

    if not document_store["chunks"] or document_store["faiss_index"] is None:
        raise HTTPException(status_code=400, detail="Upload or load a resume before generating interview questions.")

    normalized_jd = job_description.strip()
    if not normalized_jd:
        raise HTTPException(status_code=400, detail="Paste a job description to generate interview questions.")

    requirements = split_job_description_into_requirements(normalized_jd)
    if not requirements:
        raise HTTPException(
            status_code=400,
            detail="Could not identify enough usable requirements from the job description."
        )

    questions = []
    matched_requirements = 0
    gap_requirements = 0

    for requirement in requirements[:MAX_INTERVIEW_QUESTIONS]:
        requirement_embedding = get_local_embeddings([requirement])
        distances, indices = document_store["faiss_index"].search(requirement_embedding, 1)

        resume_chunk = None
        score = None
        match_label = "gap"

        idx = int(indices[0][0])
        if idx != -1:
            score = float(distances[0][0])
            resume_chunk = document_store["chunks"][idx]
            if score <= STRONG_MATCH_DISTANCE_THRESHOLD:
                match_label = "strong"
                matched_requirements += 1
            elif score <= PARTIAL_MATCH_DISTANCE_THRESHOLD:
                match_label = "partial"
                matched_requirements += 1
            else:
                gap_requirements += 1
        else:
            gap_requirements += 1

        if match_label == "gap":
            resume_chunk = resume_chunk if score is not None and score <= NO_ANSWER_DISTANCE_THRESHOLD else None

        questions.append({
            "requirement": requirement,
            "focus_area": get_requirement_focus(requirement),
            "match_label": match_label,
            "score": round(score, 4) if score is not None else None,
            "question": build_question_text(requirement, resume_chunk, match_label),
            "resume_evidence": truncate_text(resume_chunk["text"], 240) if resume_chunk else "No strong matching resume evidence found.",
            "section": resume_chunk.get("section", "general") if resume_chunk else "gap",
            "page_number": resume_chunk.get("page_number") if resume_chunk else None
        })

    if gap_requirements == 0:
        summary = "All generated questions are tied to clear resume-to-JD overlaps."
    else:
        summary = (
            f"{matched_requirements} questions are built from direct or partial overlaps, and "
            f"{gap_requirements} focus on likely interview gaps you should prepare to explain."
        )

    return {
        "status": "success",
        "job_description": normalized_jd,
        "document_id": document_store["document_id"],
        "filename": document_store["filename"],
        "questions": questions,
        "matched_requirements": matched_requirements,
        "gap_requirements": gap_requirements,
        "summary": summary
    }


def generate_summary(question, top_matches):
    if not top_matches:
        return "No reliable answer found in the selected document."

    best_match = top_matches[0]
    evidence_lines = select_relevant_evidence_lines(question, top_matches)

    if evidence_lines:
        strongest_evidence = evidence_lines[0]["line"]
    else:
        strongest_evidence = truncate_text(best_match["chunk_text"], 220)

    answer_lines = [
        f"Yes. The resume shows relevant evidence for '{question}'.",
        "",
        "Strongest proof:",
    ]

    for item in evidence_lines:
        location = f"{item['section']} section"
        if item["page_number"]:
            location += f", page {item['page_number']}"
        answer_lines.append(f"- {item['line']} ({location})")

    if not evidence_lines:
        answer_lines.append(
            f"- {strongest_evidence} ({best_match.get('section', 'general')} section, page {best_match.get('page_number')})"
        )

    answer_lines.extend([
        "",
        "Interview-ready talking point:",
        f"Focus on {strongest_evidence}. Explain the business problem, the tools you used, and the measurable outcome."
    ])

    return "\n".join(answer_lines)


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

    top_matches = rerank_top_matches(question, top_matches)

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
    return render_home(request)


@app.post("/upload", response_model=UploadResponse)
async def upload_api(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = await file.read()
    return process_pdf_upload(file_bytes, file.filename)


@app.post("/ask", response_model=AskResponse)
async def ask_api(question: str = Form(...)):
    return process_question(question)


@app.post("/generate-interview-questions", response_model=InterviewQuestionsResponse)
async def generate_interview_questions_api(job_description: str = Form(...)):
    result = generate_interview_questions(job_description)
    return {
        "status": result["status"],
        "job_description": result["job_description"],
        "document_id": result["document_id"],
        "filename": result["filename"],
        "questions": result["questions"],
        "matched_requirements": result["matched_requirements"],
        "gap_requirements": result["gap_requirements"]
    }


@app.post("/select-document", response_class=HTMLResponse)
async def select_document(request: Request, document_id: str = Form(...)):
    success = load_document_into_store(document_id)

    if not success:
        return render_home(request, best_match="Could not load selected document.")

    return render_home(
        request,
        best_match=f"Loaded document: {document_store['filename']}",
        page_title=f"Interview prep ready for {document_store['filename']}"
    )


@app.post("/upload-ui", response_class=HTMLResponse)
async def upload_ui(request: Request, file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        file_bytes = await file.read()
        result = process_pdf_upload(file_bytes, file.filename)

        return render_home(
            request,
            filename=result["filename"],
            num_chunks=result["num_chunks"],
            best_match=result["message"],
            page_title=f"Resume loaded: {result['filename']}"
        )
    except HTTPException as e:
        return render_home(request, best_match=e.detail)


@app.post("/ask-ui", response_class=HTMLResponse)
async def ask_ui(request: Request, question: str = Form(...)):
    try:
        result = process_question(question)
        best_match = result["top_matches"][0]["chunk_text"] if result["top_matches"] else result["answer"]

        return render_home(
            request,
            filename=result["filename"],
            num_chunks=len(document_store["chunks"]),
            summary=result["answer"],
            best_match=best_match,
            top_matches=result["top_matches"]
        )
    except HTTPException as e:
        return render_home(request, best_match=e.detail)


@app.post("/generate-interview-questions-ui", response_class=HTMLResponse)
async def generate_interview_questions_ui(request: Request, job_description: str = Form(...)):
    try:
        result = generate_interview_questions(job_description)
        return render_home(
            request,
            filename=result["filename"],
            num_chunks=len(document_store["chunks"]),
            job_description=result["job_description"],
            interview_questions=result["questions"],
            interview_summary=result["summary"],
            page_title=f"Interview prep ready for {result['filename']}"
        )
    except HTTPException as e:
        return render_home(
            request,
            job_description=job_description,
            best_match=e.detail
        )
    
 
