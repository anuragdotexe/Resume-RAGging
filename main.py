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
from concurrent.futures import ThreadPoolExecutor
import logging
import requests

app = FastAPI()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger("resume_raging")

model = SentenceTransformer("all-MiniLM-L6-v2")

STORAGE_DIR = "storage"
REGISTRY_PATH = os.path.join(STORAGE_DIR, "registry.json")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
WEB_RESEARCH_PROVIDER = os.getenv("WEB_RESEARCH_PROVIDER", "tavily").strip().lower()
NO_ANSWER_DISTANCE_THRESHOLD = 1.75
STRONG_MATCH_DISTANCE_THRESHOLD = 1.05
PARTIAL_MATCH_DISTANCE_THRESHOLD = 1.45
MAX_JD_REQUIREMENTS = 8
MAX_INTERVIEW_QUESTIONS = 6
EXPERIENCE_SECTION_BOOST = 0.22
SKILLS_SECTION_PENALTY = 0.12
GENERAL_SECTION_PENALTY = 0.28
QUESTION_SIMILARITY_WORD_THRESHOLD = 4
MAX_JOB_DESCRIPTION_CHARS = 200

ROLE_MODE_CONFIG = {
    "auto": {
        "label": "Auto detect",
        "search_terms": "role interview questions responsibilities",
        "priority_topics": []
    },
    "sde": {
        "label": "SDE",
        "search_terms": "software engineer coding dsa system design behavioral interview questions",
        "priority_topics": ["data structures", "algorithms", "system design", "backend", "scalability", "behavioral"]
    },
    "ai_ml": {
        "label": "AI / ML",
        "search_terms": "applied ai machine learning llm rag agent system design prompt engineering interview questions",
        "priority_topics": ["llms", "rag", "agents", "prompt engineering", "evaluation", "model performance"]
    },
    "data": {
        "label": "Data",
        "search_terms": "analytics sql dashboards experimentation stakeholder communication interview questions",
        "priority_topics": ["sql", "analytics", "dashboards", "experimentation", "metrics", "stakeholder communication"]
    },
    "product": {
        "label": "Product",
        "search_terms": "product sense execution metrics stakeholder behavioral interview questions",
        "priority_topics": ["product sense", "execution", "metrics", "prioritization", "stakeholders", "behavioral"]
    }
}

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


class InterviewPrepResponse(BaseModel):
    status: str
    company_name: str | None = None
    role_title: str | None = None
    role_mode: str
    job_description: str
    document_id: str | None = None
    filename: str | None = None
    questions: list
    matched_requirements: int
    gap_requirements: int
    company_signals: list
    interview_signals: list
    latency_summary: str
    auto_research_used: bool
    research_status: str
    research_query: str | None = None


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
        "page_title": build_page_title(document_store["filename"]),
        "company_name": "",
        "role_title": "",
        "role_mode": "auto",
        "company_context": "",
        "interview_notes": "",
        "company_signals": [],
        "interview_signals": [],
        "latency_summary": "Resume embedding is cached after upload. The slowest future step will be live company research unless you paste research notes here.",
        "auto_research_used": False,
        "research_status": (
            "Auto research available via Tavily."
            if TAVILY_API_KEY else
            "Auto research is off. Add TAVILY_API_KEY to enable live company research."
        ),
        "research_query": None,
        "question_metrics": {
            "total": 0,
            "strong": 0,
            "partial": 0,
            "adjacent": 0,
            "gap": 0,
            "jd_resume": 0,
            "interview_pattern": 0
        },
        "question_groups": [],
        "role_mode_label": "Auto detect"
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
    bullets = []
    for part in parts:
        cleaned = normalize_text(part)
        if not cleaned:
            continue
        if len(cleaned) < 12:
            continue
        if re.fullmatch(r"[\d\-\–/: ]+", cleaned):
            continue
        bullets.append(cleaned)
    return bullets


def select_relevant_evidence_lines(question: str, top_matches: list, limit=3):
    scored_lines = []
    for match in top_matches:
        for line in split_into_bullets(match["chunk_text"]):
            score = score_evidence_line(question, line, match.get("section", ""))
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


def extract_focus_keywords(text: str):
    stop_words = {
        "with", "from", "that", "this", "into", "their", "there", "about", "across",
        "ability", "skills", "skill", "knowledge", "experience", "required", "preferred",
        "should", "would", "could", "role", "team", "work", "working", "using", "build",
        "strong", "excellent", "closely", "understand", "translate", "improve", "improving"
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z\-\+&/]+", text.lower())
    keywords = []
    seen = set()
    for word in words:
        if len(word) < 4 or word in stop_words or word in seen:
            continue
        seen.add(word)
        keywords.append(word)
        if len(keywords) == 6:
            break
    return keywords


def count_keyword_overlap(keywords: list[str], text: str):
    if not keywords:
        return 0
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def is_low_signal_line(line: str):
    cleaned = normalize_text(line)
    if len(cleaned) < 20:
        return True
    if re.fullmatch(r"[\d\-\–/: ]+", cleaned):
        return True
    low_signal_patterns = [
        r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}$",
        r"^(experience|technical skills|projects|education|certifications)$",
        r"^[A-Za-z&\- ]+intern$",
        r"^[A-Za-z&\- ]+engineer$"
    ]
    lowered = cleaned.lower()
    for pattern in low_signal_patterns:
        if re.fullmatch(pattern, cleaned) or re.fullmatch(pattern, lowered):
            return True
    return False


def score_evidence_line(requirement: str, line: str, section: str = ""):
    if is_low_signal_line(line):
        return -1

    keywords = extract_focus_keywords(requirement)
    overlap = count_keyword_overlap(keywords, line)
    lowered = line.lower()
    score = overlap * 3

    action_tokens = [
        "built", "developed", "designed", "engineered", "created", "launched",
        "automated", "optimized", "implemented", "integrated", "evaluated", "improved"
    ]
    if any(token in lowered for token in action_tokens):
        score += 3
    if any(char.isdigit() for char in line):
        score += 1
    if len(line.split()) >= 8:
        score += 1
    if section.lower() in {"experience", "projects"}:
        score += 2
    if section.lower() == "technical skills":
        score -= 1

    return score


def extract_context_signals(text: str, max_items=4):
    if not text.strip():
        return []

    candidates = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        normalized = re.sub(r"^[\-\*\u2022\d\.\)\( ]+", "", line).strip()
        if len(normalized) < 20:
            continue
        for part in re.split(r"(?<=[.!?])\s+|;\s+", normalized):
            sentence = part.strip()
            if len(sentence) < 20:
                continue
            keywords = extract_focus_keywords(sentence)
            score = len(keywords) + len(sentence) / 100
            if any(token in sentence.lower() for token in ["interview", "culture", "stakeholder", "ownership", "business", "analytics"]):
                score += 1.5
            candidates.append((score, sentence))

    ranked = []
    seen = set()
    for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(sentence)
        if len(ranked) == max_items:
            break

    return ranked


def get_research_query(company_name: str, role_title: str, job_description: str, role_mode: str):
    normalized_company = company_name.strip().replace("Open AI", "OpenAI")
    scope = " ".join(part for part in [normalized_company, role_title.strip()] if part).strip()
    jd_focus = ", ".join(split_job_description_into_requirements(job_description, max_items=3))
    role_config = get_role_mode_config(role_mode)
    query_parts = [
        scope or "company role interview",
        role_config["search_terms"],
        "company overview responsibilities interview questions expectations",
        jd_focus
    ]
    return " | ".join(part for part in query_parts if part)


def fetch_tavily_results(query: str, max_results=5):
    if not TAVILY_API_KEY:
        return {
            "results": [],
            "status": "Auto research is off. Add TAVILY_API_KEY to enable live company research."
        }

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False
    }

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=12
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        logger.warning("Tavily HTTP error for query '%s': %s", query, detail)
        return {
            "results": [],
            "status": f"Tavily HTTP error: {response.status_code}. Check the API key, quota, or request payload."
        }
    except requests.exceptions.Timeout as exc:
        logger.warning("Tavily timeout for query '%s': %s", query, exc)
        return {
            "results": [],
            "status": "Tavily request timed out."
        }
    except requests.exceptions.RequestException as exc:
        logger.warning("Tavily request error for query '%s': %s", query, exc)
        return {
            "results": [],
            "status": f"Tavily request error: {str(exc)}"
        }
    except json.JSONDecodeError:
        logger.warning("Tavily returned invalid JSON for query '%s'", query)
        return {
            "results": [],
            "status": "Tavily returned an unreadable response."
        }

    results = []
    for item in data.get("results", []):
        title = normalize_text(item.get("title", ""))
        content = normalize_text(item.get("content", ""))
        url = item.get("url", "")
        if content:
            results.append({
                "title": title,
                "content": content,
                "url": url
            })
    if not results:
        logger.info("Tavily returned zero usable results for query '%s'", query)
        return {
            "results": [],
            "status": "Tavily ran successfully but returned no usable results."
        }

    logger.info("Tavily returned %s result(s) for query '%s'", len(results), query)
    return {
        "results": results,
        "status": f"Auto research used {len(results)} live company result(s)."
    }


def run_company_research(company_name: str, role_title: str, job_description: str, role_mode: str):
    if WEB_RESEARCH_PROVIDER != "tavily" or not TAVILY_API_KEY:
        return {
            "query": get_research_query(company_name, role_title, job_description, role_mode),
            "results": [],
            "status": "Auto research is off. Add TAVILY_API_KEY to enable live company research."
        }

    query = get_research_query(company_name, role_title, job_description, role_mode)
    payload = fetch_tavily_results(query)
    payload["query"] = query
    return payload


def summarize_research_results(results: list[dict], max_items=4):
    lines = []
    for item in results[:max_items]:
        title = item.get("title") or "Research result"
        content = truncate_text(item.get("content", ""), 260)
        if content:
            lines.append(f"{title}: {content}")
    return lines


def split_research_results(results: list[dict]):
    company_results = []
    interview_results = []

    for item in results:
        title = (item.get("title") or "").lower()
        url = (item.get("url") or "").lower()
        content = (item.get("content") or "").lower()
        combined = " ".join([title, url, content])

        if any(token in combined for token in ["interview question", "interview guide", "questions & answers", "glassdoor", "hellointerview", "igotanoffer", "careery"]):
            interview_results.append(item)
        else:
            company_results.append(item)

    return company_results, interview_results


def estimate_latency_summary(has_company_context: bool, has_interview_notes: bool, live_web_enabled: bool = False):
    if live_web_enabled:
        return (
            "Estimated latency: resume retrieval 1-2s, JD analysis under 1s, live company/interview search 4-10s, "
            "final synthesis under 1s. Cache company research to keep repeat runs fast."
        )
    if has_company_context or has_interview_notes:
        return (
            "Estimated latency: resume retrieval 1-2s, JD and pasted research analysis under 1s each, "
            "final synthesis under 1s. This is much faster than live web search because research is already provided."
        )
    return (
        "Estimated latency: current local flow should stay around 1-3s after resume upload. "
        "If you later add live web search, expect the research step to dominate total latency."
    )


def get_research_status(auto_research_used: bool, live_results_count: int, research_status: str):
    if research_status:
        return research_status
    if auto_research_used and live_results_count > 0:
        return f"Auto research used {live_results_count} live company result(s)."
    if TAVILY_API_KEY:
        return "Auto research is configured. If company name and role are provided, live company search will run."
    return "Auto research is off. Add TAVILY_API_KEY to enable live company research."


def select_supporting_evidence_for_requirement(requirement: str, resume_chunk: dict | None):
    if not resume_chunk:
        return "No direct resume evidence found."

    scored_lines = []
    for line in split_into_bullets(resume_chunk["text"]):
        score = score_evidence_line(requirement, line, resume_chunk.get("section", ""))
        if score > 0:
            scored_lines.append((score, line))

    scored_lines.sort(key=lambda item: item[0], reverse=True)
    best_line = scored_lines[0][1] if scored_lines else truncate_text(resume_chunk["text"], 220)

    if count_keyword_overlap(extract_focus_keywords(requirement), best_line) == 0 and resume_chunk.get("section", "").lower() == "technical skills":
        return "No direct resume evidence found."

    return truncate_text(best_line, 220)


def classify_requirement_match(requirement: str, score: float | None, resume_chunk: dict | None):
    if score is None or resume_chunk is None:
        return "gap"

    keywords = extract_focus_keywords(requirement)
    chunk_text = resume_chunk["text"]
    keyword_overlap = count_keyword_overlap(keywords, chunk_text)
    section = resume_chunk.get("section", "").lower()

    if section == "general":
        if score > 0.85 or keyword_overlap < 3:
            return "gap"
        return "adjacent"
    if keyword_overlap == 0 and section == "technical skills":
        return "gap"
    if section == "technical skills":
        if score <= 0.9 and keyword_overlap >= 2:
            return "adjacent"
        return "gap"
    if score <= STRONG_MATCH_DISTANCE_THRESHOLD and keyword_overlap >= 1:
        return "strong"
    if score <= 1.2 and keyword_overlap >= 1:
        return "partial"
    if score <= 1.0 and section in {"experience", "projects"}:
        return "adjacent"
    return "gap"


def build_question_text(requirement: str, resume_chunk: dict | None, match_label: str, resume_evidence: str | None = None):
    focus_area = get_requirement_focus(requirement)
    if resume_chunk is None or match_label == "gap":
        return (
            f"This role emphasizes '{focus_area}'. Your resume does not show strong direct evidence for it. "
            f"How would you explain your readiness, learning plan, or adjacent experience in an interview?"
        )

    evidence = truncate_text(resume_evidence or resume_chunk["text"], 140)
    section = resume_chunk.get("section", "general")

    if match_label == "strong":
        if section.lower() == "technical skills":
            return (
                f"The job description asks for '{focus_area}'. Your resume explicitly lists related capability in the "
                f"{section} section such as '{evidence}'. How would you back that skill up with a concrete project or work example in an interview?"
            )
        return (
            f"The job description asks for '{focus_area}'. Your resume mentions {section} experience such as "
            f"'{evidence}'. Can you walk through that example, the impact you created, and how it fits this role?"
        )

    if match_label == "adjacent":
        return (
            f"The role expects '{focus_area}'. Your resume shows adjacent evidence in the {section} section: "
            f"'{evidence}'. How would you position that experience as relevant, and where would you be honest about the gap?"
        )

    return (
        f"The role expects '{focus_area}'. Your resume shows related evidence in the {section} section: "
        f"'{evidence}'. How would you connect that past work to this requirement during an interview?"
    )


def normalize_topic_label(text: str):
    candidate = normalize_text(text)
    candidate = re.sub(r"^[A-Za-z0-9 &\-]+\:\s*", "", candidate)
    candidate = re.sub(r"^(the most common|common|top)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(interview questions?|questions? and answers?)\b", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bfor 20\d{2}\b", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;|-")
    return candidate


def detect_role_mode(role_title: str, job_description: str, requested_mode: str):
    if requested_mode in ROLE_MODE_CONFIG and requested_mode != "auto":
        return requested_mode

    combined = f"{role_title} {job_description}".lower()
    if any(token in combined for token in ["sde", "software engineer", "backend", "frontend", "full stack"]):
        return "sde"
    if any(token in combined for token in ["ai", "ml", "machine learning", "llm", "rag", "applied ai", "genai"]):
        return "ai_ml"
    if any(token in combined for token in ["data analyst", "data scientist", "analytics", "bi ", "power bi", "sql"]):
        return "data"
    if any(token in combined for token in ["product manager", "pm ", "product sense"]):
        return "product"
    return "auto"


def get_role_mode_config(role_mode: str):
    return ROLE_MODE_CONFIG.get(role_mode, ROLE_MODE_CONFIG["auto"])


def get_role_mode_label(role_mode: str):
    return get_role_mode_config(role_mode)["label"]


def build_question_metrics(questions: list[dict]):
    metrics = {
        "total": len(questions),
        "strong": 0,
        "partial": 0,
        "adjacent": 0,
        "gap": 0,
        "jd_resume": 0,
        "interview_pattern": 0
    }

    for item in questions:
        match_label = item.get("match_label", "gap")
        if match_label in metrics:
            metrics[match_label] += 1
        source = item.get("question_source", "")
        if source == "jd + resume":
            metrics["jd_resume"] += 1
        elif source == "interview pattern":
            metrics["interview_pattern"] += 1

    return metrics


def build_question_groups(questions: list[dict]):
    groups = []
    ordered_sources = ["jd + resume", "interview pattern"]
    titles = {
        "jd + resume": "Personalized Role-Fit Questions",
        "interview pattern": "Commonly Asked Role Questions"
    }
    descriptions = {
        "jd + resume": "These are generated from the role brief and the strongest matching evidence found in the active resume.",
        "interview pattern": "These are derived from live interview-pattern research and adapted to the active resume."
    }

    for source in ordered_sources:
        items = [item for item in questions if item.get("question_source") == source]
        if not items:
            continue
        groups.append({
            "source": source,
            "title": titles.get(source, source.title()),
            "description": descriptions.get(source, ""),
            "items": items
        })

    return groups


def categorize_requirement(requirement: str, role_mode: str, question_source: str):
    lowered = requirement.lower()
    if question_source == "interview pattern":
        if any(token in lowered for token in ["system design", "architecture", "scalable", "backend", "design a"]):
            return "role pattern: system design"
        if any(token in lowered for token in ["behavioral", "collaboration", "stakeholder", "leadership"]):
            return "role pattern: behavioral"
        if any(token in lowered for token in ["rag", "agent", "llm", "prompt", "model", "evaluation"]):
            return "role pattern: technical depth"
        return "role pattern"

    if any(token in lowered for token in ["collaborate", "stakeholder", "communicat", "business", "research teams"]):
        return "behavioral / collaboration"
    if any(token in lowered for token in ["system design", "architecture", "scalable", "backend", "pipeline", "deploy"]):
        return "system / architecture"
    if any(token in lowered for token in ["sql", "dashboard", "analysis", "metrics", "experiment"]):
        return "analytics / problem solving"
    if any(token in lowered for token in ["prompt", "llm", "rag", "agent", "model"]):
        return "domain / technical depth"
    if role_mode == "sde":
        return "software engineering"
    return "core fit"


def build_why_asked(requirement: str, question_source: str, match_label: str, category: str):
    if question_source == "interview pattern":
        return (
            f"This topic appears in role-specific interview patterns, so it is likely to be asked to test your readiness in {category}."
        )
    if match_label == "gap":
        return (
            "This is being asked because the JD emphasizes it, but your resume does not show strong direct proof yet."
        )
    if match_label == "adjacent":
        return (
            "This is being asked to see whether you can translate adjacent experience into the exact needs of the role."
        )
    return (
        "This is being asked because it is a core JD requirement and your resume suggests you should be able to defend it with specifics."
    )


def build_prep_tip(question_source: str, match_label: str, resume_evidence: str, role_mode: str, category: str):
    if question_source == "interview pattern":
        if role_mode == "sde":
            return "Prepare a structured answer with approach, tradeoffs, edge cases, and production considerations."
        if role_mode == "ai_ml":
            return "Prepare to explain the architecture, evaluation approach, tradeoffs, and how you would productionize the solution."
        return "Prepare a structured answer with context, decision process, execution details, and measurable outcome."

    if match_label == "gap":
        return "Be explicit about the gap, then bridge it with adjacent projects, fast-learning examples, and a concrete ramp-up plan."
    if match_label == "adjacent":
        return f"Anchor your answer in '{truncate_text(resume_evidence, 90)}' and explain why that experience transfers to this requirement."
    if category == "behavioral / collaboration":
        return "Answer in STAR format and emphasize cross-functional alignment, decisions, and outcomes."
    if category == "system / architecture":
        return "Explain the system context, your design choices, the tradeoffs you considered, and how you would scale or monitor it."
    return "Use one concrete example, explain your exact contribution, the tools involved, and the measurable impact."


def extract_interview_topics(interview_signals: list[str], max_items=2):
    topics = []
    seen = set()
    for signal in interview_signals:
        lowered = signal.lower()
        if ":" in signal:
            signal = signal.split(":", 1)[1]
        fragments = re.split(r",|\.|;|\|", signal)
        for fragment in fragments:
            candidate = normalize_topic_label(fragment)
            if len(candidate) < 18:
                continue
            candidate_lower = candidate.lower()
            if candidate_lower in seen:
                continue
            if any(token in candidate_lower for token in ["transformer", "rag", "prompt", "agent", "system design", "llm", "architecture", "evaluation"]):
                seen.add(candidate_lower)
                topics.append(candidate)
                if len(topics) == max_items:
                    return topics
        if "interview question" in lowered and len(topics) < max_items:
            fallback = normalize_topic_label(truncate_text(signal, 120))
            if fallback.lower() not in seen:
                seen.add(fallback.lower())
                topics.append(fallback)
                if len(topics) == max_items:
                    return topics
    return topics


def are_questions_similar(left: str, right: str):
    left_words = {
        word for word in re.findall(r"[a-zA-Z0-9\+\#]+", left.lower())
        if len(word) > 3
    }
    right_words = {
        word for word in re.findall(r"[a-zA-Z0-9\+\#]+", right.lower())
        if len(word) > 3
    }
    return len(left_words & right_words) >= QUESTION_SIMILARITY_WORD_THRESHOLD


def dedupe_questions(questions: list[dict], limit=MAX_INTERVIEW_QUESTIONS):
    deduped = []
    for item in questions:
        if any(
            are_questions_similar(item["requirement"], existing["requirement"]) or
            item["resume_evidence"] == existing["resume_evidence"]
            for existing in deduped
        ):
            continue
        deduped.append(item)
        if len(deduped) == limit:
            break
    return deduped


def score_requirement_match(requirement: str, score: float, resume_chunk: dict):
    section = resume_chunk.get("section", "").lower()
    keyword_overlap = count_keyword_overlap(extract_focus_keywords(requirement), resume_chunk["text"])
    section_bonus = 0

    if section in {"experience", "projects"}:
        section_bonus += 0.18
    elif section == "technical skills":
        section_bonus -= 0.05
    elif section == "general":
        section_bonus -= GENERAL_SECTION_PENALTY

    return round((2.2 - score) + (keyword_overlap * 0.14) + section_bonus, 4)


def find_best_resume_match_for_requirement(requirement: str, top_k: int = 5):
    requirement_embedding = get_local_embeddings([requirement])
    search_k = min(top_k, len(document_store["chunks"]))
    distances, indices = document_store["faiss_index"].search(requirement_embedding, search_k)

    candidates = []
    for idx, distance in zip(indices[0], distances[0]):
        idx = int(idx)
        if idx == -1:
            continue
        chunk = document_store["chunks"][idx]
        semantic_score = float(distance)
        candidates.append({
            "chunk": chunk,
            "score": semantic_score,
            "match_strength": score_requirement_match(requirement, semantic_score, chunk)
        })

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item["match_strength"], reverse=True)
    best = candidates[0]
    return best["chunk"], best["score"]


def build_interview_pattern_questions(interview_signals: list[str], company_name: str, role_title: str, role_mode: str):
    questions = []
    topics = extract_interview_topics(interview_signals, max_items=2)

    for topic in topics:
        resume_chunk, score = find_best_resume_match_for_requirement(topic)
        match_label = classify_requirement_match(topic, score, resume_chunk)
        evidence = select_supporting_evidence_for_requirement(topic, resume_chunk if match_label != "gap" else None)

        if match_label == "gap":
            question_text = (
                f"This is a commonly asked topic for a {role_title or 'target role'} interview: '{truncate_text(topic, 70)}'. "
                f"What answer would you give if this comes up at {company_name or 'the company'}, and how would you bridge any missing direct experience?"
            )
            resume_chunk = None
        elif match_label == "adjacent":
            section = resume_chunk.get("section", "general") if resume_chunk else "general"
            question_text = (
                f"This is a commonly asked topic for a {role_title or 'target role'} interview: '{truncate_text(topic, 70)}'. "
                f"Using your {section} evidence '{truncate_text(evidence, 120)}', how would you connect adjacent experience to the role and clearly frame the gap?"
            )
        else:
            section = resume_chunk.get("section", "general") if resume_chunk else "general"
            question_text = (
                f"This is a commonly asked topic for a {role_title or 'target role'} interview: '{truncate_text(topic, 70)}'. "
                f"Using your {section} evidence '{truncate_text(evidence, 120)}', how would you answer this if asked at {company_name or 'the company'}?"
            )

        category = categorize_requirement(topic, role_mode, "interview pattern")
        why_asked = build_why_asked(topic, "interview pattern", match_label, category)
        prep_tip = build_prep_tip("interview pattern", match_label, evidence, role_mode, category)

        questions.append({
            "requirement": topic,
            "focus_area": truncate_text(topic, 70),
            "match_label": match_label,
            "score": round(score, 4) if score is not None else None,
            "question": question_text,
            "resume_evidence": evidence,
            "section": resume_chunk.get("section", "gap") if resume_chunk else "gap",
            "page_number": resume_chunk.get("page_number") if resume_chunk else None,
            "question_source": "interview pattern",
            "question_category": category,
            "why_asked": why_asked,
            "prep_tip": prep_tip
        })

    return questions


def generate_interview_prep(
    job_description: str,
    company_name: str = "",
    role_title: str = "",
    role_mode: str = "auto",
    company_context: str = "",
    interview_notes: str = ""
):
    if not document_store["chunks"] or document_store["faiss_index"] is None:
        refresh_document_store_from_disk()

    if not document_store["chunks"] or document_store["faiss_index"] is None:
        raise HTTPException(status_code=400, detail="Upload or load a resume before generating interview questions.")

    normalized_jd = job_description.strip()
    if not normalized_jd:
        raise HTTPException(status_code=400, detail="Paste a job description to generate interview questions.")
    if len(normalized_jd) > MAX_JOB_DESCRIPTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Job description is too long ({len(normalized_jd)} characters). "
                f"Please shorten it to {MAX_JOB_DESCRIPTION_CHARS} characters or less and try again."
            )
        )

    requirements = split_job_description_into_requirements(normalized_jd)
    if not requirements:
        raise HTTPException(
            status_code=400,
            detail="Could not identify enough usable requirements from the job description."
        )

    resolved_role_mode = detect_role_mode(role_title, normalized_jd, role_mode)

    auto_research_requested = bool(company_name.strip() or role_title.strip())

    with ThreadPoolExecutor(max_workers=3) as executor:
        live_research_future = executor.submit(
            run_company_research,
            company_name,
            role_title,
            normalized_jd,
            resolved_role_mode
        ) if auto_research_requested else None
        company_future = executor.submit(extract_context_signals, company_context, 4)
        interview_future = executor.submit(extract_context_signals, interview_notes, 4)
        jd_focus_future = executor.submit(lambda items: [get_requirement_focus(item) for item in items], requirements)

        pasted_company_signals = company_future.result()
        interview_signals = interview_future.result()
        requirement_focuses = jd_focus_future.result()
        live_research_payload = live_research_future.result() if live_research_future else {
            "query": None,
            "results": [],
            "status": ""
        }

    live_research_results = live_research_payload.get("results", [])
    research_query = live_research_payload.get("query")
    live_research_status = live_research_payload.get("status", "")

    company_results, interview_results = split_research_results(live_research_results)
    auto_company_signals = summarize_research_results(company_results)
    auto_interview_signals = summarize_research_results(interview_results)

    company_signals = auto_company_signals + pasted_company_signals
    if len(company_signals) > 4:
        company_signals = company_signals[:4]
    interview_signals = auto_interview_signals + interview_signals
    if len(interview_signals) > 4:
        interview_signals = interview_signals[:4]
    auto_research_used = bool(live_research_results)

    questions = []
    role_pattern_slots = min(2, len(extract_interview_topics(interview_signals, max_items=2)))
    jd_question_limit = max(1, MAX_INTERVIEW_QUESTIONS - role_pattern_slots)

    for requirement, focus_area in zip(requirements[:jd_question_limit], requirement_focuses[:jd_question_limit]):
        resume_chunk, score = find_best_resume_match_for_requirement(requirement)

        match_label = classify_requirement_match(requirement, score, resume_chunk)
        if match_label == "gap":
            resume_chunk = None

        resume_evidence = select_supporting_evidence_for_requirement(requirement, resume_chunk)
        question_text = build_question_text(requirement, resume_chunk, match_label, resume_evidence)
        category = categorize_requirement(requirement, resolved_role_mode, "jd + resume")
        why_asked = build_why_asked(requirement, "jd + resume", match_label, category)
        prep_tip = build_prep_tip("jd + resume", match_label, resume_evidence, resolved_role_mode, category)

        questions.append({
            "requirement": requirement,
            "focus_area": focus_area,
            "match_label": match_label,
            "score": round(score, 4) if score is not None else None,
            "question": question_text,
            "resume_evidence": resume_evidence,
            "section": resume_chunk.get("section", "general") if resume_chunk else "gap",
            "page_number": resume_chunk.get("page_number") if resume_chunk else None,
            "question_source": "jd + resume",
            "question_category": category,
            "why_asked": why_asked,
            "prep_tip": prep_tip
        })

    pattern_questions = build_interview_pattern_questions(
        interview_signals,
        company_name.strip(),
        role_title.strip(),
        resolved_role_mode
    )
    questions.extend(pattern_questions[:role_pattern_slots])
    questions = dedupe_questions(questions, limit=MAX_INTERVIEW_QUESTIONS)

    matched_requirements = sum(1 for item in questions if item["match_label"] in {"strong", "partial", "adjacent"})
    gap_requirements = sum(1 for item in questions if item["match_label"] == "gap")

    if gap_requirements == 0:
        summary = "All generated questions are supported by clear resume alignment or interview-pattern relevance."
    else:
        summary = (
            f"{matched_requirements} questions are aligned to your background or adjacent experience, and "
            f"{gap_requirements} highlight likely gaps you should prepare to address directly."
        )

    if company_name or role_title:
        scope_bits = [bit for bit in [company_name.strip(), role_title.strip()] if bit]
        summary = f"{' | '.join(scope_bits)}: {summary}"

    return {
        "status": "success",
        "company_name": company_name.strip() or None,
        "role_title": role_title.strip() or None,
        "role_mode": resolved_role_mode,
        "job_description": normalized_jd,
        "document_id": document_store["document_id"],
        "filename": document_store["filename"],
        "questions": questions,
        "matched_requirements": matched_requirements,
        "gap_requirements": gap_requirements,
        "summary": summary,
        "company_signals": company_signals,
        "interview_signals": interview_signals,
        "latency_summary": estimate_latency_summary(bool(company_signals), bool(interview_signals), auto_research_used),
        "auto_research_used": auto_research_used,
        "research_status": get_research_status(auto_research_used, len(live_research_results), live_research_status),
        "research_query": research_query,
        "question_metrics": build_question_metrics(questions),
        "question_groups": build_question_groups(questions),
        "role_mode_label": get_role_mode_label(resolved_role_mode)
    }


def generate_interview_questions(job_description: str):
    return generate_interview_prep(job_description=job_description)


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


@app.post("/generate-interview-questions", response_model=InterviewPrepResponse)
async def generate_interview_questions_api(
    job_description: str = Form(...),
    company_name: str = Form(""),
    role_title: str = Form(""),
    role_mode: str = Form("auto"),
    company_context: str = Form(""),
    interview_notes: str = Form("")
):
    result = generate_interview_prep(
        job_description=job_description,
        company_name=company_name,
        role_title=role_title,
        role_mode=role_mode,
        company_context=company_context,
        interview_notes=interview_notes
    )
    return {
        "status": result["status"],
        "company_name": result["company_name"],
        "role_title": result["role_title"],
        "role_mode": result["role_mode"],
        "job_description": result["job_description"],
        "document_id": result["document_id"],
        "filename": result["filename"],
        "questions": result["questions"],
        "matched_requirements": result["matched_requirements"],
        "gap_requirements": result["gap_requirements"],
        "company_signals": result["company_signals"],
        "interview_signals": result["interview_signals"],
        "latency_summary": result["latency_summary"],
        "auto_research_used": result["auto_research_used"],
        "research_status": result["research_status"],
        "research_query": result["research_query"]
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
async def generate_interview_questions_ui(
    request: Request,
    job_description: str = Form(...),
    company_name: str = Form(""),
    role_title: str = Form(""),
    role_mode: str = Form("auto"),
    company_context: str = Form(""),
    interview_notes: str = Form("")
):
    try:
        result = generate_interview_prep(
            job_description=job_description,
            company_name=company_name,
            role_title=role_title,
            role_mode=role_mode,
            company_context=company_context,
            interview_notes=interview_notes
        )
        return render_home(
            request,
            filename=result["filename"],
            num_chunks=len(document_store["chunks"]),
            job_description=result["job_description"],
            company_name=company_name,
            role_title=role_title,
            role_mode=result["role_mode"],
            company_context=company_context,
            interview_notes=interview_notes,
            interview_questions=result["questions"],
            interview_summary=result["summary"],
            company_signals=result["company_signals"],
            interview_signals=result["interview_signals"],
            latency_summary=result["latency_summary"],
            auto_research_used=result["auto_research_used"],
            research_status=result["research_status"],
            research_query=result["research_query"],
            question_metrics=result["question_metrics"],
            question_groups=result["question_groups"],
            role_mode_label=result["role_mode_label"],
            page_title=f"Interview prep ready for {result['filename']}"
        )
    except HTTPException as e:
        return render_home(
            request,
            job_description=job_description,
            company_name=company_name,
            role_title=role_title,
            role_mode=role_mode,
            company_context=company_context,
            interview_notes=interview_notes,
            best_match=e.detail
        )
    
 
