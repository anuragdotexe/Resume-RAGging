from fastapi import FastAPI, UploadFile, File, Form
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

# temporary in-memory storage
document_store = {
    "chunks": [],
    "embeddings": None,
    "filename": None
}

@app.get("/")
def home():
    return {"status": "running"}

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# def chunk_text(text, chunk_size=500, overlap=100):
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start = end - overlap

#     return chunks


def chunk_text(text, chunk_size=500, overlap=100, min_chunk_size=200):
    chunks = []
    start = 0
    text_length = len(text)

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

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks






def get_local_embeddings(text_list):
    embeddings = model.encode(text_list)
    return embeddings

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF allowed"}

    file_bytes = await file.read()
    extracted_text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(extracted_text)
    embeddings = get_local_embeddings(chunks)

    document_store["chunks"] = chunks
    document_store["embeddings"] = embeddings
    document_store["filename"] = file.filename

    return {
        "filename": file.filename,
        "text_length": len(extracted_text),
        "num_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape)
    }



# @app.post("/ask")
# async def ask_question(question: str = Form(...)):
#     if not document_store["chunks"] or document_store["embeddings"] is None:
#         return {"error": "No document uploaded yet"}

#     question_embedding = get_local_embeddings([question])
#     similarities = cosine_similarity(question_embedding, document_store["embeddings"])[0]

#     top_indices = np.argsort(similarities)[::-1][:3]

#     results = []
#     for idx in top_indices:
#         results.append({
#             "chunk_index": int(idx),
#             "score": float(similarities[idx]),
#             "chunk_text": document_store["chunks"][idx]
#         })

#     return {
#         "question": question,
#         "document": document_store["filename"],
#         "top_matches": results
#     }






@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not document_store["chunks"] or document_store["embeddings"] is None:
        return {"error": "No document uploaded yet"}

    question_embedding = get_local_embeddings([question])
    similarities = cosine_similarity(question_embedding, document_store["embeddings"])[0]

    top_indices = np.argsort(similarities)[::-1][:3]

    top_matches = []
    for rank, idx in enumerate(top_indices, start=1):
        top_matches.append({
            "rank": rank,
            "chunk_index": int(idx),
            "score": round(float(similarities[idx]), 4),
            "chunk_text": document_store["chunks"][idx]
        })

    best_match = top_matches[0]["chunk_text"] if top_matches else ""

    return {
        "question": question,
        "document": document_store["filename"],
        "best_match": best_match,
        "top_matches": top_matches
    }