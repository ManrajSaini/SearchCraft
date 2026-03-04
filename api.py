"""
SearchCraft — FastAPI backend

Development:
    uvicorn api:app --reload --port 8081

Production (automatic on hosting platforms):
    Platforms like Render, Railway, Vercel auto-start with:
    uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("searchcraft")

from searchcraft.index import load_index, save_index, InvertedIndex
from searchcraft.loader import load_documents
from searchcraft.scorer import BM25Scorer
from searchcraft.spell_correct import SpellCorrector
from searchcraft.bloom_filter import BloomFilter
from searchcraft.tokenizer import tokenize
from searchcraft.rag import rag_query

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(ROOT, "docs")
INDEX_CACHE = os.path.join(ROOT, "data", "index_cache.json")

# ── Shared state (populated once at startup) ──────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy objects once; keep them alive for the entire process."""
    if os.path.exists(INDEX_CACHE):
        log.info("Loading index from cache: %s", INDEX_CACHE)
        idx = load_index(INDEX_CACHE)
    else:
        log.info("No cache found — building index from %s", DOCS_FOLDER)
        idx = InvertedIndex()
        n = load_documents(DOCS_FOLDER, idx)
        save_index(idx, INDEX_CACHE)
        log.info("Index built and cached — %d docs", n)

    n_docs  = len(idx.doc_store)
    n_terms = len(idx._index)
    log.info("Index ready — %d docs, %d terms", n_docs, n_terms)

    bm25      = BM25Scorer(idx)
    corrector = SpellCorrector(idx)

    # Build Bloom filter sized for current vocabulary at 0.1 % FP target
    _m = BloomFilter.optimal_size(n_terms, p=0.001)
    _k = BloomFilter.optimal_hashes(_m, n_terms)
    bloom_f = BloomFilter(size=_m, num_hashes=_k)
    for token in idx._index:
        bloom_f.add(token)
    log.info("Bloom filter built — FP rate: %.4f%%", bloom_f.false_positive_rate() * 100)

    state["idx"]        = idx
    state["bm25"]       = bm25
    state["corrector"]  = corrector
    state["bloom"]      = bloom_f
    state["n_docs"]     = n_docs
    state["n_terms"]    = n_terms
    state["started_at"] = datetime.now(timezone.utc).isoformat()

    log.info("Startup complete — ready to serve requests.")
    yield   # ← server is running

    log.info("Shutting down gracefully.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SearchCraft API",
    version="0.4.0",
    description="BM25 + RAG search engine built from scratch",
    lifespan=lifespan,
)

# CORS — allow all origins for local development.
# In production replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)


# ── Global error handlers ─────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a clean 422 with a plain message instead of Pydantic's nested detail."""
    errors = exc.errors()
    readable = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    log.warning("Validation error on %s: %s", request.url.path, readable)
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request", "detail": readable},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all — log the full traceback server-side, return a safe 500."""
    log.error(
        "Unhandled exception on %s:\n%s",
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# Serve the static frontend at /  (must be mounted AFTER routes are registered)
_frontend = os.path.join(ROOT, "frontend")
if os.path.isdir(_frontend):
    app.mount("/ui", StaticFiles(directory=_frontend, html=True), name="frontend")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {
        "status":     "ok",
        "docs":       state.get("n_docs",  0),
        "terms":      state.get("n_terms", 0),
        "started_at": state.get("started_at", None),
    }


# ── Documents ─────────────────────────────────────────────────────────────────
@app.get("/documents", tags=["search"])
def list_documents():
    """Return every indexed document ID, sorted alphabetically."""
    idx: object = state["idx"]
    doc_ids = sorted(idx.doc_store.keys())
    return {"documents": doc_ids}


@app.get("/documents/{doc_id}", tags=["search"])
def get_document(doc_id: str):
    """Return the raw text content of a single document."""
    docs_dir = os.path.join(ROOT, "docs")
    safe_id  = os.path.basename(doc_id)          # prevent path traversal
    path     = os.path.join(docs_dir, f"{safe_id}.txt")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"doc_id": doc_id, "content": content}


# ── Pydantic models ───────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class CorrectionInfo(BaseModel):
    original_query: str
    corrected_query: str
    changes: list[dict]   # [{"original": "pythn", "corrected": "python"}, …]


class SearchResult(BaseModel):
    rank:   int
    doc_id: str
    score:  float


class SearchResponse(BaseModel):
    results:    list[SearchResult]
    correction: Optional[CorrectionInfo]


# ── Spell-correction helper ───────────────────────────────────────────────────
def _apply_spell_correction(raw_query: str) -> tuple[str, Optional[CorrectionInfo]]:
    """
    Tokenize the query, run Bloom-gated spell correction on unknown tokens,
    and return (corrected_query_string, CorrectionInfo | None).
    """
    bloom:     BloomFilter     = state["bloom"]
    corrector: SpellCorrector  = state["corrector"]

    tokens = tokenize(raw_query)
    changes = []

    corrected_tokens = []
    for token in tokens:
        if not bloom.might_contain(token):
            # Definitely not in vocabulary — attempt correction
            suggestion = corrector.correct(token)
            if suggestion and suggestion != token:
                changes.append({"original": token, "corrected": suggestion})
                corrected_tokens.append(suggestion)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    corrected_query = " ".join(corrected_tokens)

    correction_info = None
    if changes:
        correction_info = CorrectionInfo(
            original_query=raw_query,
            corrected_query=corrected_query,
            changes=changes,
        )

    return corrected_query, correction_info


# ── POST /search ──────────────────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest):
    """
    BM25 search with Bloom-gated spell correction.

    Pipeline: tokenize → Bloom gate → Levenshtein correction → BM25 ranking
    """
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    try:
        corrected_query, correction_info = _apply_spell_correction(req.query)

        bm25: BM25Scorer = state["bm25"]
        hits = bm25.search(corrected_query, top_k=req.top_k)

        results = [
            SearchResult(rank=i + 1, doc_id=doc_id, score=round(score, 4))
            for i, (doc_id, score) in enumerate(hits)
        ]

        return SearchResponse(
            results=results,
            correction=correction_info,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── POST /ask ──────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceReference(BaseModel):
    doc_id:  str
    score:   float
    snippet: str


class AskResponse(BaseModel):
    answer:     str
    sources:    list[SourceReference]
    correction: Optional[CorrectionInfo]


@app.post("/ask", response_model=AskResponse, tags=["ask"])
def ask(req: AskRequest):
    """
    RAG query: spell-correct → BM25 → build prompt → LLM → answer + sources.
    """
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    try:
        corrected_query, correction_info = _apply_spell_correction(req.query)

        idx  = state["idx"]
        bm25 = state["bm25"]

        raw = rag_query(corrected_query, idx, bm25, top_k=req.top_k)

        sources = [
            SourceReference(
                doc_id=s["doc_id"],
                score=round(s["score"], 4),
                snippet=s["snippet"],
            )
            for s in raw["sources"]
        ]

        return AskResponse(
            answer=raw["answer"],
            sources=sources,
            correction=correction_info,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
