from searchcraft.index import InvertedIndex
from searchcraft.scorer import BM25Scorer
from searchcraft.llm import generate

# How many top documents to feed into the LLM
TOP_K = 5

# Max characters per document in the prompt — keeps us within free-tier token limits
DOC_SNIPPET_CHARS = 500


def build_prompt(docs: list[tuple[str, float, str]]) -> str:
    """
    Build the document context block to pass to the LLM.

    docs — list of (doc_id, score, content) tuples, already truncated
    Returns a plain string of numbered document blocks — no instructions,
    no question. The LLM layer owns those.
    """
    return "\n\n".join(
        f"[{i+1}] (source: {doc_id}, score: {score:.3f})\n{content}"
        for i, (doc_id, score, content) in enumerate(docs)
    )


def rag_query(
    query: str,
    idx: InvertedIndex,
    bm25: BM25Scorer,
    top_k: int = TOP_K,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline for a single query.

    Returns a dict with:
        answer   — LLM synthesized answer (or fallback message)
        sources  — list of {doc_id, score, snippet} for the retrieved docs
    """
    # ── Step 1: BM25 search ───────────────────────────────────────────────────
    results = bm25.search(query, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant documents found for this query.",
            "sources": [],
        }

    # ── Step 2: Collect and truncate document content ────────────────────────
    retrieved = []
    for doc_id, score in results:
        doc = idx.doc_store[doc_id]
        # Collapse whitespace then truncate to DOC_SNIPPET_CHARS
        content = " ".join(doc.content.split())
        if len(content) > DOC_SNIPPET_CHARS:
            content = content[:DOC_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
        retrieved.append((doc_id, score, content))

    # ── Step 3: Build the document context block ─────────────────────────────
    context = build_prompt(retrieved)

    if verbose:
        print("\n── Document context sent to LLM ─────────────────────────")
        print(context)
        print("────────────────────────────────────────────────────────\n")

    # ── Step 4: Call the LLM ─────────────────────────────────────────────────
    answer = generate(prompt=query, context=context)

    # ── Step 5: Package sources for display ──────────────────────────────────
    sources = [
        {"doc_id": doc_id, "score": score, "snippet": content}
        for doc_id, score, content in retrieved
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


