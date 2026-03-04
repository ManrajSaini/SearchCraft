"""
SearchCraft — a search engine built from scratch.

Public API surface:
  Core data structures
    Document        — lightweight dataclass per indexed file
    InvertedIndex   — token → postings mapping

  Indexing
    tokenize        — raw text → clean stemmed token list
    load_documents  — load .txt/.md files into an InvertedIndex
    save_index      — persist an InvertedIndex to JSON
    load_index      — restore an InvertedIndex from JSON

  Retrieval
    BM25Scorer      — BM25 ranking over an InvertedIndex

  Spellcheck
    SpellCorrector  — Levenshtein-based correction ranked by corpus frequency
    BloomFilter     — probabilistic set membership (skip-correction gate)

  AI synthesis
    generate        — single LLM call via Groq
    rag_query       — full RAG pipeline: BM25 → prompt → LLM → answer
"""

from searchcraft.index import Document, InvertedIndex, save_index, load_index
from searchcraft.tokenizer import tokenize
from searchcraft.loader import load_documents
from searchcraft.scorer import BM25Scorer
from searchcraft.spell_correct import SpellCorrector
from searchcraft.bloom_filter import BloomFilter
from searchcraft.llm import generate
from searchcraft.rag import rag_query

__all__ = [
    "Document",
    "InvertedIndex",
    "save_index",
    "load_index",
    "tokenize",
    "load_documents",
    "BM25Scorer",
    "SpellCorrector",
    "BloomFilter",
    "generate",
    "rag_query",
]
