import os
from searchcraft.index import InvertedIndex, load_index, save_index
from searchcraft.loader import load_documents
from searchcraft.scorer import BM25Scorer
from searchcraft.rag import rag_query
from searchcraft.spell_correct import SpellCorrector
from searchcraft.bloom_filter import BloomFilter
from searchcraft.tokenizer import tokenize

DOCS_FOLDER  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
INDEX_CACHE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index_cache.json")
SNIPPET_LEN  = 150
TOP_K        = 10

MODE_SEARCH = "search"   # BM25 only — zero API calls
MODE_ASK    = "ask"      # BM25 + LLM — exactly one API call per query


def build_or_load_index() -> InvertedIndex:
    """Load from cache if it exists, otherwise index from scratch and save."""
    if os.path.exists(INDEX_CACHE):
        idx = load_index(INDEX_CACHE)
    else:
        print("No cache found — indexing documents...")
        idx = InvertedIndex()
        n = load_documents(DOCS_FOLDER, idx)
        print(f"Indexed {n} documents.")
        save_index(idx, INDEX_CACHE)
    return idx


def make_snippet(content: str, length: int = SNIPPET_LEN) -> str:
    """Return the first `length` characters, collapsing whitespace."""
    snippet = " ".join(content.split())   # collapse newlines / extra spaces
    if len(snippet) <= length:
        return snippet
    # trim to the last full word before the limit
    trimmed = snippet[:length]
    last_space = trimmed.rfind(" ")
    if last_space > 0:
        trimmed = trimmed[:last_space]
    return trimmed + "..."


def run_cli(idx: InvertedIndex) -> None:
    bm25      = BM25Scorer(idx)
    corrector = SpellCorrector(idx)
    mode      = MODE_SEARCH   # default mode

    # Build Bloom Filter from vocabulary — fast gatekeeper before spell correction
    vocab     = list(idx._index.keys())
    n         = len(vocab)
    bf_size   = BloomFilter.optimal_size(n, p=0.001)
    bf_hashes = BloomFilter.optimal_hashes(bf_size, n)
    bloom     = BloomFilter(size=bf_size, num_hashes=bf_hashes)
    for token in vocab:
        bloom.add(token)
    print(f"  Bloom filter ready: {n} terms, FP rate ≈ {bloom.false_positive_rate():.4%}")

    print()
    print("╔══════════════════════════════════════╗")
    print("║          SearchCraft  v0.4           ║")
    print("║  :mode search | :mode ask | :help    ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Current mode: {mode}")
    print()

    while True:
        try:
            raw = input(f"SearchCraft [{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        # ── built-in commands ───────────────────────────────────────────────
        if raw.startswith(":"):
            cmd = raw.lower()

            if cmd in (":quit", ":exit", ":q"):
                print("Bye!")
                break

            elif cmd.startswith(":mode"):
                parts = cmd.split()
                if len(parts) < 2 or parts[1] not in (MODE_SEARCH, MODE_ASK):
                    print(f"  Usage: :mode search  |  :mode ask")
                else:
                    mode = parts[1]
                    label = (
                        "Keyword search — BM25 only (no API calls)"
                        if mode == MODE_SEARCH
                        else "AI-powered Q&A — BM25 + LLM (one API call per query)"
                    )
                    print(f"  Mode: {label}")

            elif cmd == ":help":
                print("  Type any query and press Enter.")
                print("  :mode search  — BM25 keyword results only")
                print("  :mode ask     — BM25 retrieval + LLM synthesis")
                print("  :stats        — corpus statistics")
                print("  :quit         — exit")

            elif cmd == ":stats":
                total_tokens = sum(d.token_count for d in idx.doc_store.values())
                print(f"  Documents    : {len(idx.doc_store)}")
                print(f"  Unique terms : {len(idx._index)}")
                print(f"  Total tokens : {total_tokens}")
                print(f"  Avg doc len  : {total_tokens / len(idx.doc_store):.1f} tokens")

            else:
                print(f"  Unknown command '{raw}'. Type :help for options.")

            continue

        # ── Bloom Filter + Spell correction ────────────────────────────────
        # Bloom filter gates which tokens need Levenshtein scanning:
        #   might_contain=True  → skip correction, go straight to BM25
        #   might_contain=False → DEFINITELY absent, run Levenshtein
        tokens      = tokenize(raw)
        fixed_parts = []
        final_tokens = []

        for token in tokens:
            if bloom.might_contain(token):
                # probably valid — skip the expensive Levenshtein scan
                final_tokens.append(token)
            else:
                # definitely not in vocab — run spell correction
                suggestion = corrector.correct(token)
                if suggestion:
                    fixed_parts.append(f"{token} → \"{suggestion}\"")
                    final_tokens.append(suggestion)
                # if no suggestion, drop the token silently

        if fixed_parts:
            print(f"  💡 Did you mean: {', '.join(fixed_parts)}?")
            raw = " ".join(final_tokens)
            print(f"  Searching for: \"{raw}\"")

        # ── MODE: search (BM25 only) ─────────────────────────────────────────
        if mode == MODE_SEARCH:
            results = bm25.search(raw, top_k=TOP_K)

            if not results:
                print(f"  No results found for '{raw}'.\n")
                continue

            print(f"\n🔍 Found {len(results)} result(s):\n")
            for rank, (doc_id, score) in enumerate(results, start=1):
                doc = idx.doc_store[doc_id]
                snippet = make_snippet(doc.content)
                print(f"  {rank}. [{score:.3f}]  {doc_id}")
                print(f"     \"{snippet}\"")
                print()

        # ── MODE: ask (BM25 + LLM) ───────────────────────────────────────────
        elif mode == MODE_ASK:
            result = rag_query(raw, idx, bm25, top_k=5)

            print(f"\n🔍 Retrieved {len(result['sources'])} documents:\n")
            for i, src in enumerate(result["sources"], 1):
                print(f"  {i}. [{src['score']:.3f}]  {src['doc_id']}")
            print()

            print("🤖 AI Answer:\n")
            # Indent every line of the answer for readability
            for line in result["answer"].splitlines():
                print(f"  {line}")
            print()


def main() -> None:
    idx = build_or_load_index()
    run_cli(idx)


if __name__ == "__main__":
    main()
