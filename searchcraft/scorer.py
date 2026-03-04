import math
from searchcraft.tokenizer import tokenize
from searchcraft.index import InvertedIndex


class BM25Scorer:
    """
    BM25 ranking function.

    score(q, d) = Σ IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * (|d| / avgdl)))

    Parameters:
        tf(t,d)  = term frequency of token t in document d
        k1  — controls term-frequency saturation  (default 1.2)
        b   — controls document-length normalization (default 0.75)
        |d|      = length of document d (in tokens)
        avgdl    = average document length across corpus
        IDF(t)   = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        N        = total number of documents
        df(t)    = number of documents containing token t
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        self._index = index
        self.k1 = k1
        self.b = b

        # Corpus stats — computed once at init time
        self.N = len(index.doc_store)
        self.avgdl = self._compute_avgdl()

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_avgdl(self) -> float:
        """Average document length (in tokens) across the whole corpus."""
        if self.N == 0:
            return 0.0
        total = sum(doc.token_count for doc in self._index.doc_store.values())
        return total / self.N

    def _idf(self, token: str) -> float:
        """
        Inverse document frequency for a token.

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        The '+ 1' inside the log prevents negative IDF values for very
        common terms that appear in more than half the corpus.
        """
        df = self._index.get_doc_frequency(token)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    # ── public API ────────────────────────────────────────────────────────────

    def search(self, query_string: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Score ALL documents that match any query token, return the top-k
        ranked by BM25 score (descending).

        Returns a list of (doc_id, score) tuples.
        """
        query_tokens = tokenize(query_string)
        if not query_tokens:
            return []

        # 1. Collect the union of all doc_ids that match ANY query token
        candidate_docs: set[str] = set()
        for token in query_tokens:
            postings = self._index.get_postings(token)
            candidate_docs.update(postings.keys())

        # 2. Score every candidate
        scored: list[tuple[str, float]] = []
        for doc_id in candidate_docs:
            doc = self._index.doc_store.get(doc_id)
            if doc is None:
                continue
            doc_len = doc.token_count

            doc_score = 0.0
            for token in query_tokens:
                postings = self._index.get_postings(token)
                if doc_id not in postings:
                    continue
                tf = postings[doc_id]["term_freq"]
                idf = self._idf(token)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                doc_score += idf * (numerator / denominator)

            scored.append((doc_id, doc_score))

        # 3. Sort by score descending, return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
