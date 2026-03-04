from searchcraft.index import InvertedIndex


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings using
    dynamic programming.

    Build a 2D matrix where:
      - rows  represent characters of `a`  (length m)
      - cols  represent characters of `b`  (length n)
      - dp[i][j] = min edits to turn a[:i] into b[:j]

    Three operations, each costs 1:
      - Insert    dp[i][j-1] + 1
      - Delete    dp[i-1][j] + 1
      - Replace   dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else 1)

    Examples:
      levenshtein("cat",  "car")  → 1  (replace t→r)
      levenshtein("pythn", "python") → 1  (insert o)
      levenshtein("indexng", "indexing") → 1  (insert i)
    """
    m, n = len(a), len(b)

    # dp is (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    return dp[m][n]


class SpellCorrector:
    """
    Suggests the closest vocabulary word for an unknown query token.

    Tie-breaking: when multiple candidates share the same edit distance,
    we prefer the word that appears in MORE documents (higher doc_freq).

    O(vocab_size × word_length²) per correction
    """

    def __init__(self, index: InvertedIndex):
        self.vocabulary: list[str] = list(index._index.keys())
        # Pre-build doc_freq map for fast tie-breaking
        self.doc_freq: dict[str, int] = {
            token: data["doc_freq"]
            for token, data in index._index.items()
        }

    def correct(self, word: str, max_distance: int = 2) -> str | None:
        """
        Return the best vocabulary match for `word`, or None if nothing
        is within `max_distance` edits.
        """
        candidates = []

        for vocab_word in self.vocabulary:
            # Fast pre-filter: skip words whose length differs by more than
            # max_distance — they can't possibly be within the edit budget.
            if abs(len(vocab_word) - len(word)) > max_distance:
                continue

            dist = levenshtein_distance(word, vocab_word)
            if dist <= max_distance:
                candidates.append((vocab_word, dist))

        if not candidates:
            return None

        # Primary sort: smallest edit distance
        # Tie-break: highest document frequency (more common = better guess)
        candidates.sort(key=lambda x: (x[1], -self.doc_freq.get(x[0], 0)))
        return candidates[0][0]

    def corrections_for_query(
        self, query_tokens: list[str], index: InvertedIndex
    ) -> dict[str, str | None]:
        """
        For each token that has ZERO postings, find a correction.
        Tokens that exist in the index are left as-is (None correction).

        Returns {original_token: corrected_token_or_None}
        """
        result = {}
        for token in query_tokens:
            if token not in index._index:
                result[token] = self.correct(token)
        return result
