import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "as", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "what",
    "which", "who", "how", "all", "each", "up", "out", "about", "into",
    "than", "then", "their", "there", "also"
}

VOWELS = frozenset("aeiou")

# Consonants that almost never legitimately end an English word
# their presence at the root tail strongly implies a stripped silent 'e'.
_NEEDS_E = frozenset("cgsvz")


def _dedouble(root: str) -> str:
    """Remove a doubled final consonant left by suffix attachment.

    'runn' -> 'run'  (from 'running')
    'stopp' -> 'stop' (from 'stopped')
    Skips genuine double-s roots: 'pass', 'bless' stay untouched.
    """
    if (
        len(root) > 2
        and root[-1] == root[-2]
        and root[-1] not in VOWELS
        and root[-1] != "s"          # 'pass', 'bless' are real double-s words
    ):
        return root[:-1]
    return root


def _restore_e(root: str) -> str:
    """Re-attach the silent 'e' when the root tail consonant demands it.

    'clos' -> 'close',  'danc' -> 'dance',  'lov' -> 'love'
    """
    if root and root[-1] in _NEEDS_E:
        return root + "e"
    return root


def stem(word: str) -> str:
    """Simple suffix stripper"""
    # -ing: 'running'->'run', 'jumping'->'jump'
    if len(word) > 5 and word.endswith("ing"):
        return _dedouble(word[:-3])

    # -ily: 'happily'->'happy', 'lazily'->'lazy'
    if len(word) > 5 and word.endswith("ily"):
        return word[:-3] + "y"

    # -ly: 'quickly'->'quick', 'loudly'->'loud'
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]

    # -ed: 'closed'->'close', 'stopped'->'stop', 'walked'->'walk'
    if len(word) > 4 and word.endswith("ed"):
        root = word[:-2]
        deduped = _dedouble(root)
        if deduped != root:            # doubled consonant was the culprit
            return deduped
        return _restore_e(root)        # may restore silent 'e'

    # -er: 'faster'->'fast'
    if len(word) > 4 and word.endswith("er"):
        return word[:-2]

    # -est: 'fastest'->'fast'
    if len(word) > 4 and word.endswith("est"):
        return word[:-3]

    # -s: 'dogs'->'dog'  (skip genuine double-s endings)
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def tokenize(text: str) -> list[str]:
    """
    Full pipeline:
      1. Lowercase
      2. Strip punctuation
      3. Remove stopwords
      4. Stem
    """
    # 1. Lowercase
    text = text.lower()

    # 2a. Replace hyphens with spaces so 'widely-used' -> 'widely used'
    text = text.replace("-", " ")

    # 2b. Strip remaining punctuation — keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    # 3. Split into words
    words = text.split()

    tokens = []
    for w in words:
        if w not in STOPWORDS:
            stemmed = stem(w)
            tokens.append(stemmed)

    return tokens

