import json
import os
from dataclasses import dataclass, field
from searchcraft.tokenizer import tokenize


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    token_count: int = field(default=0, init=False)  # set after tokenization


# ── Inverted Index ─────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    Maps every token to the documents it appears in, plus positions.

    Internal structure:
    {
        "python": {
            "doc_freq": 3,
            "postings": {
                "doc_001": {"term_freq": 2, "positions": [0, 15]},
                ...
            }
        },
        ...
    }
    """

    def __init__(self):
        self._index: dict = {}          # token -> {doc_freq, postings}
        self.doc_store: dict = {}       # doc_id -> Document

    # ── public API ────────────────────────────────────────────────────────────

    def add_document(self, doc: Document) -> None:
        """Tokenize doc.content, update the index and doc_store."""
        tokens = tokenize(doc.content)
        doc.token_count = len(tokens)

        # Save the document so we can retrieve it later
        self.doc_store[doc.doc_id] = doc

        # Walk through tokens keeping track of position
        for position, token in enumerate(tokens):

            # First time we've seen this token anywhere
            if token not in self._index:
                self._index[token] = {"doc_freq": 0, "postings": {}}

            postings = self._index[token]["postings"]

            # First time this token appears in THIS document
            if doc.doc_id not in postings:
                postings[doc.doc_id] = {"term_freq": 0, "positions": []}
                self._index[token]["doc_freq"] += 1

            # Record the occurrence
            postings[doc.doc_id]["term_freq"] += 1
            postings[doc.doc_id]["positions"].append(position)

    def get_postings(self, token: str) -> dict:
        """Return the full postings dict for a token, or {} if not found."""
        entry = self._index.get(token)
        if entry is None:
            return {}
        return entry["postings"]

    def get_doc_frequency(self, token: str) -> int:
        """Return how many documents contain this token."""
        entry = self._index.get(token)
        if entry is None:
            return 0
        return entry["doc_freq"]


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index(index: InvertedIndex, path: str) -> None:
    """
    Serialize the full index and doc_store to a JSON file.

    Saved structure:
    {
        "index": { ...token -> {doc_freq, postings}... },
        "doc_store": {
            "doc_001": {"doc_id": ..., "title": ..., "content": ..., "token_count": ...},
            ...
        }
    }
    """
    payload = {
        "index": index._index,
        "doc_store": {
            doc_id: {
                "doc_id":      doc.doc_id,
                "title":       doc.title,
                "content":     doc.content,
                "token_count": doc.token_count,
            }
            for doc_id, doc in index.doc_store.items()
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Index saved → {path}  "
          f"({os.path.getsize(path) / 1024:.1f} KB, "
          f"{len(index._index)} terms, "
          f"{len(index.doc_store)} docs)")


def load_index(path: str) -> InvertedIndex:
    """
    Deserialize a JSON file produced by save_index back into an InvertedIndex.
    Does NOT re-tokenize — the serialized _index is restored directly.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    idx = InvertedIndex()

    # Restore the raw index dict (token → {doc_freq, postings})
    idx._index = payload["index"]

    # Rebuild Document objects in the doc_store
    for doc_id, data in payload["doc_store"].items():
        doc = Document(
            doc_id=data["doc_id"],
            title=data["title"],
            content=data["content"],
        )
        doc.token_count = data["token_count"]
        idx.doc_store[doc_id] = doc

    print(f"Index loaded ← {path}  "
          f"({len(idx._index)} terms, "
          f"{len(idx.doc_store)} docs)")

    return idx
