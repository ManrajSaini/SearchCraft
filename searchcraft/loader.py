import os
from searchcraft.index import Document, InvertedIndex


def load_documents(folder_path: str, index: InvertedIndex) -> int:
    """
    Read all .txt and .md files from folder_path, build Document objects,
    and feed them into the provided InvertedIndex.

    Returns the number of documents successfully indexed.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    count = 0

    for filename in sorted(os.listdir(folder_path)):
        if not (filename.endswith(".txt") or filename.endswith(".md")):
            continue  # skip anything that isn't a text/markdown file

        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        if not content:
            continue  # skip empty files

        # Use the filename (without extension) as the doc_id
        doc_id = os.path.splitext(filename)[0]

        # Use the doc_id as a human-readable title (underscores → spaces)
        title = doc_id.replace("_", " ").replace("-", " ").title()

        doc = Document(doc_id=doc_id, title=title, content=content)
        index.add_document(doc)
        count += 1

    return count
