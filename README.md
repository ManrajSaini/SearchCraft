# SearchCraft

A modern search engine built from scratch with BM25 ranking, spell correction, and retrieval-augmented generation (RAG) powered by LLMs.

Open your browser to `http://127.0.0.1:8081/ui` after starting the server.

---

## Features

### 🔍 **Dual Search Modes**

- **Search Mode** — Fast BM25 keyword-based ranking with no API calls
- **Ask AI Mode** — BM25 retrieval + LLM synthesis delivering natural language answers with cited sources

### 🧠 **Intelligent Query Processing**

- **Tokenization** — Lowercase, punctuation removal, stopword filtering, stemming
- **Bloom Filter Gating** — Fast probabilistic vocabulary check before expensive spell correction
- **Spell Correction** — Levenshtein distance + document frequency tiebreaking
- **Live Correction Feedback** — Spell corrections displayed prominently in the UI

### 📊 **Production-Grade Ranking**

- **BM25 Scoring** — Okapi BM25 with IDF, term frequency saturation, and document-length normalization
- **Inverted Index** — Token → postings mapping with term/doc frequencies and positions
- **Cached Index** — Auto-generated from source documents, persisted to JSON

### 🤖 **RAG Pipeline**

- Retrieves top-5 documents by BM25 relevance
- Builds a grounded prompt with document context
- Calls Groq LLM (free tier via LangChain) for synthesis
- Returns AI answer + ranked sources with snippets

---

## Quick Start

### Prerequisites

- Python 3.11+
- A free Groq API key (get one at https://console.groq.com)

### Installation

```bash
# 1. Clone or download SearchCraft
cd SearchCraft

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### Running the Server

```bash
# Start the FastAPI server (auto-rebuilds index if missing)
uvicorn api:app --host 127.0.0.1 --port 8081 --reload

# Open browser to:
# http://127.0.0.1:8081/ui
```

### Using the CLI (Optional)

```bash
# Interactive terminal-based interface (no server needed)
# Loads from cache or builds index on first run
python search_cli.py

SearchCraft [search] > what is machine learning?
SearchCraft [search] > :mode ask
SearchCraft [ask] > :help
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│                    "pythn tutorial"                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   TOKENIZER        │
                │ lowercase, stem,   │
                │ remove stopwords   │
                └──────────┬─────────┘
                           │
                    ["pythn", "tutorial"]
                           │
                ┌──────────▼──────────────────┐
                │   BLOOM FILTER              │
                │ Check: "pythn" in vocab?   │
                │ "tutorial" in vocab?        │
                └──────────┬───────────────────┘
                           │
          ┌────────────────┴────────────────┐
  "pythn" │ NOT IN VOCAB              "tutorial" │ IN VOCAB
  (0.1%)  │  → Run Levenshtein        (99.9%)   │ → Skip
          │  → Suggest "python"                  │
          │                                      │
          └───────────────────┬───────────────────┘
                              │
                    "python tutorial"
                              │
                    ┌─────────▼────────────┐
                    │  INVERTED INDEX      │
                    │ Find all docs with   │
                    │ "python" OR "tutorial"
                    └─────────┬────────────┘
                              │
                    50 candidate documents
                              │
              ┌───────────────▼────────────────┐
              │ BM25 SCORING                   │
              │ Rank by relevance              │
              │ (IDF, tf saturation, length)   │
              └───────────────┬────────────────┘
                              │
      ┌───────────────────────┴──────────────────────┐
      │                                              │
   SEARCH MODE                                  ASK AI MODE
      │                                              │
      ├─► Top-10 ranked results                     │
      │   (BM25 scores only)                        │
      │   Returned in 10-50 ms                      │
      │                                              │
      └─ Show in results panel                      │
                                                     │
                                         ┌──────────▼──────────┐
                                         │ TOP-5 DOCS          │
                                         │ + PROMPT BUILDER    │
                                         └──────────┬──────────┘
                                                    │
                                         ┌──────────▼──────────┐
                                         │ LLM API CALL        │
                                         │ (Groq free tier)    │
                                         └──────────┬──────────┘
                                                    │
                                         ┌──────────▼──────────┐
                                         │ AI ANSWER + SOURCES │
                                         │ Returned in 1-3 sec │
                                         └──────────┬──────────┘
                                                    │
                                         Show in answer panel
```

---

## Core Data Structures & Algorithms

### Document Class

```python
@dataclass
class Document:
    doc_id: str              # Unique identifier (e.g., "python_basics")
    title: str               # Human-readable title
    content: str             # Full document text
    token_count: int = 0     # Calculated after tokenization
```

### InvertedIndex Class

Maps every token to the documents it appears in, with positional and frequency information.

**Internal Structure:**

```python
{
    "python": {
        "doc_freq": 3,           # Number of docs containing "python"
        "postings": {
            "machine_learning": {
                "term_freq": 5,       # Occurrences in this doc
                "positions": [0, 15, 42, ...]
            },
            "python_basics": {
                "term_freq": 8,
                "positions": [1, 3, 7, ...]
            }
        }
    },
    ...
}
```

**Key Methods:**

- `add_document(doc)` — Tokenize and index a document
- `get_postings(token)` — Returns all docs containing a token
- `get_doc_frequency(token)` — Count of docs with this token

### BM25 Score Formula

Okapi BM25 is the industry-standard ranking function for full-text search.

$$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

**Components:**

- **IDF (Inverse Document Frequency):**
  $$\text{IDF}(t) = \log\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$
  - `N` = total number of documents
  - `df(t)` = number of documents containing token `t`
  - Penalizes common terms, rewards rare terms

- **Numerator (term frequency component):**
  $$f(t,d) \cdot (k_1 + 1)$$
  - `f(t,d)` = frequency of token `t` in document `d`
  - `k_1` = saturation parameter (default 1.2)
  - More occurrences = higher score, but with diminishing returns

- **Denominator (length normalization):**
  $$f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)$$
  - `|d|` = length of document `d` in tokens
  - `avgdl` = average document length in corpus
  - `b` = length normalization parameter (default 0.75)
  - Penalizes artificially long documents

**Default Parameters (SearchCraft):**

- `k1 = 1.2` — Standard Okapi recommendation
- `b = 0.75` — Standard Okapi recommendation

### Bloom Filter

A probabilistic data structure for fast membership testing.

**False-Positive Rate Formula:**
$$p = \left(1 - e^{-kn/m}\right)^k$$

Where:

- `k` = number of hash functions
- `n` = number of items added
- `m` = size of bit array
- `p` = estimated false-positive rate

**Optimal Sizing (SearchCraft uses):**

Optimal bit array size:
$$m = -\frac{n \cdot \ln(p)}{(\ln 2)^2}$$

Optimal number of hash functions:
$$k = \frac{m}{n} \cdot \ln 2$$

**Example (375 tokens, 0.1% target FP rate):**

- `m` ≈ 7,125 bits (891 bytes)
- `k` ≈ 5 hash functions
- Actual FP rate ≈ 0.09%

**Why it matters:** Before running expensive Levenshtein distance computation for spell correction, we check the Bloom filter. If it returns false, the token is **definitely not in vocabulary** and needs correction. If true, it's **probably in vocabulary** (rare false positive), so we skip correction.

### Levenshtein Distance (Edit Distance)

Minimum number of single-character edits (insert, delete, replace) to transform one string to another.

**Dynamic Programming Matrix:**

For `a = "pythn"` and `b = "python"`:

```
        ""  p  y  t  h  o  n
    ""   0  1  2  3  4  5  6
    p    1  0  1  2  3  4  5
    y    2  1  0  1  2  3  4
    t    3  2  1  0  1  2  3
    h    4  3  2  1  0  1  2
    n    5  4  3  2  1  1  1
```

**Recurrence Relation:**

$$
dp[i][j] = \begin{cases}
i & \text{if } j = 0 \\
j & \text{if } i = 0 \\
dp[i-1][j-1] & \text{if } a[i-1] = b[j-1] \\
1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) & \text{otherwise}
\end{cases}
$$

- `dp[i-1][j] + 1` — delete from `a`
- `dp[i][j-1] + 1` — insert into `a`
- `dp[i-1][j-1] + 1` — replace

**Time Complexity:** O(m × n) where m, n = string lengths  
**Space Complexity:** O(m × n) [can be optimized to O(min(m, n))]

**SearchCraft Usage:**

- `max_distance = 2` — only suggest corrections within 2 edits
- Fast pre-filter by length: skip candidates differing by > 2 chars

---

## Configuration

### Environment Variables (`.env`)

```env
GROQ_API_KEY=gsk_your_api_key_here
```

Get a free key at https://console.groq.com

### Tuning Parameters

**In `api.py` (lifespan):**

- Bloom filter false-positive rate: `p=0.001` (0.1%)
- BM25 parameters: `k1=1.2`, `b=0.75` (standard Okapi defaults)

**In `searchcraft/rag.py`:**

- `TOP_K = 5` — documents to retrieve for RAG
- `DOC_SNIPPET_CHARS = 500` — truncation for prompt size

**In `searchcraft/spell_correct.py`:**

- `max_distance = 2` — max Levenshtein edits to consider

---

## Tech Stack

| Layer             | Technology                                 |
| ----------------- | ------------------------------------------ |
| **Backend**       | FastAPI (Python) 0.110+, Uvicorn 0.29+     |
| **Frontend**      | HTML5, CSS3, Vanilla JavaScript            |
| **LLM**           | Groq (free tier via LangChain)             |
| **Search Engine** | Built from scratch (no Elasticsearch/Solr) |
| **Persistence**   | JSON (inverted index cache)                |

---

**Happy searching!** 🔍
