
const API = "http://127.0.0.1:8081";

// ── DOM refs ──────────────────────────────────────────────────────
const input = document.getElementById("query-input");
const btnSearch = document.getElementById("btn-search");
const btnAsk = document.getElementById("btn-ask");
const corrBanner = document.getElementById("correction-banner");
const corrText = document.getElementById("correction-text");
const placeholder = document.getElementById("results-placeholder");
const resultList = document.getElementById("result-list");
const askResult = document.getElementById("ask-result");
const askAnswer = document.getElementById("ask-answer");
const sourceList = document.getElementById("source-list");
const docList = document.getElementById("doc-list");

// ── Helpers ───────────────────────────────────────────────────────

/** "python_basics" → "Python Basics" */
function fmtDocId(id) {
  return id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Lock / unlock both buttons while a request is in flight */
function setLoading(activeBtn, loading) {
  const label = activeBtn === btnSearch ? "Search" : "Ask AI";
  btnSearch.disabled = loading;
  btnAsk.disabled = loading;
  activeBtn.textContent = loading ? "Searching…" : label;
}

/** Show / hide the spell-correction banner */
function showCorrection(correction) {
  if (!correction || !correction.changes || correction.changes.length === 0) {
    corrBanner.hidden = true;
    return;
  }
  const parts = correction.changes
    .map((c) => `"${c.original}" → "${c.corrected}"`)
    .join(", ");
  corrText.textContent = `Spell-corrected: ${parts}. Showing results for "${correction.corrected_query}".`;
  corrBanner.hidden = false;
}

/** Reset the results area to a clean slate before every query */
function clearResults() {
  corrBanner.hidden = true;
  placeholder.hidden = true;
  resultList.hidden = true;
  askResult.hidden = true;
  resultList.innerHTML = "";
  sourceList.innerHTML = "";
  askAnswer.textContent = "";
}

/** Show a plain error message in the results area */
function showError(msg) {
  placeholder.textContent = `⚠ ${msg}`;
  placeholder.hidden = false;
}

/** Restore the idle placeholder */
function showPlaceholder() {
  placeholder.textContent = "Enter a query above to see results.";
  placeholder.hidden = false;
}

// Sidebar: populate document list (expandable) ────────────
async function loadDocuments() {
  try {
    const res = await fetch(`${API}/documents`);
    const data = await res.json();
    docList.innerHTML = "";
    if (!data.documents || data.documents.length === 0) {
      docList.innerHTML =
        '<li class="doc-item doc-placeholder">No documents found.</li>';
      return;
    }
    for (const id of data.documents) {
      const li = document.createElement("li");
      li.className = "doc-item";

      const header = document.createElement("div");
      header.className = "doc-item-header";
      header.innerHTML = `
        <span class="doc-chevron">&#9654;</span>
        <span class="doc-item-name">${fmtDocId(id)}</span>
      `;

      const body = document.createElement("div");
      body.className = "doc-item-body";

      li.appendChild(header);
      li.appendChild(body);
      docList.appendChild(li);

      // Toggle open/close; lazy-load content on first expand
      header.addEventListener("click", async () => {
        const isOpen = li.classList.toggle("open");
        if (!isOpen) return; // collapsed — nothing to load
        if (body.dataset.loaded) return; // content already fetched

        body.innerHTML = '<span class="doc-item-loading">Loading…</span>';
        try {
          const r = await fetch(`${API}/documents/${encodeURIComponent(id)}`);
          const d = await r.json();
          if (!r.ok) throw new Error(d.detail || "Failed to load content.");
          body.textContent = d.content.trim();
          body.dataset.loaded = "1";
        } catch (e) {
          body.textContent = "⚠ Could not load content.";
        }
      });
    }
  } catch (err) {
    docList.innerHTML =
      '<li class="doc-item doc-placeholder">Could not load documents.</li>';
  }
}

// Core: render search results ────────────────────────────
function renderSearchResults(results) {
  if (!results || results.length === 0) {
    showError("No results found for this query.");
    return;
  }
  resultList.innerHTML = "";
  for (const r of results) {
    const li = document.createElement("li");
    li.className = "result-card";
    li.innerHTML = `
      <span class="result-rank">#${r.rank}</span>
      <span class="result-name">${fmtDocId(r.doc_id)}</span>
      <span class="result-score">${r.score.toFixed(3)}</span>
    `;
    resultList.appendChild(li);
  }
  resultList.hidden = false;
}

// Core: render Ask AI answer + sources ───────────────────
function renderAskResult(answer, sources) {
  // Answer block
  askAnswer.textContent = answer;

  // Sources list
  sourceList.innerHTML = "";
  if (sources && sources.length > 0) {
    for (let i = 0; i < sources.length; i++) {
      const s = sources[i];
      const li = document.createElement("li");
      li.className = "source-card";
      li.innerHTML = `
        <span class="source-rank">${i + 1}.</span>
        <span class="source-name">${fmtDocId(s.doc_id)}</span>
        <span class="source-score">${s.score.toFixed(3)}</span>
      `;
      sourceList.appendChild(li);
    }
  }
  askResult.hidden = false;
}

// Core: POST /search ─────────────────────────────────────
async function runSearch() {
  const query = input.value.trim();
  if (!query) {
    input.focus();
    return;
  }

  clearResults();
  setLoading(btnSearch, true);

  try {
    const res = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 10 }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || data.error || `Server error (${res.status})`);
      return;
    }

    showCorrection(data.correction);
    renderSearchResults(data.results);
  } catch (err) {
    showError("Could not reach the server. Is it running?");
  } finally {
    setLoading(btnSearch, false);
  }
}

// Core: POST /ask ────────────────────────────────────────
async function runAsk() {
  const query = input.value.trim();
  if (!query) {
    input.focus();
    return;
  }

  clearResults();
  setLoading(btnAsk, true);

  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || data.error || `Server error (${res.status})`);
      return;
    }

    showCorrection(data.correction);
    renderAskResult(data.answer, data.sources);
  } catch (err) {
    showError("Could not reach the server. Is it running?");
  } finally {
    setLoading(btnAsk, false);
  }
}

// Event wiring ──────────────────────────────────────────────────
btnSearch.addEventListener("click", runSearch);
btnAsk.addEventListener("click", runAsk);


input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    runSearch();
  }
  if (e.key === "Enter" && e.shiftKey) {
    e.preventDefault();
    runAsk();
  }
});

// Boot ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadDocuments();
  showPlaceholder();
  input.focus();
});
