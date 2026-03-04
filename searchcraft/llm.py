import os
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MODEL_NAME = "llama-3.1-8b-instant"
MAX_TOKENS = 300
TIMEOUT_SECONDS = 10
FALLBACK_MESSAGE = "[AI summary unavailable — showing search results only]"


def generate(prompt: str, context: str) -> str:
    """
    Call the Groq LLM with a question and a retrieved context string.

    prompt  — the user's question, e.g. "What is Python?"
    context — text retrieved from the search engine to ground the answer

    Returns the model's response, or FALLBACK_MESSAGE if the call fails.
    """
    if not GROQ_API_KEY:
        return (
            "[AI summary unavailable — GROQ_API_KEY not set.\n"
            " Add it to your .env file:  GROQ_API_KEY=gsk_...]"
        )

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
            timeout=TIMEOUT_SECONDS,
        )

        messages = [
            SystemMessage(content=(
                "You are a precise search assistant. "
                "Answer the user's question using ONLY the provided documents. "
                "Cite which source(s) support your answer. "
                "If the documents don't contain enough information, say so honestly."
            )),
            HumanMessage(content=(
                f"Documents:\n{context}\n\n"
                f"Question: {prompt}"
            )),
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except httpx.TimeoutException:
        print(f"[llm] Request timed out after {TIMEOUT_SECONDS}s.")
        return FALLBACK_MESSAGE

    except httpx.HTTPStatusError as e:
        print(f"[llm] HTTP error {e.response.status_code}: {e.response.text[:120]}")
        return FALLBACK_MESSAGE

    except Exception as e:
        # Catch-all: auth errors, model errors, network failures, etc.
        print(f"[llm] Unexpected error ({type(e).__name__}): {e}")
        return FALLBACK_MESSAGE
