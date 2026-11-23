"""
Shared helpers for the RAG (page 11) and Q&A (page 12) pages.

Designed to run both locally and on Streamlit Community Cloud:
- Review embeddings are PRE-COMPUTED once (scripts/build_rag_index.py) and loaded
  from disk -> no 15 000-review embedding pass at startup.
- Only the user query is embedded at runtime, with fastembed (ONNX, no torch).
- Generation goes through any OpenAI-compatible endpoint (Groq, Mistral, OpenRouter,
  or a local Ollama), configured with st.secrets / environment variables.
"""

import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "rag_index"
REVIEWS_PATH = INDEX_DIR / "reviews.parquet"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
TEST_REVIEWS_PATH = INDEX_DIR / "test_reviews.parquet"

# Must be the same model as the one used in scripts/build_rag_index.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =====================================================
# Data & index (loaded once per server process)
# =====================================================
def _check_index_files():
    missing = [p.name for p in (REVIEWS_PATH, EMBEDDINGS_PATH) if not p.exists()]
    if missing:
        st.error(
            f"Missing RAG index files in `data/rag_index/`: {', '.join(missing)}.\n\n"
            "Run `python scripts/build_rag_index.py` once locally and commit the generated files."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    _check_index_files()
    return pd.read_parquet(REVIEWS_PATH)


@st.cache_data(show_spinner=False)
def load_test_reviews() -> list[str]:
    if not TEST_REVIEWS_PATH.exists():
        return []
    return pd.read_parquet(TEST_REVIEWS_PATH)["avis_en"].tolist()


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss_index():
    _check_index_files()
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)  # stored as float16
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vectors = cosine
    index.add(embeddings)
    return index


@st.cache_resource(show_spinner="Loading embedding model...")
def _load_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBEDDING_MODEL)


@st.cache_data(show_spinner=False, max_entries=512)
def embed_query(text: str) -> np.ndarray:
    vec = np.array(list(_load_embedder().embed([text]))[0], dtype=np.float32)[None, :]
    faiss.normalize_L2(vec)
    return vec


def search(query: str, k: int = 5) -> list[int]:
    """Return the row positions of the k reviews closest to `query`."""
    index = load_faiss_index()
    _, indices = index.search(embed_query(query), k)
    return [int(i) for i in indices[0] if i >= 0]


# =====================================================
# LLM (OpenAI-compatible endpoint)
# =====================================================
def _setting(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml at all
        pass
    return os.environ.get(name, default)


def llm_config() -> dict:
    """
    LLM_API_KEY set  -> hosted provider (Groq by default).
    no LLM_API_KEY   -> local Ollama (http://localhost:11434/v1).
    """
    api_key = _setting("LLM_API_KEY")
    if api_key:
        return {
            "base_url": _setting("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            "api_key": api_key,
            "model": _setting("LLM_MODEL", "llama-3.1-8b-instant"),
        }
    return {
        "base_url": _setting("LLM_BASE_URL", "http://localhost:11434/v1"),
        "api_key": "ollama",
        "model": _setting("LLM_MODEL", "mistral"),
    }


@st.cache_resource(show_spinner=False)
def _llm_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=60)


def stream_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 600):
    """Yield the answer chunk by chunk (to be used with st.write_stream)."""
    cfg = llm_config()
    client = _llm_client(cfg["base_url"], cfg["api_key"])
    stream = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def llm_caption() -> str:
    cfg = llm_config()
    return f"LLM: `{cfg['model']}` via `{cfg['base_url']}`"
