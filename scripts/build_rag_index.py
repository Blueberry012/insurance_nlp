"""
Pre-compute the review embeddings used by pages 11 (RAG) and 12 (Q&A).

Run ONCE locally (from the project root), then commit data/rag_index/:
    pip install fastembed pyarrow
    python scripts/build_rag_index.py

Outputs (data/rag_index/):
    reviews.parquet       cleaned reviews + metadata (row i <-> embedding i)
    embeddings.npy        float16 matrix (n_reviews x 384)
    test_reviews.parquet  English test reviews for the page-11 selectbox
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rag_index"

# Keep in sync with rag_utils.EMBEDDING_MODEL
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading data/reviews_nlp.xlsx ...")
    df = pd.read_excel(ROOT / "data" / "reviews_nlp.xlsx")
    df = df.dropna(subset=["avis_cor", "avis_en", "note"])
    df["avis_cor"] = clean(df["avis_cor"])
    df["avis_en"] = clean(df["avis_en"])
    df = df[(df["avis_cor"] != "") & (df["avis_en"] != "")]
    df["note"] = df["note"].astype(int)
    df["assureur"] = df["assureur"].astype(str)
    df = df[["note", "assureur", "produit", "avis_cor", "avis_en"]].reset_index(drop=True)
    print(f"{len(df)} reviews")

    # The embedding model is English-only and the questions are asked in English,
    # so we embed the English translation (avis_en).
    texts = [" ".join(t.split()[:200]) for t in df["avis_en"]]

    print(f"Embedding with {EMBEDDING_MODEL} ...")
    t0 = time.time()
    model = TextEmbedding(EMBEDDING_MODEL)
    emb = np.array(list(model.embed(texts, batch_size=256)), dtype=np.float32)
    print(f"Done in {time.time() - t0:.0f}s -> {emb.shape}")

    np.save(OUT / "embeddings.npy", emb.astype(np.float16))
    df.to_parquet(OUT / "reviews.parquet", index=False)

    print("Exporting test reviews ...")
    df_test = pd.read_excel(ROOT / "data" / "reviews_clean.xlsx")
    df_test = df_test[df_test["type"] == "test"].dropna(subset=["avis_en"])
    pd.DataFrame({"avis_en": clean(df_test["avis_en"])}).to_parquet(
        OUT / "test_reviews.parquet", index=False
    )

    for f in sorted(OUT.iterdir()):
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
