"""Document representations.

Two representations, so a finding can be checked against a change of
representation:

* sentence embeddings (sentence-transformers; needs the model downloaded once
  from Hugging Face, then runs offline), with chunk averaging for long texts;
* TF-IDF + LSA (scikit-learn only, no download), a bag-of-words baseline.

Both reduce a document to one unit vector compared by cosine similarity, so
both share the limitation discussed in the README: relations are read off
positions in one global space.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CHUNK_SIZE = 1800    # characters, roughly 512 tokens
CHUNK_OVERLAP = 200


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks or [text]


def sentence_embeddings(texts: list[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """Mean of normalised chunk embeddings per document, re-normalised.
    Abstracts fit in one chunk, so for abstracts this is a plain encoding."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    out = []
    for text in texts:
        emb = model.encode(chunk_text(text), normalize_embeddings=True, show_progress_bar=False)
        v = emb.mean(axis=0)
        out.append(v / (np.linalg.norm(v) + 1e-12))
    return np.asarray(out)


def lsa_embeddings(texts: list[str], n_components: int = 100, seed: int = 0) -> np.ndarray:
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.8, sublinear_tf=True,
                            ngram_range=(1, 2))
    Z = tfidf.fit_transform(texts)
    n_components = min(n_components, Z.shape[1] - 1, Z.shape[0] - 1)
    return normalize(TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(Z))


def distinctive_terms(texts: list[str], group: np.ndarray, top_n: int = 12) -> list[tuple[str, float]]:
    """Terms whose mean TF-IDF in the group most exceeds the corpus mean.

    v2 ranked by the group's mean TF-IDF alone, which surfaces corpus-wide
    words ("ai", "knowledge", "data") for every bridge.
    """
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.8, ngram_range=(1, 2))
    Z = tfidf.fit_transform(texts)
    names = tfidf.get_feature_names_out()
    diff = np.asarray(Z[group].mean(axis=0)).ravel() - np.asarray(Z.mean(axis=0)).ravel()
    top = np.argsort(diff)[::-1][:top_n]
    return [(names[i], round(float(diff[i]), 4)) for i in top]
