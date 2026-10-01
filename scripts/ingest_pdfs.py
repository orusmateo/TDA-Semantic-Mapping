"""Build a two-field corpus from local PDFs (one or more folders per field).

Full text is extracted with PyMuPDF. Long documents are embedded later by
chunk averaging (bridge_tda.embed.sentence_embeddings).

Write the output under data/private/, which git ignores. Full-text papers are
usually not openly licensed, and community documents (for example pod
reports) are governed by their communities' data agreements (OCAP®), so
neither belongs in a public repository.

python scripts/ingest_pdfs.py --field-a ~/papers/indigenous --field-b ~/papers/ai_safety \
    --out data/private/my_corpus --embed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge_tda.corpus import FIELD_A, FIELD_B, audit_relevance, preprocess, summarize_relevance  # noqa: E402


def extract(path: Path) -> tuple[str, str]:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        pages = [page.get_text() for page in doc]
    first_lines = [ln.strip() for ln in (pages[0] if pages else "").splitlines() if len(ln.strip()) > 10]
    title = first_lines[0] if first_lines else path.stem
    return title, " ".join(" ".join(pages).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-a", nargs="+", required=True, help="folders of Field A PDFs")
    ap.add_argument("--field-b", nargs="+", required=True, help="folders of Field B PDFs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", action="store_true")
    args = ap.parse_args()

    rows = []
    for label, folders in ((FIELD_A, args.field_a), (FIELD_B, args.field_b)):
        for folder in folders:
            for pdf in sorted(Path(folder).expanduser().rglob("*.pdf")):
                try:
                    title, text = extract(pdf)
                except Exception as exc:  # unreadable or encrypted PDF
                    print(f"skip {pdf}: {exc}")
                    continue
                rows.append({"title": title, "text": text, "label": label, "url": str(pdf)})
    df = pd.DataFrame(rows)
    df["text_clean"] = df["text"].map(preprocess)
    short = df["text_clean"].str.len() < 100
    if short.any():
        print(f"dropping {short.sum()} documents with <100 characters of text (scanned? run OCR first)")
    df = audit_relevance(df[~short].drop_duplicates("title").reset_index(drop=True))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "corpus.csv", index=False)
    print(summarize_relevance(df).as_dict())
    if args.embed:
        from bridge_tda.embed import sentence_embeddings

        np.save(out / "embeddings_mpnet.npy", sentence_embeddings(df["text_clean"].tolist()))


if __name__ == "__main__":
    main()
