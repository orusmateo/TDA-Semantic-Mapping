"""Harvest a two-field corpus from arXiv with phrase-quoted queries.

Writes corpus.csv and manifest.json (queries, retrieval time, versions) to
--out. Papers returned by both queries are kept with label "Both" so they
can be studied as natural bridges; v2 silently assigned them to Field A.

Needs network access to export.arxiv.org. Add --embed to compute
all-mpnet-base-v2 embeddings (needs huggingface.co once, to download the
model).

python scripts/fetch_arxiv.py --max-results 500 --out data/arxiv_phrase_2026-10 --embed
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge_tda.corpus import ARXIV_QUERIES, audit_relevance, preprocess, summarize_relevance  # noqa: E402


def harvest(max_results: int) -> pd.DataFrame:
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    records: dict[str, dict] = {}
    for field, query in ARXIV_QUERIES.items():
        print(f"{field}: {query}")
        search = arxiv.Search(query=query, max_results=max_results,
                              sort_by=arxiv.SortCriterion.Relevance)
        for r in client.results(search):
            key = r.get_short_id().split("v")[0]
            if key in records:
                records[key]["retrieved_by"].add(field)
                continue
            records[key] = {
                "arxiv_id": r.get_short_id(),
                "url": r.entry_id,
                "title": " ".join(r.title.split()),
                "text": " ".join(r.summary.split()),
                "published": str(r.published.date()),
                "categories": ",".join(r.categories),
                "retrieved_by": {field},
            }
    rows = []
    for rec in records.values():
        fields = sorted(rec.pop("retrieved_by"))
        rec["label"] = fields[0] if len(fields) == 1 else "Both"
        rec["retrieved_by"] = ";".join(fields)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-results", type=int, default=500, help="per field")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = harvest(args.max_results)
    df["text_clean"] = df["text"].map(preprocess)
    df = df[df["text_clean"].str.len() >= 100].reset_index(drop=True)
    df = audit_relevance(df)
    df.to_csv(out / "corpus.csv", index=False)

    labelled = df[df["label"] != "Both"]
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queries": ARXIV_QUERIES,
        "max_results_per_field": args.max_results,
        "label_counts": df["label"].value_counts().to_dict(),
        "relevance": summarize_relevance(labelled).as_dict(),
        "versions": {p: version(p) for p in ("arxiv", "pandas", "numpy")},
    }
    if args.embed:
        from bridge_tda.embed import sentence_embeddings

        np.save(out / "embeddings_mpnet.npy", sentence_embeddings(df["text_clean"].tolist()))
        manifest["versions"]["sentence-transformers"] = version("sentence-transformers")
        manifest["embedding_model"] = "all-mpnet-base-v2"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
