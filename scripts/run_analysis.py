"""Run the bridge analysis on a corpus and write results to a folder.

Examples
--------
python scripts/run_analysis.py --corpus data/legacy_arxiv998/corpus.csv \
    --embeddings data/legacy_arxiv998/embeddings_mpnet.npy --subset all \
    --out results/arxiv998_all_mpnet --legacy-check

python scripts/run_analysis.py --corpus data/legacy_arxiv998/corpus.csv \
    --representation lsa --subset on_topic --out results/arxiv998_ontopic_lsa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge_tda.bridges import (BridgeCriteria, classify_nodes, knn_segregation,  # noqa: E402
                                legacy_bridge_stats, membership_matrix, permutation_test)
from bridge_tda.corpus import FIELD_A, FIELD_B, audit_relevance, summarize_relevance  # noqa: E402
from bridge_tda.embed import distinctive_terms, lsa_embeddings, sentence_embeddings  # noqa: E402
from bridge_tda.mapper_graph import build_graph, dbscan_eps, graph_summary, umap_lens  # noqa: E402
from bridge_tda.stability import GridSpec, persistent_pairs, run_grid  # noqa: E402

PRIMARY = {"lens": "umap_seed42", "n_cubes": 10, "overlap": 0.5}   # fixed a priori, mid-grid


def fisher(table: list[list[int]]) -> dict:
    odds, p = fisher_exact(table)
    return {"table": table, "odds_ratio": None if not np.isfinite(odds) else round(float(odds), 3),
            "p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embeddings", help=".npy aligned with corpus rows (omit to compute)")
    ap.add_argument("--representation", default="mpnet", choices=["mpnet", "lsa"])
    ap.add_argument("--subset", default="all", choices=["all", "on_topic"])
    ap.add_argument("--eps-quantile", type=float, default=0.5)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--grid-perm", type=int, default=200)
    ap.add_argument("--min-balance", type=float, default=0.5)
    ap.add_argument("--min-per-field", type=int, default=3)
    ap.add_argument("--persistence", type=float, default=0.5,
                    help="share of settings a document or pair must be a bridge in")
    ap.add_argument("--legacy-check", action="store_true",
                    help="rebuild v2's selected graph (n_cubes=20, overlap=0.6, eps=0.5)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.corpus)
    df = df[df["label"].isin([FIELD_A, FIELD_B])].reset_index(drop=True)
    df = audit_relevance(df)
    relevance_all = summarize_relevance(df).as_dict()

    if args.representation == "mpnet":
        if args.embeddings:
            X = np.load(args.embeddings)
            assert len(X) == len(df), "embeddings must align with corpus rows"
        else:
            X = sentence_embeddings(df["text_clean"].tolist())
    keep = np.ones(len(df), bool) if args.subset == "all" else df["on_topic"].to_numpy()
    df = df[keep].reset_index(drop=True)
    if args.representation == "mpnet":
        X = X[keep]
    else:
        X = lsa_embeddings((df["title"] + ". " + df["text_clean"]).tolist())
    is_a = (df["label"] == FIELD_A).to_numpy()
    texts = (df["title"] + ". " + df["text_clean"]).tolist()
    criteria = BridgeCriteria(min_per_field=args.min_per_field, min_balance=args.min_balance)
    persistence = args.persistence
    summary: dict = {"args": vars(args), "relevance_full_corpus": relevance_all,
                     "analysed": summarize_relevance(df).as_dict(), "criteria": criteria.__dict__,
                     "primary_setting": PRIMARY, "persistence_threshold": persistence}

    # Mapper-free segregation in the full embedding space.
    summary["knn_segregation"] = knn_segregation(X, is_a, k=10, n_perm=args.n_perm)

    # Mapper-free: the closest cross-field pairs by cosine similarity.
    S = X[is_a] @ X[~is_a].T
    a_rows, b_rows = np.flatnonzero(is_a), np.flatnonzero(~is_a)
    top = np.argsort(S, axis=None)[::-1][:50]
    ai, bi = np.unravel_index(top, S.shape)
    nearest = pd.DataFrame({"cosine": S[ai, bi], "title_a": df["title"].to_numpy()[a_rows[ai]],
                            "title_b": df["title"].to_numpy()[b_rows[bi]],
                            "a_on_topic": df["on_topic"].to_numpy()[a_rows[ai]]})
    nearest.to_csv(out / "nearest_cross_field_pairs.csv", index=False)
    within_a = X[is_a] @ X[is_a].T
    summary["cosine"] = {
        "cross_field_max": float(S.max()),
        "cross_field_median": float(np.median(S)),
        "within_a_median": float(np.median(within_a[np.triu_indices(len(within_a), 1)])),
        "nearest_b_for_each_a_median": float(np.median(S.max(axis=1))),
    }

    eps = dbscan_eps(X, args.min_samples, args.eps_quantile)
    summary["dbscan"] = {"eps": eps, "eps_quantile": args.eps_quantile, "min_samples": args.min_samples}

    spec = GridSpec()
    grid = run_grid(X, is_a, spec, eps=eps, min_samples=args.min_samples, criteria=criteria,
                    n_perm=args.grid_perm)
    grid.configs.to_csv(out / "grid_settings.csv", index=False)
    cfg = grid.configs
    summary["grid"] = {
        "n_settings": grid.n_settings,
        "segregation_index_median": float(cfg["segregation_index"].median()),
        "segregation_index_range": [float(cfg["segregation_index"].min()), float(cfg["segregation_index"].max())],
        "settings_less_mixed_than_chance_p05": int((cfg["p_less_mixed"] < 0.05).sum()),
        "bridge_nodes_median": float(cfg["n_bridge"].median()),
        "legacy_bridge_nodes_median": float(cfg["legacy_bridge_nodes"].median()),
        "coverage_median": float(cfg["coverage"].median()),
        "extra_clusters_median": float(cfg["extra_clusters"].median()),
        "corr_legacy_bridges_vs_nodes": float(np.corrcoef(cfg["legacy_bridge_nodes"], cfg["nodes"])[0, 1]),
    }

    # Primary graph, fixed in advance.
    lens = grid.lenses[PRIMARY["lens"]]
    g = build_graph(lens, X, PRIMARY["n_cubes"], PRIMARY["overlap"], eps, args.min_samples)
    ids, M = membership_matrix(g, len(df))
    table = classify_nodes(M, is_a, ids, criteria)
    table.to_csv(out / "primary_nodes.csv", index=False)
    summary["primary_graph"] = {**graph_summary(g, len(df)),
                                "classes": table["class"].value_counts().to_dict(),
                                "legacy_bridge_nodes": legacy_bridge_stats(M, is_a)[0],
                                "permutation": permutation_test(M, is_a, criteria.min_size, args.n_perm)}

    # Persistent bridge documents and pairs.
    df["bridge_freq"] = grid.doc_bridge_freq
    persistent = df["bridge_freq"] >= persistence
    cols = ["bridge_freq", "label", "on_topic", "dual", "title", "url"]
    df.loc[persistent, cols].sort_values("bridge_freq", ascending=False).to_csv(
        out / "persistent_bridge_docs.csv", index=False)
    pairs = persistent_pairs(grid, persistence)
    pairs["title_a"] = df["title"].to_numpy()[pairs["doc_a"]]
    pairs["title_b"] = df["title"].to_numpy()[pairs["doc_b"]]
    pairs.to_csv(out / "persistent_pairs.csv", index=False)
    summary["persistence"] = {
        "docs": int(persistent.sum()),
        "docs_field_a": int((persistent & (df["label"] == FIELD_A)).sum()),
        "docs_field_b": int((persistent & (df["label"] == FIELD_B)).sum()),
        "pairs": int(len(pairs)),
        "doc_freq_quantiles": {str(q): float(np.quantile(grid.doc_bridge_freq, q)) for q in (0.5, 0.9, 0.99)},
    }

    # Bridge groups: connected components of the persistent-pair graph.
    G = nx.Graph()
    G.add_edges_from(zip(pairs["doc_a"], pairs["doc_b"]))
    groups = []
    for comp in sorted(nx.connected_components(G), key=len, reverse=True):
        members = np.array(sorted(comp))
        mask = np.zeros(len(df), bool)
        mask[members] = True
        groups.append({
            "size": int(len(members)),
            "field_a": int(is_a[members].sum()),
            "field_b": int((~is_a[members]).sum()),
            "on_topic_share": float(df["on_topic"].to_numpy()[members].mean()),
            "distinctive_terms": distinctive_terms(texts, mask),
            "titles": [f"[{df['label'][i]}{'' if df['on_topic'][i] else ', off-topic'}] {df['title'][i]}"
                       for i in members],
        })
    summary["bridge_groups"] = groups

    # Controls.
    controls = {}
    a_rows = df["label"] == FIELD_A
    if args.subset == "all" and (~df.loc[a_rows, "on_topic"]).any():
        off = a_rows & ~df["on_topic"]
        on = a_rows & df["on_topic"]
        controls["field_a_off_topic_in_bridges"] = fisher([
            [int((off & persistent).sum()), int((off & ~persistent).sum())],
            [int((on & persistent).sum()), int((on & ~persistent).sum())]])
    dual = df["dual"]
    controls["dual_lexicon_in_bridges"] = fisher([
        [int((dual & persistent).sum()), int((dual & ~persistent).sum())],
        [int((~dual & persistent).sum()), int((~dual & ~persistent).sum())]])
    controls["dual_lexicon_docs"] = [
        {"title": t, "label": lab, "bridge_freq": float(f)}
        for t, lab, f in zip(df.loc[dual, "title"], df.loc[dual, "label"], df.loc[dual, "bridge_freq"])]
    summary["controls"] = controls

    if args.legacy_check:
        lens42 = umap_lens(X, 42)
        g_old = build_graph(lens42, X, 20, 0.6, 0.5, 5)
        _, M_old = membership_matrix(g_old, len(df))
        n_b, mix = legacy_bridge_stats(M_old, is_a)
        perm = permutation_test(M_old, is_a, min_size=1, n_perm=args.n_perm)
        rng = np.random.default_rng(0)
        null_v2 = np.array([legacy_bridge_stats(M_old, rng.permutation(is_a)) for _ in range(args.n_perm)])
        summary["legacy_check"] = {**graph_summary(g_old, len(df)), "legacy_bridge_nodes": n_b,
                                   "legacy_mean_mixing": mix,
                                   "legacy_null_bridge_nodes_mean": float(null_v2[:, 0].mean()),
                                   "legacy_null_bridge_nodes_min": float(null_v2[:, 0].min()),
                                   "legacy_null_mean_mixing": float(null_v2[:, 1].mean()),
                                   "permutation_min_size1": perm,
                                   "v2_reported": {"nodes": 235, "bridge_nodes": 140, "mean_mixing": 0.222}}

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("analysed", "knn_segregation", "cosine", "dbscan", "grid", "persistence")},
                     indent=1, default=float))
    print("primary:", json.dumps(summary["primary_graph"], default=float))
    print("controls:", json.dumps({k: v for k, v in controls.items() if k != "dual_lexicon_docs"}, default=float))
    if args.legacy_check:
        print("legacy:", json.dumps(summary["legacy_check"], default=float))
    for grp in groups[:8]:
        print(f"\n group size={grp['size']} A={grp['field_a']} B={grp['field_b']} "
              f"on_topic={grp['on_topic_share']:.2f} terms={[t for t, _ in grp['distinctive_terms'][:8]]}")
        for t in grp["titles"][:6]:
            print("   ", t[:120])


if __name__ == "__main__":
    main()
