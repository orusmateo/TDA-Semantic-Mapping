"""Bridge persistence across cover parameters, lenses and UMAP seeds.

v2 picked the cover setting that maximised the bridge-node count. Bridge
count grows with resolution and overlap, so that rule always picks the
corner of the grid. Here no setting is privileged: a document or a
cross-field pair counts as a bridge only if it lands in a bridge node in a
large share of all settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bridges import BridgeCriteria, classify_nodes, legacy_bridge_stats, membership_matrix, permutation_test
from .mapper_graph import build_graph, graph_summary, pca_lens, umap_lens


@dataclass
class GridSpec:
    n_cubes: tuple = (5, 7, 10, 15, 20)
    overlaps: tuple = (0.3, 0.4, 0.5, 0.6)
    umap_seeds: tuple = (42, 123, 456, 789, 1024)
    include_pca: bool = True
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


@dataclass
class GridResult:
    configs: pd.DataFrame
    doc_bridge_freq: np.ndarray          # share of settings in which each doc is in a bridge node
    pair_freq: np.ndarray                # (n_a x n_b) share of settings in which a pair shares a bridge node
    a_index: np.ndarray                  # corpus row of each pair_freq row
    b_index: np.ndarray                  # corpus row of each pair_freq column
    n_settings: int
    lenses: dict = field(repr=False, default_factory=dict)


def make_lenses(X: np.ndarray, spec: GridSpec) -> dict:
    lenses = {f"umap_seed{s}": umap_lens(X, s, spec.umap_n_neighbors, spec.umap_min_dist)
              for s in spec.umap_seeds}
    if spec.include_pca:
        lenses["pca"] = pca_lens(X)
    return lenses


def run_grid(X: np.ndarray, is_a: np.ndarray, spec: GridSpec, eps: float, min_samples: int,
             criteria: BridgeCriteria = BridgeCriteria(), n_perm: int = 200, seed: int = 0,
             lenses: dict | None = None) -> GridResult:
    is_a = np.asarray(is_a, dtype=bool)
    n_docs = len(is_a)
    a_index = np.flatnonzero(is_a)
    b_index = np.flatnonzero(~is_a)
    a_pos = {d: i for i, d in enumerate(a_index)}
    b_pos = {d: i for i, d in enumerate(b_index)}

    lenses = lenses if lenses is not None else make_lenses(X, spec)
    doc_hits = np.zeros(n_docs)
    pair_hits = np.zeros((len(a_index), len(b_index)))
    rows = []
    for lens_name, lens in lenses.items():
        for nc in spec.n_cubes:
            for ov in spec.overlaps:
                graph = build_graph(lens, X, nc, ov, eps, min_samples)
                info = {"lens": lens_name, "n_cubes": nc, "overlap": ov}
                info.update(graph_summary(graph, n_docs))
                if not graph["nodes"]:
                    rows.append(info)
                    continue
                node_ids, M = membership_matrix(graph, n_docs)
                table = classify_nodes(M, is_a, node_ids, criteria)
                info["legacy_bridge_nodes"], info["legacy_mean_mixing"] = legacy_bridge_stats(M, is_a)
                for cls in ("bridge", "A-enriched", "B-enriched", "mixed", "small"):
                    info[f"n_{cls}"] = int((table["class"] == cls).sum())
                perm = permutation_test(M, is_a, criteria.min_size, n_perm=n_perm, seed=seed)
                info["mean_balance"] = perm["observed"]
                info["null_mean_balance"] = perm["null_mean"]
                info["segregation_index"] = perm["segregation_index"]
                info["p_less_mixed"] = perm["p_less_mixed"]
                rows.append(info)

                bridge_rows = np.flatnonzero(table["class"].to_numpy() == "bridge")
                in_bridge = np.zeros(n_docs, dtype=bool)
                pair_mask = np.zeros_like(pair_hits, dtype=bool)
                for r in bridge_rows:
                    members = M[r].indices
                    in_bridge[members] = True
                    ai = [a_pos[d] for d in members if is_a[d]]
                    bi = [b_pos[d] for d in members if not is_a[d]]
                    pair_mask[np.ix_(ai, bi)] = True
                doc_hits += in_bridge
                pair_hits += pair_mask

    n_settings = len(rows)
    return GridResult(configs=pd.DataFrame(rows), doc_bridge_freq=doc_hits / n_settings,
                      pair_freq=pair_hits / n_settings, a_index=a_index, b_index=b_index,
                      n_settings=n_settings, lenses=lenses)


def persistent_pairs(result: GridResult, threshold: float) -> pd.DataFrame:
    ai, bi = np.nonzero(result.pair_freq >= threshold)
    return pd.DataFrame({
        "doc_a": result.a_index[ai],
        "doc_b": result.b_index[bi],
        "freq": result.pair_freq[ai, bi],
    }).sort_values("freq", ascending=False, ignore_index=True)
