"""Node composition, bridge classification and permutation nulls.

v2 called any Mapper node holding at least one document from each field a
"bridge node" and scored it with min(a, b) / max(a, b). Two problems:

* With labels shuffled at random almost every node satisfies "one of each",
  so the bridge count mostly measures how many nodes there are.
* min/max of raw counts assumes the two fields are the same size. Once the
  corpus is filtered for relevance the fields are unbalanced.

Here composition is compared with the corpus base rate (balance), each node
is tested for enrichment with a hypergeometric test (Benjamini-Hochberg FDR
across nodes), and graph-level mixing is compared with a label-permutation
null that keeps the graph fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom


@dataclass(frozen=True)
class BridgeCriteria:
    """A priori thresholds for calling a node a bridge.

    min_size:       nodes smaller than this are left unclassified.
    min_per_field:  documents required from each field.
    min_balance:    base-rate-adjusted balance (1 = node mirrors the corpus).
    alpha:          FDR level; a bridge must not be significantly enriched
                    for either field.
    """

    min_size: int = 8
    min_per_field: int = 3
    min_balance: float = 0.5
    alpha: float = 0.05


def membership_matrix(graph: dict, n_docs: int) -> tuple[list[str], sparse.csr_matrix]:
    """Return node ids and a binary (n_nodes x n_docs) membership matrix."""
    node_ids = list(graph["nodes"].keys())
    rows, cols = [], []
    for r, nid in enumerate(node_ids):
        members = graph["nodes"][nid]
        rows.extend([r] * len(members))
        cols.extend(members)
    data = np.ones(len(rows), dtype=np.int32)
    M = sparse.csr_matrix((data, (rows, cols)), shape=(len(node_ids), n_docs))
    M.data[:] = 1
    return node_ids, M


def balance(a: np.ndarray, n: np.ndarray, p_a: float) -> np.ndarray:
    """Base-rate-adjusted mixing in [0, 1].

    r_a = (a/n) / p_a and r_b = (b/n) / (1 - p_a) are each field's share in
    the node relative to its share in the corpus. Balance is min/max of the
    two, so 1 means the node mirrors the corpus and 0 means a pure node. With
    p_a = 0.5 this equals v2's min(a, b) / max(a, b).
    """
    a = np.asarray(a, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_a = (a / n) / p_a
        r_b = ((n - a) / n) / (1.0 - p_a)
        out = np.minimum(r_a, r_b) / np.maximum(r_a, r_b)
    return np.nan_to_num(out, nan=0.0)


def enrichment_pvalues(a: np.ndarray, n: np.ndarray, k_a: int, n_total: int) -> np.ndarray:
    """Two-sided hypergeometric p-value for a field-A count in each node."""
    a = np.asarray(a)
    n = np.asarray(n)
    lower = hypergeom.cdf(a, n_total, k_a, n)
    upper = hypergeom.sf(a - 1, n_total, k_a, n)
    return np.minimum(1.0, 2.0 * np.minimum(lower, upper))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def classify_nodes(M: sparse.csr_matrix, is_a: np.ndarray, node_ids: list[str],
                   criteria: BridgeCriteria = BridgeCriteria()) -> pd.DataFrame:
    """One row per node: counts, balance, enrichment q-value and class.

    Classes: 'A-enriched', 'B-enriched', 'bridge', 'mixed' (neither enriched
    nor balanced enough to be a bridge) and 'small'.
    """
    is_a = np.asarray(is_a, dtype=bool)
    n_total = len(is_a)
    k_a = int(is_a.sum())
    p_a = k_a / n_total
    n = np.asarray(M.sum(axis=1)).ravel()
    a = np.asarray(M @ is_a.astype(np.int32)).ravel()
    b = n - a
    bal = balance(a, n, p_a)
    p = enrichment_pvalues(a, n, k_a, n_total)
    big = n >= criteria.min_size
    q = np.ones_like(p)
    q[big] = benjamini_hochberg(p[big])

    cls = np.full(len(n), "mixed", dtype=object)
    cls[~big] = "small"
    enriched = big & (q < criteria.alpha)
    cls[enriched & (a / np.maximum(n, 1) > p_a)] = "A-enriched"
    cls[enriched & (a / np.maximum(n, 1) <= p_a)] = "B-enriched"
    bridge = (big & ~enriched & (bal >= criteria.min_balance)
              & (a >= criteria.min_per_field) & (b >= criteria.min_per_field))
    cls[bridge] = "bridge"
    return pd.DataFrame({"node_id": node_ids, "size": n, "a": a, "b": b,
                         "balance": bal, "p": p, "q": q, "class": cls})


def legacy_bridge_stats(M: sparse.csr_matrix, is_a: np.ndarray) -> tuple[int, float]:
    """v2's statistics: nodes with >=1 doc per field, and their mean min/max."""
    is_a = np.asarray(is_a, dtype=bool)
    n = np.asarray(M.sum(axis=1)).ravel()
    a = np.asarray(M @ is_a.astype(np.int32)).ravel()
    b = n - a
    mask = (a > 0) & (b > 0)
    if not mask.any():
        return 0, 0.0
    return int(mask.sum()), float(np.mean(np.minimum(a, b)[mask] / np.maximum(a, b)[mask]))


def mean_balance(M: sparse.csr_matrix, is_a: np.ndarray, min_size: int) -> float:
    """Size-weighted mean balance over nodes of at least min_size."""
    is_a = np.asarray(is_a, dtype=bool)
    n = np.asarray(M.sum(axis=1)).ravel()
    a = np.asarray(M @ is_a.astype(np.int32)).ravel()
    keep = n >= min_size
    if not keep.any():
        return float("nan")
    bal = balance(a[keep], n[keep], is_a.mean())
    return float(np.average(bal, weights=n[keep]))


def permutation_test(M: sparse.csr_matrix, is_a: np.ndarray, min_size: int,
                     n_perm: int = 1000, seed: int = 0) -> dict:
    """Graph-level mixing against shuffled labels on the same graph.

    Returns the observed size-weighted mean balance, the null mean and SD,
    a one-sided p-value for "less mixed than chance" (with +1 correction),
    and a segregation index 1 - observed / null_mean (0 = chance mixing,
    1 = complete separation).

    Two literatures retrieved with different queries will almost always be
    less mixed than chance, so a small p here is expected, not a discovery.
    The index is the useful number: it says how far from chance.
    """
    rng = np.random.default_rng(seed)
    is_a = np.asarray(is_a, dtype=bool)
    obs = mean_balance(M, is_a, min_size)
    null = np.array([mean_balance(M, rng.permutation(is_a), min_size) for _ in range(n_perm)])
    p = (1 + np.sum(null <= obs)) / (n_perm + 1)
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "p_less_mixed": float(p), "segregation_index": float(1 - obs / null.mean()),
            "n_perm": n_perm}


def knn_segregation(X: np.ndarray, is_a: np.ndarray, k: int = 10, n_perm: int = 1000,
                    seed: int = 0) -> dict:
    """Mapper-free check: share of each document's k nearest neighbours
    (cosine, in the full embedding space) that carry the same label."""
    from sklearn.neighbors import NearestNeighbors

    is_a = np.asarray(is_a, dtype=bool)
    _, idx = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X).kneighbors(X)
    idx = idx[:, 1:]

    def same_rate(lab: np.ndarray) -> float:
        return float((lab[idx] == lab[:, None]).mean())

    rng = np.random.default_rng(seed)
    obs = same_rate(is_a)
    null = np.array([same_rate(rng.permutation(is_a)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return {"k": k, "observed_same_label": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)), "p_more_segregated": float(p), "n_perm": n_perm}
