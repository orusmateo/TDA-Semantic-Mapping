"""End-to-end validation on synthetic data with a planted bridge.

Three well-separated clusters on the unit sphere: one pure field A, one pure
field B, and one planted cluster that is half A and half B. The pipeline
should mark the planted documents as persistent bridges and the pure
documents as not, and should report chance-level mixing when labels are
random.
"""

import numpy as np
import pytest

from bridge_tda.bridges import BridgeCriteria
from bridge_tda.mapper_graph import dbscan_eps
from bridge_tda.stability import GridSpec, persistent_pairs, run_grid


def planted_data(seed=0, dim=20, n_pure=100, n_bridge=30, noise=0.15):
    rng = np.random.default_rng(seed)
    centres = np.eye(dim)[:3]
    sizes = [n_pure, n_pure, n_bridge]
    X = np.vstack([c + noise * rng.standard_normal((s, dim)) for c, s in zip(centres, sizes)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    is_a = np.concatenate([np.ones(n_pure, bool), np.zeros(n_pure, bool),
                           np.arange(n_bridge) % 2 == 0])
    planted = np.concatenate([np.zeros(2 * n_pure, bool), np.ones(n_bridge, bool)])
    return X, is_a, planted


SPEC = GridSpec(n_cubes=(4, 6), overlaps=(0.3, 0.5), umap_seeds=(0, 1), include_pca=True)
CRIT = BridgeCriteria(min_size=8, min_per_field=3, min_balance=0.5)


@pytest.fixture(scope="module")
def planted_result():
    X, is_a, planted = planted_data()
    eps = dbscan_eps(X, min_samples=5, quantile=0.9)
    res = run_grid(X, is_a, SPEC, eps=eps, min_samples=5, criteria=CRIT, n_perm=100)
    return X, is_a, planted, res


def test_planted_bridge_is_recovered(planted_result):
    _, _, planted, res = planted_result
    persistent = res.doc_bridge_freq >= 0.5
    recall = (persistent & planted).sum() / planted.sum()
    precision = (persistent & planted).sum() / max(persistent.sum(), 1)
    assert recall >= 0.9, recall
    assert precision >= 0.9, precision


def test_persistent_pairs_are_inside_planted_cluster(planted_result):
    _, _, planted, res = planted_result
    pairs = persistent_pairs(res, 0.5)
    assert len(pairs) > 0
    assert planted[pairs.doc_a].all() and planted[pairs.doc_b].all()


def test_fields_reported_as_segregated(planted_result):
    _, _, _, res = planted_result
    assert (res.configs["p_less_mixed"] < 0.05).all()


def test_random_labels_read_as_chance_mixing():
    X, is_a, _ = planted_data()
    rng = np.random.default_rng(3)
    shuffled = rng.permutation(is_a)
    eps = dbscan_eps(X, min_samples=5, quantile=0.9)
    res = run_grid(X, shuffled, SPEC, eps=eps, min_samples=5, criteria=CRIT, n_perm=100)
    assert res.configs["segregation_index"].abs().median() < 0.15


def test_overlapping_clusters_new_rule_beats_legacy_rule():
    """With overlapping clusters, v2's rule (any node holding one document
    from each field) flags nearly every document; the new rule stays precise."""
    from bridge_tda.bridges import membership_matrix
    from bridge_tda.mapper_graph import build_graph

    X, is_a, planted = planted_data(noise=0.4)
    eps = dbscan_eps(X, min_samples=5, quantile=0.9)
    res = run_grid(X, is_a, SPEC, eps=eps, min_samples=5, criteria=CRIT, n_perm=100)
    flagged = res.doc_bridge_freq >= 0.5
    assert (flagged & planted).sum() / planted.sum() >= 0.6
    assert (flagged & planted).sum() / max(flagged.sum(), 1) >= 0.85

    graph = build_graph(res.lenses["umap_seed0"], X, 4, 0.5, eps, 5)
    _, M = membership_matrix(graph, len(X))
    a = np.asarray(M @ is_a.astype(int)).ravel()
    n = np.asarray(M.sum(axis=1)).ravel()
    legacy_docs = np.unique(M[np.flatnonzero((a > 0) & (a < n))].indices)
    assert planted[legacy_docs].mean() < 0.3
