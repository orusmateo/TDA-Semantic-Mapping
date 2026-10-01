import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

from bridge_tda.bridges import (BridgeCriteria, balance, benjamini_hochberg, classify_nodes,
                                enrichment_pvalues, legacy_bridge_stats, membership_matrix,
                                permutation_test)


def test_balance_matches_legacy_mixing_when_fields_are_equal_size():
    a = np.array([13, 6, 2, 0])
    n = np.array([26, 18, 25, 10])
    legacy = np.minimum(a, n - a) / np.maximum(a, n - a)
    assert np.allclose(balance(a, n, 0.5), legacy)


def test_balance_adjusts_for_base_rate():
    # Corpus is 20% field A; a node that is 20% A mirrors the corpus.
    assert balance(np.array([4]), np.array([20]), 0.2)[0] == 1.0
    # Raw min/max would call this node very unbalanced.
    assert 4 / 16 < 0.3


def test_enrichment_pvalue_is_two_sided_hypergeometric():
    p = enrichment_pvalues(np.array([10]), np.array([10]), k_a=50, n_total=100)[0]
    expected = 2 * hypergeom.sf(9, 100, 50, 10)
    assert np.isclose(p, expected)
    centred = enrichment_pvalues(np.array([5]), np.array([10]), k_a=50, n_total=100)[0]
    assert centred == 1.0


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_membership_matrix_and_classes():
    graph = {"nodes": {"pureA": list(range(0, 20)),
                       "pureB": list(range(40, 60)),
                       "mix": list(range(15, 25)) + list(range(55, 65)),
                       "tiny": [0, 79]},
             "links": {}}
    is_a = np.array([True] * 40 + [False] * 40)
    ids, M = membership_matrix(graph, 80)
    assert M.shape == (4, 80) and M.sum() == 20 + 20 + 20 + 2
    table = classify_nodes(M, is_a, ids, BridgeCriteria(min_size=8))
    cls = dict(zip(table.node_id, table["class"]))
    assert cls == {"pureA": "A-enriched", "pureB": "B-enriched", "mix": "bridge", "tiny": "small"}


def test_legacy_bridge_stats_counts_any_mixed_node():
    M = sparse.csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]]))
    n, mix = legacy_bridge_stats(M, np.array([True, False, True, False]))
    assert n == 2 and mix == 1.0


def test_permutation_test_detects_segregation_and_not_chance():
    rng = np.random.default_rng(1)
    n_docs = 200
    nodes = [rng.choice(n_docs, 20, replace=False) for _ in range(30)]
    M = sparse.csr_matrix((np.ones(600), (np.repeat(np.arange(30), 20), np.concatenate(nodes))),
                          shape=(30, n_docs))
    random_labels = rng.permutation(np.arange(n_docs) < 100)
    chance = permutation_test(M, random_labels, min_size=8, n_perm=300)
    assert chance["p_less_mixed"] > 0.05
    assert abs(chance["segregation_index"]) < 0.1

    sorted_nodes = [np.arange(i * 20, i * 20 + 20) % n_docs for i in range(10)]
    M2 = sparse.csr_matrix((np.ones(200), (np.repeat(np.arange(10), 20), np.concatenate(sorted_nodes))),
                           shape=(10, n_docs))
    seg = permutation_test(M2, np.arange(n_docs) < 100, min_size=8, n_perm=300)
    assert seg["p_less_mixed"] < 0.01
    assert seg["segregation_index"] > 0.9
