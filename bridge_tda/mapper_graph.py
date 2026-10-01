"""Lens construction and Mapper graph building."""

from __future__ import annotations

import warnings

import kmapper as km
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def umap_lens(X: np.ndarray, seed: int, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proj = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                         n_components=2, random_state=seed).fit_transform(X)
    return MinMaxScaler().fit_transform(proj)


def pca_lens(X: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(PCA(n_components=2, random_state=0).fit_transform(X))


def dbscan_eps(X: np.ndarray, min_samples: int, quantile: float) -> float:
    """eps as a quantile of each point's distance to its min_samples-th
    nearest neighbour (cosine, self included), so DBSCAN's neighbourhood size
    is set from the data, not hand-picked.

    v2 used a fixed eps of 0.5. On the 998-abstract mpnet embeddings that is
    about the 90th percentile of this distance, so DBSCAN almost never split
    a cover cell (232 of 235 nodes were the only cluster in their cell) and
    Mapper nodes were effectively the lens bins.
    """
    nn = NearestNeighbors(n_neighbors=min_samples, metric="cosine").fit(X)
    dist, _ = nn.kneighbors(X)
    return float(np.quantile(dist[:, -1], quantile))


def build_graph(lens: np.ndarray, X: np.ndarray, n_cubes: int, overlap: float,
                eps: float, min_samples: int) -> dict:
    mapper = km.KeplerMapper(verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mapper.map(lens, X, cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
                          clusterer=DBSCAN(eps=eps, min_samples=min_samples, metric="cosine"))


def graph_summary(graph: dict, n_docs: int) -> dict:
    G = nx.Graph()
    G.add_nodes_from(graph["nodes"].keys())
    for src, targets in graph["links"].items():
        for t in targets:
            G.add_edge(src, t)
    covered = set()
    for members in graph["nodes"].values():
        covered.update(members)
    split_cells = sum(1 for nid in graph["nodes"] if not nid.endswith("cluster0"))
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_connected_components(G) if G.number_of_nodes() else 0,
        "coverage": len(covered) / n_docs,
        "extra_clusters": split_cells,
    }
