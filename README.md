# Topological Data Analysis of Indigenous Knowledge & AI

This repository contains the source code and visualization output for the research project: **"Operationalizing Isomorphology of Thematic Analysis in Topological Latent Space."**

## Abstract
This study applies **Topological Data Analysis (TDA)** via the Mapper algorithm to a corpus of 100 academic abstracts. It identifies a topological "bridge" between **Indigenous Knowledge Systems** and **AI Safety Protocols**, quantitatively demonstrating semantic isomorphism between concepts such as "Contextual Anchoring" and "Relational Accountability."

## Repository Contents
* `tda_pipeline.ipynb`: Interactive Computational Notebook performing the sentence embedding (SBERT), dimensionality reduction (UMAP), and simplicial complex construction (KeplerMapper).
* `thesis_real_data.html`: Interactive topological visualization of the semantic manifold.
* `requirements.txt`: Python dependencies.

## Methodology
1. **Vectorization:** SentenceTransformer (`all-MiniLM-L6-v2`)
2. **Lens:** UMAP (`n_neighbors=10`, `min_dist=0.1`)
3. **Mapper:** KeplerMapper w/ KMeans Clustering (`n_cubes=5`, `overlap=40%`)

## Usage
To replicate the analysis:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   