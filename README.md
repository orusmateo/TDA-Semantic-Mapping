# TDA-Semantic-Mapping

Topological bridge detection between two research literatures: Mapper graphs
(KeplerMapper) built over document embeddings, used to ask where papers from
two fields sit together. The test case is Indigenous knowledge systems
(Field A) and AI safety and ethics (Field B), using arXiv abstracts.

This is version 3. It audits and replaces the v1 and v2 pipelines
(`legacy/`). Their headline claims do not survive the audit, and the reasons
are documented below as results in their own right.

## Findings in brief

All numbers come from `results/` and can be regenerated with
`scripts/reproduce.sh`. The corpus is the 998 arXiv abstracts harvested by
v2 (published 1996 to February 2026), frozen in `data/legacy_arxiv998/`
together with v2's `all-mpnet-base-v2` embeddings.

1. **Most of the corpus is not about either field.** v2's arXiv queries were
   unquoted, so a single word such as "knowledge" or "AI" was enough to
   retrieve a paper. Only 76 of the 500 "Indigenous knowledge" abstracts
   (15.2%) use any Indigenous, TEK or decolonial term. For the "AI safety"
   abstracts the figure is 352 of 498 (70.7%). The rest are knowledge graphs,
   reinforcement learning, dental imaging and similar.

2. **v2's 140 "bridge nodes" do not indicate bridging.** Rebuilding v2's
   chosen graph gives 244 nodes, 142 bridge nodes and mean mixing 0.214
   (v2 reported 235 / 140 / 0.222; the difference is UMAP version drift).
   With the field labels shuffled at random on the same graph, v2's rule
   finds 243 bridge nodes on average (never fewer than 235) and mean mixing
   0.708. A node counts as a bridge under v2's rule if it holds one paper
   from each field, and almost every node of 20+ papers does by chance. The
   observed graph is far *less* mixed than chance.

3. **The two literatures are strongly separated in embedding space.** 88.7%
   of each abstract's 10 nearest neighbours carry its own label (49.9%
   expected by chance). On the on-topic subset it is 96.9% (70.8% by
   chance). Each run builds 120 Mapper graphs (5 cover resolutions × 4
   overlaps × 6 lenses). In every run, all 120 are less mixed than chance
   (label-permutation p < 0.05), with one exception: one setting in the
   on-topic LSA run (p = 0.22). The median segregation index per run is
   0.85 to 1.0, where 0 means chance mixing and 1 means none.

4. **In the full corpus, the stable bridges are artefacts of the query
   leak.** A paper is a persistent bridge if it sits in a bridge node in at
   least half of the 120 settings. Depending on DBSCAN radius and
   representation, there are 9, 39, 50 or 74 such papers. Their Field A
   members are almost all off-topic: 4 of 4, 23 of 23, 22 of 23 and 39 of
   40. The bridge groups are neurosymbolic AI, reinforcement learning and
   continual learning, meeting "knowledge"-titled machine-learning papers
   that the Field A query swept in.

5. **Restricted to on-topic papers (76 + 352), no persistent bridge appears.**
   This holds under both representations (mpnet sentence embeddings and a
   TF-IDF/LSA bag-of-words baseline) and both DBSCAN radii. With relaxed
   criteria (minority at ≥25% of its expected share, ≥2 papers per field,
   persistence ≥25%), one group appears, in one of the four runs only (LSA,
   wide radius): AI bias and fairness, holding 2 Indigenous-field papers
   ("Effectiveness of Debiasing Techniques: An Indigenous Qualitative
   Analysis"; "The Effect of Stereotypes on Perceived Competence of
   Indigenous Software Practitioners") and 14 AI-fairness papers. This is a
   candidate, not a finding.

6. **Contact between the fields runs through a few individual papers, and
   this method cannot see them.** Without Mapper, look at the 10 closest
   cross-field pairs (mpnet cosine 0.61–0.69). Eight involve one of three
   Indigenous-field papers that the lexicon also marks as using AI-ethics
   vocabulary: Indigenous data sovereignty, AI for endangered Indigenous
   languages, and TEK-informed ecosystem restoration with machine learning.
   The other two ("Ethical Considerations for Machine Translation of
   Indigenous Languages"; "Evaluating Machine Perception of Indigeneity")
   are about the same intersection, but the lexicon misses them, a known
   false negative. Eight papers use both vocabularies. Treated as a positive
   control, none of them reaches the persistence threshold in any run
   (highest bridge frequency 0.33). Mapper bridge detection looks for
   *regions* where the fields mix; contact carried by a handful of papers
   is below what it can register.

### What this does and does not show

It shows that, in arXiv abstracts, Indigenous-knowledge and AI-safety
research occupy separate regions of two quite different representations.
It also shows that the contact that exists is sparse and paper-level, not
regional. It does **not** show that the fields have no conceptual relation.
Three things limit the reach of the result:

* **arXiv** under-represents Indigenous scholarship, which is mostly
  published in journals that arXiv does not index (for example *AlterNative*
  and the *Journal of Ethnobiology*). Field A here is the part of
  Indigenous-knowledge research that reaches a STEM preprint server.
* **The relevance lexicons** (`bridge_tda/corpus.py`) are hand-written and
  were tightened by reading this corpus. Every exclusion is listed in the
  code and tested. They will misclassify some papers, and their
  false-negative rate has not been measured against hand labels.
* **The representation.** Both representations reduce a document to one
  vector and read relatedness off cosine similarity in one shared space.
  The author's "Euclidean Error" argument holds that this kind of
  representation cannot hold knowledge systems in which relation comes
  before position
  ([abstract](https://pes.hds.harvard.edu/publication/geometry-belonging-euclidean-error-large-language-model-processing-anishinaabe)).
  The separation found here is therefore partly a statement about what this
  geometry can register. A relation-first encoding could find contact that
  this pipeline cannot. That is a hypothesis this repository does not test.

## What changed from v2, and why

| v2 | Problem found | v3 |
|---|---|---|
| Unquoted arXiv queries | "knowledge" alone retrieves a paper; 15% of Field A on topic | Phrase-quoted, field-prefixed queries (`bridge_tda/corpus.py`); every paper audited against a field lexicon |
| Duplicate titles dropped, first kept | Papers retrieved by *both* queries (natural bridges) silently became Field A | Kept with label `Both` (`scripts/fetch_arxiv.py`) |
| Bridge node = ≥1 paper from each field | True for almost every node under random labels | Node is a bridge only if not significantly enriched for either field (hypergeometric test, BH-FDR 0.05), base-rate-adjusted balance ≥0.5, ≥3 papers per field, size ≥8 |
| Mixing = min(a, b) / max(a, b) | Assumes equal-sized fields | Balance relative to corpus base rate; equals v2's ratio when fields are equal |
| No null model | No way to tell structure from chance | Label-permutation null on the same graph; Mapper-free kNN segregation test |
| Cover chosen to maximise bridge count | Bridge count rises with resolution, so the rule picks the grid corner | No setting privileged: results reported as persistence across 120 settings; a fixed mid-grid setting (10 cubes, 50% overlap, UMAP seed 42) for the node table |
| DBSCAN eps = 0.5 (cosine) | ≈90th percentile of neighbour distance; split 3 of 235 cover cells, so nodes were lens bins | eps set from the data (quantile of 5th-neighbour distance); median and 90th percentile both reported |
| "Run 5 seeds and average" | Only seed 42 was used for the lens | Five UMAP seeds and a PCA lens are all part of the grid |
| SPECTER2 model comparison | Never ran (`peft` missing; see notebook output) | Bag-of-words TF-IDF/LSA baseline that runs with no download |
| TF-IDF "bridge themes" by mean weight | Surfaces corpus-wide words ("ai", "knowledge", "data") | Terms ranked by excess over the corpus mean |

The v1 README claimed to "quantitatively demonstrat[e] semantic isomorphism"
between "Contextual Anchoring" and "Relational Accountability". Neither
phrase appears in the v1 visualization (`legacy/v1/thesis_real_data.html`),
and no code in this repository supports the claim. v1 used KMeans with two
clusters in every cover cell, which forces a split whether or not one exists.

## Validation

`python -m pytest` runs 22 tests in about 30 seconds.

* **Planted bridge (synthetic).** Three clusters on a sphere: pure Field A,
  pure Field B, and a planted 50/50 cluster.
  * Well-separated clusters: 93% of the planted papers are recovered, at
    100% precision.
  * Overlapping clusters: the new rule recovers 70% at 91% precision. v2's
    rule flags 229 of 230 papers (13% precision).
* **Chance calibration.** With random labels the segregation index stays
  near 0 and the permutation test does not reject.
* **Statistics.** Base-rate balance, hypergeometric p-values and BH
  q-values are checked against known values.
* **Lexicons.** Each exclusion found in the audit (for example "indigenous
  bundles", "indigenous crops", "traditional knowledge tracing") and each
  kept sense is tested.
* **Harvesters.** Dual-query labelling is tested against a mocked arXiv
  client; PDF ingestion against generated PDFs. Neither was run against
  live arXiv while building v3, because the build environment could not
  reach `export.arxiv.org` or `huggingface.co`.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest
./scripts/reproduce.sh          # about 20 minutes on 4 cores; JOBS=3 by default
```

`reproduce.sh` runs 13 analyses and rebuilds `results/SUMMARY.md`. Each run
folder holds:

* `summary.json`
* `grid_settings.csv` (every one of the 120 settings)
* `primary_nodes.csv`
* `persistent_bridge_docs.csv`
* `persistent_pairs.csv`
* `nearest_cross_field_pairs.csv`

Runs are deterministic on a given machine (fixed seeds). UMAP output can
differ slightly across library versions.

## Running on a new corpus

```bash
# arXiv (needs export.arxiv.org; --embed also needs huggingface.co once and sentence-transformers)
python scripts/fetch_arxiv.py --max-results 500 --out data/arxiv_phrase_YYYY-MM --embed

# Local PDFs, e.g. reports or papers you hold (stays out of git: data/private/ is ignored)
python scripts/ingest_pdfs.py --field-a <folder> --field-b <folder> --out data/private/<name> --embed

python scripts/run_analysis.py --corpus <dir>/corpus.csv --embeddings <dir>/embeddings_mpnet.npy \
    --subset on_topic --out results/<name>
python scripts/run_analysis.py --corpus <dir>/corpus.csv --representation lsa \
    --subset on_topic --out results/<name>_lsa
```

Community documents, such as research-pod reports, are governed by their
communities' data agreements (OCAP®). Keep their corpora, embeddings and
results under `data/private/` or another ignored path. The scripts never
upload anything.

## Method

1. **Representation.** Either the abstract's sentence embedding
   (`all-mpnet-base-v2`, unit length; chunk-averaged for long texts) or
   TF-IDF (unigrams and bigrams) reduced to 100 dimensions by truncated SVD
   (LSA).
2. **Lens.** UMAP to 2-D (cosine, 15 neighbours, min_dist 0.1) with seeds
   42, 123, 456, 789 and 1024, plus PCA. Each is rescaled to [0, 1].
3. **Cover and clustering.** 5, 7, 10, 15 or 20 cubes per axis with
   30–60% overlap. DBSCAN (cosine, min_samples 5), with eps set to the
   median or 90th percentile of each point's 5th-neighbour distance.
   DBSCAN discards points it cannot cluster, so coverage is reported.
4. **Node classification.** Each node is compared with the corpus base
   rate. Two-sided hypergeometric test, Benjamini–Hochberg across nodes of
   size ≥8. Classes: A-enriched, B-enriched, bridge, mixed or small.
5. **Graph-level nulls.**
   * Size-weighted mean balance against 200 label permutations per setting
     (1,000 for the fixed setting).
   * Same-label share among 10 nearest neighbours against 1,000
     permutations.
6. **Persistence.** For each paper, and for each Field A–Field B pair, the
   share of the 120 settings in which it sits in a bridge node. Persistent
   means ≥0.5. Groups are the connected components of the persistent-pair
   graph. Each group's distinctive terms are those whose mean TF-IDF in the
   group most exceeds the corpus mean.

What was fixed in advance, and what was not:

* The bridge criteria, persistence threshold and fixed setting were chosen
  before any run on real data.
* The 90th-percentile eps run was added afterwards, because the median eps
  discarded 37% of papers as noise.
* The lexicons were tightened after reading the corpus.
* The relaxed criteria are a sensitivity check, not an alternative result.

## Layout

```
bridge_tda/        corpus.py (queries, lexicons), embed.py, mapper_graph.py,
                   bridges.py (classification, nulls), stability.py (grid, persistence)
scripts/           fetch_arxiv.py, ingest_pdfs.py, run_analysis.py,
                   summarize_results.py, reproduce.sh
tests/             unit, synthetic-validation and harvester tests
data/              legacy_arxiv998/, legacy_arxiv100/: v2's corpora and mpnet embeddings
results/           one folder per run, plus SUMMARY.md
legacy/v1/         v1 visualization (KMeans, 5 cubes)
legacy/v2/         v2 notebooks with saved outputs, and their result files
```

## Data and licence

Code: MIT (`LICENSE.txt`). The corpora in `data/` are arXiv titles and
abstracts, which arXiv distributes under CC0 1.0
([arXiv licences](https://info.arxiv.org/help/license/index.html)). Full-text
PDFs are not redistributed.

Developed with the assistance of Anthropic Claude.
