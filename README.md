# Tweet Clustering

A short-text clustering benchmark built for a thesis on clustering social network texts. The central premise is simple but disciplined: **if short-text clustering is genuinely hard, the dataset should not quietly make it easy**. The repo therefore treats dataset curation as a first-class research contribution, not a preprocessing footnote.

Short texts like tweets are notoriously difficult to cluster. Their sparse representation, high information density, and abundance of noise, misspellings, and non-English content all make the task harder than standard document clustering. This project builds a controlled benchmark, evaluates full end-to-end pipelines (not isolated models), and persists enough state to inspect failures without rerunning expensive jobs.

<div align="center">
  <table>
    <tr>
      <td align="center" valign="top">
        <img src="plots/metric_corr.png" alt="Metric correlation heatmap" width="520" style="display:block;">
      </td>
      <td align="center" valign="top">
        <img src="plots/cluster_contingency_matrix.png" alt="Cluster contingency matrix" width="600" style="display:block;">
      </td>
    </tr>
  </table>
</div>

## Highlights

- **Best result**: `distilroberta+umap5+hdbscan` achieves ARI `0.914` on the 70k-tweet benchmark — outperforming MPNet-based pipelines despite MPNet's general superiority on smaller datasets.
- **UMAP consistently wins**: across all embedding families tested, low-dimensional UMAP projections (size 5–15) outperform both raw embeddings and Truncated SVD before clustering.
- **Speed/quality tradeoff**: `use+umap5+kmeans` is the best choice when speed matters. MPNet is the slowest. Word2Vec is a reasonable baseline if you want no pretrained transformer dependency.
- **LDA and Doc2Vec fail**: both perform poorly on short texts — LDA needs large token counts, and the corpus is too small for Doc2Vec to train well.

## What makes this codebase distinctive

**Dataset generation is not a preprocessing footnote.** `generate_dataset.ipynb` is where much of the experimental rigor lives. Several topic slices deliberately strip obvious hashtag cues like `#WorldCup2022`, `#auspol`, `#covid19`, and `#chatgpt` from part of the corpus before sampling, turning the task into genuine topical clustering rather than hashtag recovery.

**The evaluation pipeline keeps more than scores.** `EvaluationResults` stores a metrics table in CSV and the corresponding embeddings and cluster assignments in compressed NPZ. This lets `analysis.ipynb` inspect failures, outliers, and metric correlations without recomputing expensive runs.

**Pipelines are compared end-to-end, not in isolation.** Text representation, dimensionality reduction, and clustering are all swappable stages. Results are tracked under stable composed labels like `distilroberta+umap5+hdbscan`, so it is always clear what combination produced a given score.

## Architecture

The system is notebook-driven with a small Python support layer:

1. **`generate_dataset.ipynb`** — loads raw topic CSVs from `data/source/`, normalizes schemas, cleans tweet text, removes near-duplicates with RapidFuzz, optionally strips topic markers, samples each topic, and outputs the final benchmark CSV.
2. **`clustering.ipynb`** — tokenizes tweets with spaCy (preserving hashtags and mentions as atomic tokens, normalizing links to `%link` and numbers to `%number`), builds embeddings from TF-IDF through sentence transformers, optionally reduces with Truncated SVD or UMAP, then runs one of K-Means, CLARA, HDBSCAN, Gaussian mixtures, or LDA.
3. **`evaluation.py`** — aligns arbitrary cluster IDs to ground truth with Hungarian matching, computes all metrics, and persists both summary results and intermediate artifacts.
4. **`analysis.ipynb`** — works entirely from saved results. Covers error concentration, outlier inspection, metric correlation, scatter plots, and LaTeX-ready tables.

## Methods overview

### Text embeddings

| Model | Notes |
|---|---|
| TF-IDF | Sparse bag-of-words baseline; words above 0.5 document frequency excluded |
| Word2Vec | Skip-gram, sentence embedding by averaged word vectors; good baseline without pretraining |
| DistilRoBERTa | 6-layer distilled RoBERTa (82M params); ~2× faster than RoBERTa-base; best overall result |
| MPNet | Combines BERT's MLM with XLNet's permuted LM; slower but stronger on small data |
| Universal Sentence Encoder | 512-dimensional sentence encoder; order-of-magnitude faster than transformers |

### Dimensionality reduction

| Method | Notes |
|---|---|
| Truncated SVD | Fast, linear — good for sparse TF-IDF matrices; cannot capture nonlinear structure |
| UMAP | Nonlinear manifold learning; consistently best results; `n_neighbors=15`, output size 5–15 |

### Clustering

| Method | Notes |
|---|---|
| K-Means | Fast, well-understood; sensitive to high dimensionality — combine with UMAP |
| CLARA | K-Medoids on random subsets; more robust to outliers than K-Means |
| HDBSCAN | Density-based; auto-detects cluster count; noise points reassigned to nearest cluster for fair evaluation |
| Gaussian Mixture Model | Soft clustering; competitive with HDBSCAN on UMAP-reduced embeddings |
| LDA | Probabilistic topic model; poor on short texts, included as baseline |

### Metrics

All reported metrics range from 0 to 1 (higher is better), except running time. The full set includes ARI, AMI, V-measure, homogeneity, completeness, accuracy, F1, minimum precision, and minimum recall. Cluster labels are aligned to ground truth via Hungarian matching before any classification metric is computed. Silhouette score is the single intrinsic metric retained (not comparable across embedding spaces).

## Dataset

The benchmark at `data/dataset70000.csv` contains 70,406 tweets across nine deliberately disjunctive topics:

| Topic | Samples |
|---|---:|
| climate change | 10,000 |
| australian elections | 10,000 |
| covid19 | 10,000 |
| chatgpt | 10,000 |
| stock market crash | 10,000 |
| airline support | 8,468 |
| fifa world cup | 6,854 |
| self-driving cars | 4,497 |
| weather | 587 |

The imbalance is intentional and reflects real source availability after cleaning. The topics were chosen to be mutually disjunctive — it is rare for a tweet to belong to two topics simultaneously, which reduces ground truth ambiguity.

### Cleaning pipeline

Raw source datasets were collected by hashtag without manual verification, so noise removal was essential. The cleaning steps applied per topic:

- **Length filtering** — topic-specific thresholds; minimum character counts vary from 30 to 70 depending on the source
- **Word fraction filtering** — removes tweets with too few alphabetic tokens (low information content)
- **Token length filtering** — tweets with mean token length outside (3, 10) are typically noise
- **Language detection** — Google's CLD2 (`pycld2`); tweets below confidence 800 are filtered
- **Near-duplicate removal** — pairwise RapidFuzz similarity; tweets with score > 0.8 to any subsequent tweet are removed
- **Retweet prefix stripping** — `rt @...` prefixes removed when retweets dominate a topic
- **Topic cue removal** — high-frequency topic hashtags randomly stripped from part of each topic slice to prevent trivial recovery

## Results snapshot

Among checked-in runs in `evaluation/dataset70000.csv`:

| Pipeline | ARI |
|---|---:|
| `distilroberta+umap5+hdbscan` | **0.914** |
| `distilroberta+umap5+gmm` | ~0.91 |
| `word2vec300+umap5+kmeans` | ~0.89 |
| `use+umap5+kmeans` | strong, fastest |
| `lda` | poor |

UMAP appears in every top result. The pattern holds across embedding families: raw or SVD-reduced embeddings consistently underperform UMAP-projected ones before clustering.

## Project structure

```
data/
  source/                  Raw topic-specific tweet datasets
  dataset70000.csv         Generated benchmark
evaluation/
  dataset70000.csv         Saved metrics for all checked-in runs
generate_dataset.ipynb     Topic curation and benchmark construction
clustering.ipynb           End-to-end pipeline benchmarking
analysis.ipynb             Post-hoc inspection, plots, thesis tables
evaluation.py              Metrics, label alignment, result persistence
utils.py                   IO, formatting, and timing helpers
deep_embedded_clustering.py  Standalone DEC / autoencoder experiment (not integrated)
thesis.pdf                 Full thesis
```

## Installation

```bash
uv venv --python 3.9
uv pip install -r misc/requirements.txt
uv run jupyter lab
```

Python 3.9.16. GPU execution is strongly recommended for transformer embeddings. The stack includes spaCy, sentence-transformers, TensorFlow Hub, PyTorch, HDBSCAN, and Jupyter.

## Usage

### Build or regenerate a dataset

Open `generate_dataset.ipynb`, adjust per-topic constants near the top, run the topic blocks you need, and save the merged dataset at the end. Each topic section is self-contained because cleaning rules differ by source.

### Benchmark clustering pipelines

Open `clustering.ipynb`, set `DATASET` to the target CSV, choose an embedding block, optionally apply a reduction block, then run one clustering block. Results are saved under a composed label via `EvaluationResults`.

### Analyze saved experiments

Open `analysis.ipynb` to inspect saved runs, compare metrics, visualize projection structure, find consistently misclustered tweets, and export LaTeX tables.

## Notes

- `deep_embedded_clustering.py` implements Deep Embedded Clustering (DEC) but is not integrated into the main benchmark. Training it to competitive ARI on this corpus proved difficult — likely an architecture sensitivity issue rather than a fundamental limitation of the approach.
- The `chatgpt` topic is in the benchmark but its source file `data/source/chatgpt.csv` is not checked in. Rebuilding that slice requires supplying the file yourself.
- Most project logic lives in notebooks deliberately: the real unit of work here is an experiment, not a package API.

## Quickstart

The fastest way to understand the system in practice is to open `clustering.ipynb`, load `data/dataset70000.csv`, and rerun the `distilroberta+umap5+hdbscan` pipeline end to end.
