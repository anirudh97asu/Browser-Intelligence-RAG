# RAG Pipeline Evaluation

## Induction Hypothesis

If the retriever achieves measurable Recall@K and MRR on a human-annotated
benchmark, the same pipeline transfers to custom documents — because CLIP
embeddings, BM25, RRF, and CrossEncoder are corpus-agnostic.

## Run

```bash
cd /data/rag2

# Recommended: RAGBench techqa (technical QA, closest to your ML/RAG PDFs)
uv run python eval/run_hf_eval.py \
    --dataset rungalileo/ragbench \
    --config  techqa \
    --samples 100 \
    --top-k   5

# COVID QA (good for testing factual retrieval)
uv run python eval/run_hf_eval.py \
    --dataset rungalileo/ragbench \
    --config  covidqa \
    --samples 100

# Larger general RAG dataset (12k QA pairs)
uv run python eval/run_hf_eval.py \
    --dataset neural-bridge/rag-dataset-12000 \
    --samples 200

# Save results for later
uv run python eval/run_hf_eval.py \
    --dataset rungalileo/ragbench \
    --config  techqa \
    --samples 100 \
    --output  eval/results_techqa.json \
    --keep-index
```

## Output

```
============================================================
  RAG PIPELINE EVALUATION REPORT
============================================================
  Dataset:   rungalileo/ragbench / techqa
  Samples:   100 evaluated / 100 requested
  Timestamp: 2026-04-08 18:00

── Retrieval Metrics ────────────────────────────────────
  Recall@1:   0.61  (61/100 queries)
  Recall@3:   0.79  (79/100 queries)
  Recall@5:   0.85  (85/100 queries)
  MRR:        0.71
  NDCG@5:     0.74

── Answer Quality ───────────────────────────────────────
  Token F1:   0.56
  Exact Match:0.12

── Performance ──────────────────────────────────────────
  Avg latency:    1240 ms/query
  Median latency: 1100 ms/query
  Total time:     124.0 s
============================================================
```

## Metrics explained

| Metric | What it measures |
|--------|-----------------|
| Recall@K | Was the ground truth chunk in the top K retrieved? |
| MRR | Mean Reciprocal Rank — how high does the correct chunk rank on average? |
| NDCG@5 | Normalised Discounted Cumulative Gain — penalises correct chunks ranked lower |
| Token F1 | Overlap between system answer tokens and ground truth answer tokens |
| Exact Match | Strict: does the system answer exactly match ground truth? |

## Supported datasets

| Dataset | Config | Domain | Size |
|---------|--------|--------|------|
| `rungalileo/ragbench` | `techqa` | Technical QA | ~1k test |
| `rungalileo/ragbench` | `covidqa` | COVID/medical | ~500 test |
| `rungalileo/ragbench` | `hotpotqa` | Multi-hop QA | ~7k test |
| `neural-bridge/rag-dataset-12000` | — | General ML/AI | 12k total |
| `explodinggradients/amnesty_qa` | — | Legal/reports | ~20 test |
| `explodinggradients/fiqa` | — | Financial | ~648 test |
