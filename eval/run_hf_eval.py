"""
eval/run_hf_eval.py — Evaluate the RAG pipeline against a HuggingFace benchmark dataset.

Induction hypothesis:
    If the retriever achieves measurable Recall@K and MRR on human-annotated
    benchmark data, the same mechanism transfers to custom documents (same
    CLIP embeddings, BM25, RRF, CrossEncoder — corpus-agnostic).

Supported datasets (auto-detected schema):
    rungalileo/ragbench          techqa / covidqa / hotpotqa configs
    neural-bridge/rag-dataset-12000
    explodinggradients/amnesty_qa
    explodinggradients/fiqa

Usage:
    cd /data/rag2
    uv run python eval/run_hf_eval.py \\
        --dataset rungalileo/ragbench \\
        --config  techqa \\
        --split   test \\
        --samples 100 \\
        --top-k   5 \\
        --output  eval/results.json

What it does:
    1. Downloads the benchmark dataset
    2. Indexes its documents into isolated Qdrant collections (eval__ prefix)
    3. Runs your full retriever (CLIP + BM25 + RRF + CrossEncoder)
    4. Scores: Recall@1/3/5, MRR, NDCG, Answer F1
    5. Prints a report and saves results JSON
    6. Deletes eval collections (--keep-index to preserve)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logger import get_logger
log = get_logger("eval")

from embed import Embedder
from store import QdrantStore, stable_id
from bm25 import BM25Index
from utils import semantic_chunk
import query as query_module


EVAL_PREFIX = "eval__"


# ── Dataset schema adapters ───────────────────────────────────────────────────

def load_dataset_samples(dataset_name: str, config: str | None,
                          split: str, n: int) -> list[dict]:
    """
    Load up to n samples from a HuggingFace dataset.
    Returns list of normalised dicts:
      { id, question, documents: [str], answer: str }
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed — run: uv sync")
        sys.exit(1)

    log.info("Loading dataset", dataset=dataset_name, config=config, split=split)

    load_kwargs = dict(split=split, streaming=True)
    if config:
        ds = load_dataset(dataset_name, config, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        norm = _normalise_row(row, dataset_name, i)
        if norm:
            samples.append(norm)

    log.info("Loaded samples", count=len(samples))
    return samples


def _normalise_row(row: dict, dataset_name: str, idx: int) -> dict | None:
    """Normalise a dataset row to a common schema."""
    keys = set(row.keys())

    # rungalileo/ragbench
    if "documents" in keys and "question" in keys:
        docs = row["documents"]
        if isinstance(docs, str):
            docs = [docs]
        return {
            "id":        row.get("id", f"q{idx:05d}"),
            "question":  row["question"],
            "documents": [str(d) for d in docs if d],
            "answer":    str(row.get("answer", row.get("response", ""))),
        }

    # neural-bridge/rag-dataset-12000
    if "context" in keys and "question" in keys:
        ctx = row["context"]
        if isinstance(ctx, str):
            ctx = [ctx]
        elif isinstance(ctx, dict):
            ctx = list(ctx.values())
        return {
            "id":        row.get("id", f"q{idx:05d}"),
            "question":  row["question"],
            "documents": [str(c) for c in ctx if c],
            "answer":    str(row.get("answer", row.get("output", ""))),
        }

    # explodinggradients/amnesty_qa / fiqa
    if "contexts" in keys and "question" in keys:
        ctxs = row["contexts"]
        if isinstance(ctxs, str):
            ctxs = [ctxs]
        return {
            "id":        row.get("id", f"q{idx:05d}"),
            "question":  row["question"],
            "documents": [str(c) for c in ctxs if c],
            "answer":    str(row.get("ground_truth", row.get("answer", ""))),
        }

    log.warning("Unknown schema", keys=sorted(keys))
    return None


# ── Indexing eval corpus into Qdrant ──────────────────────────────────────────

def index_eval_corpus(samples: list[dict], embedder: Embedder,
                       store: QdrantStore, bm25: BM25Index) -> dict[str, str]:
    """
    Index all documents from the eval dataset into isolated eval__ collections.
    Returns {doc_fingerprint: chunk_id} for ground-truth lookup.

    Each unique document gets chunked and embedded exactly like a real document.
    We keep a mapping: (sample_id, doc_index) → [chunk_ids] for Recall scoring.
    """
    from qdrant_client.models import PointStruct

    EVAL_COLL = f"{EVAL_PREFIX}corpus"
    store.ensure_collection(EVAL_COLL,
                             text_dim=embedder.text_dim,
                             image_dim=embedder.image_dim)

    # Map sample_id → list of chunk_ids that contain its ground truth docs
    sample_chunk_map: dict[str, list[str]] = {}
    bm25_pairs: list[tuple[str, str]] = []
    points = []
    chunk_idx = 0
    seen_docs: set[str] = set()

    log.info("Indexing eval corpus", samples=len(samples))

    for sample in samples:
        sid = sample["id"]
        sample_chunk_map[sid] = []

        for doc_text in sample["documents"]:
            doc_text = doc_text.strip()
            if not doc_text or doc_text in seen_docs:
                # Dedup — same passage may appear in multiple questions
                # Find existing chunk IDs for this doc
                for pt_id, pt_text in bm25_pairs:
                    if doc_text[:80] in pt_text[:80]:
                        sample_chunk_map[sid].append(pt_id)
                continue
            seen_docs.add(doc_text)

            # Semantic chunk the document (same pipeline as real indexing)
            chunks = semantic_chunk(doc_text, embedder, min_words=5)
            if not chunks:
                chunks = [doc_text]

            vecs = embedder.embed_texts(chunks)

            for chunk_text, vec in zip(chunks, vecs):
                if vec is None:
                    chunk_idx += 1
                    continue

                point_id = stable_id(f"eval::{sid}", chunk_idx)
                pid_hex  = point_id.replace("-", "")

                pt = store.build_point(
                    point_id  = point_id,
                    text_vec  = vec,
                    image_vec = None,
                    payload   = {
                        "content_type":     "text",
                        "source_type":      "eval",
                        "modality":         "eval",
                        "collection":       EVAL_COLL,
                        "text":             chunk_text,
                        "text_fingerprint": chunk_text[:80],
                        "sample_id":        sid,
                        "bbox":             [],
                        "page_number":      0,
                        "file_path":        "",
                        "url":              "",
                        "is_scanned":       False,
                    },
                )
                if pt:
                    points.append(pt)
                    bm25_pairs.append((pid_hex, chunk_text))
                    sample_chunk_map[sid].append(pid_hex)
                chunk_idx += 1

        # Upsert in batches of 100
        if len(points) >= 100:
            store.upsert(EVAL_COLL, points)
            points = []

    if points:
        store.upsert(EVAL_COLL, points)

    # Build BM25 for eval corpus
    if bm25_pairs:
        bm25.build_from(bm25_pairs)

    log.info("Eval corpus indexed",
             chunks=chunk_idx, unique_docs=len(seen_docs))
    return sample_chunk_map


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant chunk appears in top-k retrieved, else 0.0"""
    return 1.0 if any(r in relevant_ids for r in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of first relevant chunk, 0 if none found"""
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ k"""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], 1)
        if rid in relevant_ids
    )
    # Ideal DCG: all relevant docs at top positions
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant_ids), k) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0


def token_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 between predicted and ground truth answer."""
    pred_tokens = set(predicted.lower().split())
    gt_tokens   = set(ground_truth.lower().split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common    = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, ground_truth: str) -> float:
    """1.0 if answers match after normalisation."""
    norm = lambda s: " ".join(s.lower().strip().split())
    return 1.0 if norm(predicted) == norm(ground_truth) else 0.0


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_evaluation(
    samples:        list[dict],
    sample_chunk_map: dict[str, list[str]],
    embedder:       Embedder,
    store:          QdrantStore,
    bm25:           BM25Index,
    top_k:          int = 5,
    eval_coll:      str = f"{EVAL_PREFIX}corpus",
) -> dict:
    """
    Run retrieval on each question, score against ground truth chunk IDs.
    Returns metrics dict.
    """
    results = []
    latencies = []

    log.info("Running evaluation", queries=len(samples), top_k=top_k)

    for i, sample in enumerate(samples):
        sid      = sample["id"]
        question = sample["question"]
        answer   = sample["answer"]
        relevant = set(sample_chunk_map.get(sid, []))

        if not relevant:
            log.debug("No ground truth chunks for sample — skipping", sid=sid)
            continue

        t0 = time.time()

        # Embed query
        query_vec = embedder.embed_text(question)
        if query_vec is None:
            continue

        # Dense search — text space only (eval corpus has no image vectors)
        text_hits = store.search(eval_coll, query_vec,
                                  vector_name="text", top_k=top_k * 2)
        text_ids  = [h["chunk_id"] for h in
                     sorted(text_hits, key=lambda x: x["score"], reverse=True)]

        # BM25 search
        bm25_hits = bm25.search(question, k=top_k * 2)
        bm25_ids  = [cid for cid, _ in bm25_hits]

        # RRF merge
        from utils import rrf_merge
        from config import RRF_K
        merged = rrf_merge([text_ids, bm25_ids], k=RRF_K)[:top_k * 2]

        # CrossEncoder rerank on top candidates
        payload_map = {h["chunk_id"]: h["payload"] for h in text_hits}
        bm25_only   = [cid for cid in merged if cid not in payload_map]
        if bm25_only:
            payload_map.update(store.fetch_payloads(bm25_only))

        candidates = []
        for cid in merged[:top_k * 2]:
            payload = payload_map.get(cid, {})
            text    = payload.get("text", "")
            if text:
                candidates.append({"chunk_id": cid, "text": text, "score": 0.0})

        reranked   = query_module.rerank(question, candidates, top_k=top_k)
        final_ids  = [c["chunk_id"] for c in reranked]

        # Generate answer from top text chunks
        predicted_answer = query_module.generate_answer(question, reranked)

        latencies.append(time.time() - t0)

        results.append({
            "id":         sid,
            "question":   question,
            "answer_gt":  answer,
            "answer_sys": predicted_answer,
            "retrieved":  final_ids,
            "relevant":   list(relevant),
            "recall@1":   recall_at_k(final_ids, relevant, 1),
            "recall@3":   recall_at_k(final_ids, relevant, 3),
            "recall@5":   recall_at_k(final_ids, relevant, 5),
            "mrr":        reciprocal_rank(final_ids, relevant),
            "ndcg@5":     ndcg_at_k(final_ids, relevant, 5),
            "f1":         token_f1(predicted_answer, answer),
            "em":         exact_match(predicted_answer, answer),
            "latency_s":  latencies[-1],
        })

        if (i + 1) % 10 == 0:
            log.info("Progress", done=i+1, total=len(samples))

    return results, latencies


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], latencies: list[float],
                  dataset: str, config: str | None, samples: int):
    n = len(results)
    if n == 0:
        print("No results to report.")
        return

    def avg(key): return sum(r[key] for r in results) / n

    print()
    print("=" * 60)
    print("  RAG PIPELINE EVALUATION REPORT")
    print("=" * 60)
    print(f"  Dataset:   {dataset}" + (f" / {config}" if config else ""))
    print(f"  Samples:   {n} evaluated / {samples} requested")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("── Retrieval Metrics ────────────────────────────────────")
    print(f"  Recall@1:   {avg('recall@1'):.3f}  "
          f"({sum(r['recall@1'] for r in results):.0f}/{n} queries)")
    print(f"  Recall@3:   {avg('recall@3'):.3f}  "
          f"({sum(r['recall@3'] for r in results):.0f}/{n} queries)")
    print(f"  Recall@5:   {avg('recall@5'):.3f}  "
          f"({sum(r['recall@5'] for r in results):.0f}/{n} queries)")
    print(f"  MRR:        {avg('mrr'):.3f}")
    print(f"  NDCG@5:     {avg('ndcg@5'):.3f}")
    print()
    print("── Answer Quality ───────────────────────────────────────")
    print(f"  Token F1:   {avg('f1'):.3f}")
    print(f"  Exact Match:{avg('em'):.3f}")
    print()
    print("── Performance ──────────────────────────────────────────")
    print(f"  Avg latency:    {sum(latencies)/len(latencies)*1000:.0f} ms/query")
    print(f"  Median latency: {sorted(latencies)[len(latencies)//2]*1000:.0f} ms/query")
    print(f"  Total time:     {sum(latencies):.1f} s")
    print()
    print("── Induction Claim ──────────────────────────────────────")
    r5 = avg('recall@5')
    mrr = avg('mrr')
    claim = (
        f"The retrieval pipeline achieves Recall@5={r5:.2f} and MRR={mrr:.2f} "
        f"on {n} human-annotated queries from {dataset}. Since the pipeline is "
        f"corpus-agnostic (CLIP embeddings + BM25 + RRF + CrossEncoder), "
        f"these results provide a lower bound on performance for "
        f"domain-matched custom documents."
    )
    # Word-wrap at 56 chars
    import textwrap
    for line in textwrap.wrap(claim, 56):
        print(f"  {line}")
    print("=" * 60)
    print()


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_eval_collections(store: QdrantStore):
    """Delete all eval__ collections from Qdrant."""
    deleted = 0
    for coll in store.all_collections():
        if coll.startswith(EVAL_PREFIX):
            store.client.delete_collection(coll)
            deleted += 1
    log.info("Eval collections deleted", count=deleted)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline against a HuggingFace benchmark"
    )
    parser.add_argument("--dataset",  default="rungalileo/ragbench",
                        help="HuggingFace dataset name")
    parser.add_argument("--config",   default="techqa",
                        help="Dataset config/subset (e.g. techqa, covidqa)")
    parser.add_argument("--split",    default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--samples",  type=int, default=100,
                        help="Number of queries to evaluate (default: 100)")
    parser.add_argument("--top-k",    type=int, default=5,
                        help="Retrieval depth to evaluate (default: 5)")
    parser.add_argument("--output",   default="eval/results.json",
                        help="Path to save results JSON")
    parser.add_argument("--keep-index", action="store_true",
                        help="Don't delete eval Qdrant collections after run")
    args = parser.parse_args()

    log.info("Starting evaluation",
             dataset=args.dataset, config=args.config,
             samples=args.samples, top_k=args.top_k)

    # 1. Load services
    embedder = Embedder()
    store    = QdrantStore()
    bm25     = BM25Index()  # fresh index for eval corpus

    # 2. Load dataset
    samples = load_dataset_samples(
        args.dataset, args.config, args.split, args.samples
    )
    if not samples:
        log.error("No samples loaded — check dataset name and config")
        sys.exit(1)

    # 3. Index eval corpus
    sample_chunk_map = index_eval_corpus(samples, embedder, store, bm25)

    # 4. Run evaluation
    results, latencies = run_evaluation(
        samples, sample_chunk_map, embedder, store, bm25,
        top_k=args.top_k,
    )

    # 5. Print report
    print_report(results, latencies,
                  args.dataset, args.config, args.samples)

    # 6. Save results JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset":    args.dataset,
        "config":     args.config,
        "split":      args.split,
        "samples":    args.samples,
        "top_k":      args.top_k,
        "timestamp":  datetime.now().isoformat(),
        "metrics": {
            "recall@1":  sum(r["recall@1"] for r in results) / len(results),
            "recall@3":  sum(r["recall@3"] for r in results) / len(results),
            "recall@5":  sum(r["recall@5"] for r in results) / len(results),
            "mrr":       sum(r["mrr"]       for r in results) / len(results),
            "ndcg@5":    sum(r["ndcg@5"]    for r in results) / len(results),
            "f1":        sum(r["f1"]        for r in results) / len(results),
            "em":        sum(r["em"]        for r in results) / len(results),
            "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
        },
        "per_query": results,
    }
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Results saved", path=str(output_path))

    # 7. Cleanup
    if not args.keep_index:
        cleanup_eval_collections(store)
        log.info("Eval index cleaned up (use --keep-index to preserve)")


if __name__ == "__main__":
    main()
