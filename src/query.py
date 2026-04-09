"""
query.py — Query pipeline: embed → hybrid search → RRF → rerank → LLM answer.

Steps:
  1. Embed query text with CLIP (shared space — finds both text and image chunks)
  2. Dense vector search across all PDF + HTML collections (text vector space)
  3. BM25 keyword search
  4. RRF merge
  5. Cross-encoder rerank (BAAI/bge-reranker-v2-m3 or fallback score sort)
  6. LLM answer generation with top chunks as context

Run:
  python query.py "What is embedding drift?"
"""

import sys
import json
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger

log = get_logger(__name__)

from config import (
    OLLAMA_URL, LLM_MODEL,
    TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL, RRF_K,
)
from embed import Embedder
from store import QdrantStore
from bm25 import BM25Index
from utils import rrf_merge


# ── Reranker ──────────────────────────────────────────────────────────────────

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        log.info("Loading reranker")
        # Force CPU — GPU is occupied by CLIP (5.65GB VRAM, CLIP uses ~4GB)
        # CrossEncoder on CPU is fast enough for re-ranking top-20 candidates
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
        log.info("Reranker ready")
    except Exception as e:
        log.warning("Reranker unavailable — using score order", error=str(e))
        _reranker = "unavailable"
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    Cross-encoder rerank. candidates = [{"text": ..., ...}, ...]
    Returns top_k sorted by rerank score.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    if reranker == "unavailable" or reranker is None:
        return candidates[:top_k]

    pairs = [(query, c["text"]) for c in candidates]
    try:
        scores = reranker.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        log.error("Rerank error", error=str(e))
    finally:
        # Free GPU memory after reranker inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return candidates[:top_k]


# ── LLM answer ────────────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: list[dict]) -> tuple[str, float]:
    """
    Generate a grounded answer using the LLM with GPU acceleration.
    Returns (answer_text, duration_ms).

    GPU acceleration is achieved by:
      - num_gpu=-1  → Ollama uses ALL available GPU layers (offloads entire model)
      - num_thread=4 → minimal CPU threads (GPU does the heavy lifting)
      - stream=True  → collect tokens as they arrive, minimising time-to-first-token
    """
    import time
    if not chunks:
        return "No relevant content found.", 0.0

    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("file_path") or c.get("url") or "unknown"
        page   = f" (page {c['page_number']})" if c.get("page_number") else ""
        context_parts.append(f"[{i}] {source}{page}\n{c['text']}")

    context = "\n\n".join(context_parts)

    prompt = (
        "Answer the question below using ONLY the provided context. "
        "Be concise and factual. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    t0 = time.perf_counter()
    try:
        # stream=True: consume token chunks as they arrive — lower latency
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   LLM_MODEL,
                "prompt":  prompt,
                "stream":  True,
                "options": {
                    "temperature":  0.1,
                    "num_predict":  512,
                    "num_gpu":      -1,    # offload ALL layers to GPU
                    "num_thread":   4,     # minimal CPU threads when GPU is active
                    "num_ctx":      4096,  # context window
                },
            },
            stream=True,
            timeout=120,
        )
        tokens = []
        for line in r.iter_lines():
            if not line:
                continue
            try:
                import json as _json
                obj = _json.loads(line)
                tok = obj.get("response", "")
                if tok:
                    tokens.append(tok)
                if obj.get("done"):
                    break
            except Exception:
                continue
        answer = "".join(tokens).strip()
        duration_ms = (time.perf_counter() - t0) * 1000
        return answer or "No answer generated.", duration_ms
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        return f"LLM error: {e}", duration_ms


# ── Main query pipeline ───────────────────────────────────────────────────────

def query(
    query_text: str,
    embedder: Embedder,
    store: QdrantStore,
    bm25: BM25Index,
    top_k_vector: int = TOP_K_VECTOR,
    top_k_bm25: int = TOP_K_BM25,
    top_k_final: int = TOP_K_FINAL,
    search_pdfs:   bool = True,
    search_html:   bool = True,
    search_claude: bool = True,
) -> dict:
    """
    Full query pipeline. Returns:
    {
        "answer": str,
        "chunks": [{...}],
        "query": str,
        "retrieval_duration_ms": float,
        "generation_duration_ms": float,
        "total_chunks_searched": int,
    }

    Retrieval: top_k_vector candidates fetched independently from each modality
    (pdf / html / claude) so no source type is drowned out.  Six-way RRF then
    merges all ranked lists before reranking.
    """
    import time
    log.info("Query received", query=query_text[:80])

    t_retrieval_start = time.perf_counter()

    # 1. Embed query
    query_vec = embedder.embed_text(query_text)
    if query_vec is None:
        return {
            "answer": "Embedding failed.", "chunks": [], "query": query_text,
            "retrieval_duration_ms": 0.0, "generation_duration_ms": 0.0,
            "total_chunks_searched": 0,
        }

    # 2. Dense search — k per modality × vector-space (independent budgets)
    pdf_text_results    = store.search_all_pdf(query_vec,    "text",  top_k_vector) if search_pdfs   else []
    pdf_image_results   = store.search_all_pdf(query_vec,    "image", top_k_vector) if search_pdfs   else []
    html_text_results   = store.search_all_html(query_vec,   "text",  top_k_vector) if search_html   else []
    html_image_results  = store.search_all_html(query_vec,   "image", top_k_vector) if search_html   else []
    claude_text_results = store.search_all_claude(query_vec, "text",  top_k_vector) if search_claude else []

    def _sorted_ids(results):
        return [r["chunk_id"] for r in sorted(results, key=lambda x: x["score"], reverse=True)]

    pdf_text_ids    = _sorted_ids(pdf_text_results)
    pdf_image_ids   = _sorted_ids(pdf_image_results)
    html_text_ids   = _sorted_ids(html_text_results)
    html_image_ids  = _sorted_ids(html_image_results)
    claude_text_ids = _sorted_ids(claude_text_results)

    # Unified dense result map (deduplicated, first occurrence wins)
    all_dense = (pdf_text_results + pdf_image_results +
                 html_text_results + html_image_results + claude_text_results)
    seen_dense: set = set()
    dense_results: list = []
    for r in all_dense:
        if r["chunk_id"] not in seen_dense:
            seen_dense.add(r["chunk_id"])
            dense_results.append(r)

    total_chunks_searched = len(dense_results)
    log.debug("Dense hits per modality",
              pdf_text=len(pdf_text_ids), pdf_image=len(pdf_image_ids),
              html_text=len(html_text_ids), html_image=len(html_image_ids),
              claude=len(claude_text_ids))

    # 3. BM25 keyword search
    bm25_hits = bm25.search(query_text, k=top_k_bm25)
    bm25_ids  = [cid for cid, _ in bm25_hits]
    log.debug("BM25 hits", count=len(bm25_ids))

    # 4. Six-way RRF — one ranked list per (modality × vector-space) + BM25
    merged_ids = rrf_merge(
        [pdf_text_ids, pdf_image_ids,
         html_text_ids, html_image_ids,
         claude_text_ids, bm25_ids],
        k=RRF_K,
    )
    log.debug("RRF merged", count=len(merged_ids))

    # 5. Build candidate list with payloads
    payload_map     = {r["chunk_id"]: r["payload"] for r in dense_results}
    dense_score_map = {r["chunk_id"]: r["score"]   for r in dense_results}
    rrf_score_map   = {cid: 1.0 / (RRF_K + rank + 1) for rank, cid in enumerate(merged_ids)}

    bm25_only = [cid for cid in merged_ids if cid not in payload_map]
    if bm25_only:
        payload_map.update(store.fetch_payloads(bm25_only))

    candidates = []
    for cid in merged_ids[:top_k_vector]:
        payload      = payload_map.get(cid, {})
        text         = payload.get("text", "")
        content_type = payload.get("content_type", "text")
        if not text:
            if content_type in ("image_page", "image_embedded", "image"):
                text = payload.get("img_alt", "") or payload.get("filename", "") or "image"
            else:
                continue
        candidates.append({
            "chunk_id":         cid,
            "text":             text,
            "score":            rrf_score_map.get(cid, 0.0),
            "similarity_score": dense_score_map.get(cid, 0.0),
            "content_type":     content_type,
            "modality":         payload.get("modality", payload.get("source_type", "")),
            "source_id":        payload.get("source_id", payload.get("collection", "")),
            "source_path":      payload.get("source_path", payload.get("file_path", "")),
            "source_url":       payload.get("source_url", payload.get("url", "")),
            "source_type":      payload.get("source_type", ""),
            "filename":         payload.get("filename", ""),
            "file_path":        payload.get("file_path", ""),
            "page_number":      payload.get("page_number", 0),
            "bbox":             payload.get("bbox", []),
            "is_scanned":       payload.get("is_scanned", False),
            "ocr_failed":       payload.get("ocr_failed", False),
            "section_title":    payload.get("section_title", ""),
            "img_src":          payload.get("img_src", ""),
            "img_alt":          payload.get("img_alt", ""),
            "image_path":       payload.get("image_path", ""),
            "url":              payload.get("url", ""),
            "conversation_id":  payload.get("conversation_id", payload.get("conv_uuid", "")),
            "turn_index":       payload.get("turn_index", payload.get("turn_idx", 0)),
            "speaker_role":     payload.get("speaker_role", ""),
            "timestamp":        payload.get("timestamp", ""),
            "message_id":       payload.get("message_id", ""),
            "text_fingerprint": payload.get("text_fingerprint", ""),
            "collection":       payload.get("collection", ""),
        })

    # 6. Rerank text; keep image chunks separate, 2 per source type
    text_candidates  = [c for c in candidates if c["content_type"] == "text"]
    image_candidates = [c for c in candidates if c["content_type"] != "text"]

    reranked_text   = rerank(query_text, text_candidates, top_k=top_k_final)
    pdf_img_chunks  = [c for c in image_candidates if c.get("source_type") == "pdf"][:2]
    html_img_chunks = [c for c in image_candidates if c.get("source_type") == "html"][:2]
    final_chunks    = reranked_text + pdf_img_chunks + html_img_chunks

    retrieval_duration_ms = (time.perf_counter() - t_retrieval_start) * 1000
    log.debug("Final chunks", total=len(final_chunks),
              text=len(reranked_text), pdf_img=len(pdf_img_chunks),
              html_img=len(html_img_chunks))

    # 7. Generate answer — GPU-accelerated streaming
    answer, generation_duration_ms = generate_answer(query_text, reranked_text)
    log.debug("Answer generated", preview=answer[:80], gen_ms=round(generation_duration_ms))

    return {
        "answer":                 answer,
        "chunks":                 final_chunks,
        "query":                  query_text,
        "retrieval_duration_ms":  round(retrieval_duration_ms, 1),
        "generation_duration_ms": round(generation_duration_ms, 1),
        "total_chunks_searched":  total_chunks_searched,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)

    query_text = " ".join(sys.argv[1:])

    embedder = Embedder()
    store    = QdrantStore()
    bm25     = BM25Index()
    bm25.load()

    result = query(query_text, embedder, store, bm25)

    print("\n" + "="*60)
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for i, c in enumerate(result["chunks"], 1):
        src = c["file_path"] or c["url"]
        page = f" p{c['page_number']}" if c["page_number"] else ""
        print(f"  [{i}] {c['content_type']:15s} score={c['score']:.3f}  {src}{page}")
        if c["bbox"]:
            print(f"       bbox={[round(x,1) for x in c['bbox']]}")
    print("="*60)


if __name__ == "__main__":
    main()
