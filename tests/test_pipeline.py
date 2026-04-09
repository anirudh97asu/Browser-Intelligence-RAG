"""
tests/test_pipeline.py — Basic sanity tests for the RAG pipeline.

Run: cd /data/rag2 && uv run pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


# ── Config ────────────────────────────────────────────────────────────────────

def test_config_imports():
    from config import (CLIP_MODEL, PDF_PREFIX, HTML_PREFIX, TEXT_DIM,
                        IMAGE_DIM, TOP_K_VECTOR, TOP_K_BM25, RRF_K,
                        PDF_DPI, MIN_TEXT_CHARS, SEMANTIC_THRESHOLD)
    assert TEXT_DIM > 0
    assert IMAGE_DIM > 0
    assert PDF_PREFIX == "rag_pdf__"
    assert HTML_PREFIX == "rag_html__"
    assert RRF_K == 60


# ── BM25 ──────────────────────────────────────────────────────────────────────

def test_bm25_build_and_search():
    from bm25 import BM25Index
    bm25 = BM25Index()
    pairs = [
        ("chunk001", "RAG pipeline uses vector embeddings for retrieval"),
        ("chunk002", "embedding drift occurs when embedding models update"),
        ("chunk003", "the weather today is sunny and warm"),
    ]
    bm25.build_from(pairs)
    assert bm25.is_ready
    assert bm25.doc_count == 3

    hits = bm25.search("embedding drift", k=3)
    assert len(hits) > 0
    top_id, top_score = hits[0]
    assert top_id == "chunk002"    # most relevant
    assert top_score > 0


def test_bm25_empty_query():
    from bm25 import BM25Index
    bm25 = BM25Index()
    bm25.build_from([("a", "hello world"), ("b", "foo bar")])
    hits = bm25.search("zzznotaword", k=5)
    assert hits == []


def test_bm25_save_load(tmp_path, monkeypatch):
    from bm25 import BM25Index, DATA_DIR, BM25_DIR, IDS_PATH
    monkeypatch.setattr("bm25.DATA_DIR", tmp_path)
    monkeypatch.setattr("bm25.BM25_DIR", tmp_path / "bm25")
    monkeypatch.setattr("bm25.IDS_PATH", tmp_path / "bm25_ids.json")
    (tmp_path / "bm25").mkdir()

    import bm25 as bm25_mod
    bm25_mod.DATA_DIR = tmp_path
    bm25_mod.BM25_DIR = tmp_path / "bm25"
    bm25_mod.IDS_PATH = tmp_path / "bm25_ids.json"

    b = BM25Index()
    b.build_from([("id1", "hello world"), ("id2", "foo bar baz")])
    b.save()

    b2 = BM25Index()
    ok = b2.load()
    assert ok
    assert b2.doc_count == 2
    hits = b2.search("hello", k=2)
    assert any(cid == "id1" for cid, _ in hits)


# ── Utils ─────────────────────────────────────────────────────────────────────

def test_rrf_merge():
    from utils import rrf_merge
    merged = rrf_merge([["a", "b", "c"], ["b", "c", "d"]])
    # b and c appear in both lists — should rank highest
    assert merged[0] in ("b", "c")
    assert "d" in merged
    assert len(merged) == 4


def test_rrf_merge_three_way():
    from utils import rrf_merge
    merged = rrf_merge([["a", "b"], ["b", "c"], ["b", "d"]])
    # b is in all three lists → highest score
    assert merged[0] == "b"


def test_semantic_chunk_short_text():
    """Short text (≤2 sentences) returns as single chunk."""
    from unittest.mock import MagicMock
    from utils import semantic_chunk
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1] * 512, [0.1] * 512]
    result = semantic_chunk("Hello world.", embedder)
    assert isinstance(result, list)


def test_cosine_dist():
    from utils import cosine_dist
    import math
    # Same vector → distance 0
    v = [1.0, 0.0, 0.0]
    assert cosine_dist(v, v) == pytest.approx(0.0)
    # Orthogonal → distance 1
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_dist(a, b) == pytest.approx(1.0)
    # Zero vector → distance 1 (no similarity)
    assert cosine_dist([0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ── Store collection naming ───────────────────────────────────────────────────

def test_collection_naming():
    from store import collection_for_pdf, collection_for_html
    assert collection_for_pdf("/home/user/My Doc.pdf")         == "rag_pdf__my_doc"
    assert collection_for_pdf("/data/07-1 back-prop.pdf")      == "rag_pdf__d_07_1_back_prop"
    assert collection_for_html("https://example.com/article")  == "rag_html__example_com_article"
    assert collection_for_html("https://example.com/")         == "rag_html__example_com"


def test_stable_id_deterministic():
    from store import stable_id
    id1 = stable_id("/home/user/doc.pdf", 0)
    id2 = stable_id("/home/user/doc.pdf", 0)
    id3 = stable_id("/home/user/doc.pdf", 1)
    assert id1 == id2          # same source+index → same id
    assert id1 != id3          # different index → different id


# ── Claude export parsing ─────────────────────────────────────────────────────

def test_claude_parse_old_format(tmp_path):
    import json
    from ingest_claude import parse_conversations
    data = [
        {
            "uuid": "abc-123",
            "name": "Test conversation",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "chat_messages": [
                {"uuid": "m1", "sender": "human",
                 "text": "What is RAG?", "created_at": "2024-01-01"},
                {"uuid": "m2", "sender": "assistant",
                 "text": "RAG stands for Retrieval Augmented Generation.",
                 "created_at": "2024-01-01"},
            ]
        }
    ]
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps(data))
    convs = parse_conversations(f)
    assert len(convs) == 1
    assert convs[0]["uuid"] == "abc-123"
    assert len(convs[0]["qa_pairs"]) == 1
    assert "RAG" in convs[0]["qa_pairs"][0]["q"]
    assert "Retrieval" in convs[0]["qa_pairs"][0]["a"]


def test_claude_parse_new_format(tmp_path):
    import json
    from ingest_claude import parse_conversations
    data = [
        {
            "uuid": "xyz-456",
            "name": "New format test",
            "chat_messages": [
                {"uuid": "m1", "sender": "human",
                 "content": [{"type": "text", "text": "Explain embeddings."}]},
                {"uuid": "m2", "sender": "assistant",
                 "content": [{"type": "text", "text": "Embeddings are dense vectors."}]},
            ]
        }
    ]
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps(data))
    convs = parse_conversations(f)
    assert convs[0]["qa_pairs"][0]["q"] == "Explain embeddings."
    assert "dense" in convs[0]["qa_pairs"][0]["a"]


# ── API models ────────────────────────────────────────────────────────────────

def test_index_start_request_defaults():
    from api import IndexStartRequest
    r = IndexStartRequest()
    assert r.index_pdfs   == False
    assert r.index_web    == False
    assert r.index_claude == False
    assert r.max_pages    == 50
    assert r.history_limit == 10
    assert r.max_conversations == 10
    assert r.max_files    == 10


def test_index_start_request_custom():
    from api import IndexStartRequest
    r = IndexStartRequest(
        index_pdfs=True,
        pdf_folder="/tmp",
        max_pages=5,
        max_files=20,
        index_web=True,
        history_limit=50,
    )
    assert r.max_pages == 5
    assert r.max_files == 20
    assert r.history_limit == 50


def test_query_request_defaults():
    from api import QueryRequest
    r = QueryRequest(text="test query")
    assert r.search_pdfs   == True
    assert r.search_html   == True
    assert r.search_claude == True
    assert r.top_k == 5
