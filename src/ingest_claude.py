"""
ingest_claude.py — Index Claude conversation exports into Qdrant.

Claude exports a conversations.json file with this structure:
[
  {
    "uuid": "...",
    "name": "conversation title",
    "created_at": "...",
    "updated_at": "...",
    "chat_messages": [
      {
        "uuid": "...",
        "sender": "human" | "assistant",
        "text": "...",
        "created_at": "...",
        "content": [{"type": "text", "text": "..."}]  ← newer export format
      }
    ]
  }
]

Pipeline per conversation:
  - Pair human+assistant turns into Q+A units (kept atomic)
  - Each Q+A pair → semantic chunk → CLIP text embed
  - Stored under collection: rag_claude__<conversation_uuid[:16]>
  - URL key: claude://conversation/<uuid>

Run:
  python ingest_claude.py --path /home/user/Downloads/conversations.json
  python ingest_claude.py --path /home/user/Downloads/conversations.json --limit 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger
log = get_logger(__name__)

from embed import Embedder
from store import QdrantStore, stable_id
from bm25 import BM25Index
from utils import semantic_chunk
from config import HTML_PREFIX


CLAUDE_PREFIX = "rag_claude__"


def collection_for_claude(conv_uuid: str) -> str:
    safe = conv_uuid.replace("-", "")[:24]
    return CLAUDE_PREFIX + safe


def _extract_text(message: dict) -> str:
    """
    Extract plain text from a chat message.
    Handles both old format (text: str) and new format (content: [{type, text}]).
    """
    # New format: content array
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "").strip()
                if t:
                    parts.append(t)
        if parts:
            return " ".join(parts)

    # Old format: text field
    text = message.get("text", "").strip()
    return text


def _extract_attachments(message: dict) -> list[dict]:
    """
    Extract file/image attachments from a chat message.
    Returns list of {name, type, content} for text attachments
    and {name, type, b64} for image attachments.
    Claude export attachments format:
      "attachments": [{"file_name": "...", "file_type": "...", "extracted_content": "..."}]
      "files": [{"file_name": "...", "file_type": "...", "preview_url": "..."}]
    """
    result = []
    for key in ("attachments", "files"):
        for att in message.get(key, []) or []:
            name      = att.get("file_name", "")
            ftype     = att.get("file_type", "").lower()
            content   = att.get("extracted_content", "").strip()
            if content and ftype not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
                result.append({"name": name, "type": "text", "content": content})
    return result


def parse_conversations(json_path: Path, limit: int | None = None) -> list[dict]:
    """
    Parse conversations.json. Returns list of conversation dicts.
    Each dict: {uuid, name, url, qa_pairs: [{q, a, turn_idx}]}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of conversations")

    if limit:
        data = data[:limit]

    conversations = []
    for conv in data:
        uuid  = conv.get("uuid", "unknown")
        name  = conv.get("name", "Untitled conversation")
        msgs  = conv.get("chat_messages", [])
        url   = f"claude://conversation/{uuid}"

        # Build Q+A pairs — pair each human turn with the following assistant turn
        qa_pairs = []
        i = 0
        while i < len(msgs):
            msg    = msgs[i]
            sender = msg.get("sender", "")
            text   = _extract_text(msg)

            if sender == "human" and text:
                attachments = _extract_attachments(msg)
                att_text    = " ".join(a["content"] for a in attachments if a["type"] == "text")
                full_q      = (text + " " + att_text).strip() if att_text else text

                # Capture human turn metadata
                q_msg_id    = msg.get("uuid", "")
                q_timestamp = msg.get("created_at", "")

                # Look for next assistant reply
                answer      = ""
                a_msg_id    = ""
                a_timestamp = ""
                if i + 1 < len(msgs) and msgs[i + 1].get("sender") == "assistant":
                    answer      = _extract_text(msgs[i + 1])
                    a_msg_id    = msgs[i + 1].get("uuid", "")
                    a_timestamp = msgs[i + 1].get("created_at", "")
                    i += 2
                else:
                    i += 1

                if full_q or answer:
                    qa_pairs.append({
                        "q":            full_q,
                        "a":            answer,
                        "turn_idx":     len(qa_pairs),
                        "q_msg_id":     q_msg_id,
                        "a_msg_id":     a_msg_id,
                        "q_timestamp":  q_timestamp,
                        "a_timestamp":  a_timestamp,
                    })
            else:
                i += 1

        if qa_pairs:
            conversations.append({
                "uuid":     uuid,
                "name":     name,
                "url":      url,
                "qa_pairs": qa_pairs,
            })

    return conversations


def index_claude(
    json_path: Path,
    embedder:  Embedder,
    store:     QdrantStore,
    bm25:      BM25Index,
    limit:     int | None = 10,   # default max 10 conversations
) -> int:
    """
    Index all conversations from a Claude export file.
    Returns total points stored.
    """
    log.info("Indexing Claude chats", path=json_path.name)

    # Let FileNotFoundError propagate — caller (api.py) will record it as an error
    if not json_path.exists():
        raise FileNotFoundError(
            f"conversations.json not found: {json_path}\n"
            f"Export your Claude chats from claude.ai → Settings → Export Data"
        )
    try:
        conversations = parse_conversations(json_path, limit=limit)
    except FileNotFoundError:
        raise   # re-raise so api.py records it
    except Exception as e:
        log.error("Failed to parse Claude export", path=str(json_path), error=str(e))
        return 0

    log.info("Conversations found", count=len(conversations))
    total = 0

    for conv in conversations:
        uuid     = conv["uuid"]
        name     = conv["name"]
        url      = conv["url"]
        qa_pairs = conv["qa_pairs"]
        coll     = collection_for_claude(uuid)

        store.ensure_collection(
            coll,
            text_dim=embedder.text_dim,
            image_dim=embedder.image_dim,
        )

        if store.point_count(coll) > 0:
            log.info("Conversation already indexed — skipping", uuid=uuid[:8])
            continue

        points    = []
        chunk_idx = 0

        for qa in qa_pairs:
            # Keep Q+A atomic — index as a single unit
            # This ensures the question and answer are always retrieved together
            combined = ""
            if qa["q"]:
                combined += f"Q: {qa['q']}"
            if qa["a"]:
                combined += f"\nA: {qa['a']}"
            if not combined.strip():
                continue

            # Semantic chunk in case Q+A is very long
            chunks = semantic_chunk(combined, embedder)
            vecs   = embedder.embed_texts(chunks)

            for chunk_text, vec in zip(chunks, vecs):
                if vec is None:
                    chunk_idx += 1
                    continue

                point_id = stable_id(url, chunk_idx)
                pt = store.build_point(
                    point_id  = point_id,
                    text_vec  = vec,
                    image_vec = None,
                    payload   = {
                        "modality":         "claude_chat_json",
                        "content_type":     "text",
                        "source_type":      "claude",
                        "source_id":        coll,
                        "source_path":      str(json_path),
                        "source_url":       url,
                        "url":              url,
                        "filename":         json_path.name,
                        "text":             chunk_text,
                        "text_fingerprint": chunk_text[:80],
                        "img_src":          "",
                        "img_alt":          "",
                        "collection":       coll,
                        "conv_name":        name,
                        "conv_uuid":        uuid,
                        "conversation_id":  uuid,
                        "turn_idx":         qa["turn_idx"],
                        "turn_index":       qa["turn_idx"],
                        "message_id":       qa.get("q_msg_id", ""),
                        "speaker_role":     "human+assistant",
                        "timestamp":        qa.get("q_timestamp", ""),
                        "page_number":      0,
                        "bbox":             [],
                        "section_title":    "",
                        "image_path":       "",
                        "is_scanned":       False,
                    },
                    text_dim  = embedder.text_dim,
                    image_dim = embedder.image_dim,
                )
                if pt:
                    points.append(pt)
                chunk_idx += 1

        stored = store.upsert(coll, points)
        log.info("Conversation indexed", name=name[:50], points=stored)
        total += stored

    return total


def main():
    parser = argparse.ArgumentParser(description="Index Claude chat export into Qdrant")
    parser.add_argument("--path",  required=True, help="Path to conversations.json")
    parser.add_argument("--limit", type=int,       help="Max conversations to index")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing collections and re-index")
    args = parser.parse_args()

    json_path = Path(args.path)
    if not json_path.exists():
        log.error("File not found", path=args.path)
        sys.exit(1)

    embedder = Embedder()
    store    = QdrantStore()
    bm25     = BM25Index()
    bm25.load()

    if args.force:
        # Wipe all claude collections
        for coll in store.all_collections():
            if coll.startswith(CLAUDE_PREFIX):
                store.client.delete_collection(coll)
                log.info("Deleted collection", coll=coll)

    total = index_claude(json_path, embedder, store, bm25, limit=args.limit)

    pairs = store.scroll_text_chunks()
    if pairs:
        bm25.build_from(pairs)
        bm25.save()

    log.info("Done", total_stored=total, bm25_docs=bm25.doc_count)


if __name__ == "__main__":
    main()
