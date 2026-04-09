"""
ingest_browser.py — Index browser history URLs via Trafilatura pipeline.

Reads URL history from the browser's SQLite database:
  Brave:  ~/.config/BraveSoftware/Brave-Browser/Default/History
  Chrome: ~/.config/google-chrome/Default/History
  Edge:   ~/.config/microsoft-edge/Default/History

For each URL:
  - Skip non-http/https, internal pages, duplicates
  - Fetch the live page
  - Run through ingest_html pipeline (Trafilatura → chunks → CLIP embed)
  - Failed fetches are logged and skipped

Run:
  python ingest_browser.py --browser brave --limit 100
  python ingest_browser.py --browser chrome --limit 500 --force
"""

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger
log = get_logger(__name__)

from embed import Embedder
from store import QdrantStore, collection_for_html
from bm25 import BM25Index
from ingest_html import fetch_html, index_html


# ── Browser history DB paths ──────────────────────────────────────────────────

BROWSER_PATHS = {
    "brave":  Path.home() / ".config/BraveSoftware/Brave-Browser/Default/History",
    "chrome": Path.home() / ".config/google-chrome/Default/History",
    "edge":   Path.home() / ".config/microsoft-edge/Default/History",
}

SKIP_SCHEMES   = {"chrome", "chrome-extension", "brave", "about", "file", "data"}
SKIP_HOSTS     = {"localhost", "127.0.0.1", "0.0.0.0", "newtab"}
SKIP_PATTERNS  = ["chrome://", "brave://", "chrome-extension://",
                  "about:", "data:", "javascript:"]

# Domains that are JS-rendered or auth-gated — trafilatura always fails on these.
# Skipping them upfront saves the URL budget for pages that can actually be extracted.
SKIP_DOMAINS   = {
    "youtube.com", "www.youtube.com", "youtu.be",
    "twitter.com", "x.com", "www.twitter.com",
    "instagram.com", "www.instagram.com",
    "facebook.com", "www.facebook.com",
    "reddit.com", "www.reddit.com",
    "tiktok.com", "www.tiktok.com",
    "linkedin.com", "www.linkedin.com",
    "chatgpt.com", "www.chatgpt.com",
    "claude.ai",
    "netflix.com", "www.netflix.com",
    "twitch.tv", "www.twitch.tv",
    "mail.google.com", "web.whatsapp.com",
    "calendar.google.com", "docs.google.com",
    "sheets.google.com", "drive.google.com",
}


def _should_skip(url: str) -> bool:
    """Return True if this URL should not be indexed."""
    if not url:
        return True
    if any(url.startswith(p) for p in SKIP_PATTERNS):
        return True
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return True
        host = p.hostname or ""
        if host in SKIP_HOSTS:
            return True
        # Skip JS-rendered / auth-gated domains that trafilatura can't extract
        if host in SKIP_DOMAINS:
            return True
        # Also match subdomains: mail.google.com → skip if google.com is listed
        # (not implemented — keep it simple, add to SKIP_DOMAINS as needed)
    except Exception:
        return True
    return False


def read_history(browser: str, limit: int) -> list[tuple[str, str]]:
    """
    Read URL history from browser SQLite database.
    Returns [(url, title)] sorted by most recent visit first.
    The DB is copied to a temp file first — the browser locks the original.
    """
    db_path = BROWSER_PATHS.get(browser.lower())
    if not db_path:
        raise ValueError(f"Unknown browser: {browser!r}. "
                         f"Supported: {list(BROWSER_PATHS)}")

    if not db_path.exists():
        raise FileNotFoundError(
            f"History DB not found: {db_path}\n"
            f"Is {browser} installed and has been used?"
        )

    # Copy to temp — browser keeps a lock on the original
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(db_path, tmp_path)

    try:
        con = sqlite3.connect(tmp_path)
        # Fetch more URLs than the target limit — many will be skipped
        # (JS-rendered, auth-gated, empty) before we reach 'limit' successes.
        fetch_limit = max(limit * 20, 200)
        rows = con.execute(
            """
            SELECT u.url, u.title
            FROM urls u
            JOIN visits v ON v.url = u.id
            WHERE u.url LIKE 'http%'
            GROUP BY u.url
            ORDER BY MAX(v.visit_time) DESC
            LIMIT ?
            """,
            (fetch_limit,),
        ).fetchall()
        con.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    # Deduplicate and filter
    seen = set()
    result = []
    for url, title in rows:
        if url in seen or _should_skip(url):
            continue
        seen.add(url)
        result.append((url, title or ""))

    return result


def index_browser_history(
    browser:  str,
    limit:    int,
    embedder: Embedder,
    store:    QdrantStore,
    bm25:     BM25Index,
    run_id:   str = "",
) -> int:
    """
    Read browser history → fetch each URL → Trafilatura → CLIP embed → Qdrant.
    Skips already-indexed URLs. Failed fetches are logged and skipped.
    Returns total points stored.
    """
    log.info("Reading browser history", browser=browser, limit=limit)
    urls = read_history(browser, limit)
    log.info("URLs to index", count=len(urls), browser=browser)

    total    = 0
    skipped  = 0
    failed   = 0

    successful = 0   # pages successfully downloaded AND content extracted

    for i, (url, title) in enumerate(urls):
        # Keep going until we have 'limit' successful extractions.
        # Failed fetches, auth errors, empty pages, and already-indexed URLs
        # do NOT count — we only stop when we have 'limit' real successes.
        if successful >= limit:
            log.info("Reached extraction limit", limit=limit)
            break

        # Already indexed — skip silently, don't count toward limit
        coll = collection_for_html(url)
        if store.point_count(coll) > 0:
            skipped += 1
            log.debug("Already indexed — skipping", url=url[:80])
            continue

        try:
            log.info("Fetching", url=url[:80],
                     progress=f"{successful+1}/{limit}")
            html_bytes = fetch_html(url, timeout=15)
            n = index_html(url, html_bytes, embedder, store, bm25)
            if n > 0:
                # Only count as successful if content was actually extracted and stored
                total += n
                successful += 1
                log.info("Indexed successfully",
                         url=url[:80], points=n, successful=successful)
            else:
                # Fetched but no content extracted (JS page, empty, etc.)
                failed += 1
                log.debug("No content extracted — not counting", url=url[:80])
        except Exception as e:
            failed += 1
            err_str = str(e)
            if any(code in err_str for code in ["401", "403", "Forbidden", "Unauthorized"]):
                log.debug("Auth-gated page — skipping", url=url[:80])
            else:
                log.warning("Fetch failed — skipping", url=url[:80], error=err_str)
            continue

    log.info("Browser history indexing complete",
             browser=browser, stored=total, skipped=skipped,
             failed=failed, successful=successful)
    return total


def main():
    parser = argparse.ArgumentParser(description="Index browser history into Qdrant")
    parser.add_argument("--browser", default="brave",
                        choices=list(BROWSER_PATHS),
                        help="Browser to read history from (default: brave)")
    parser.add_argument("--limit",  type=int, default=200,
                        help="Max URLs to index (default: 200)")
    parser.add_argument("--force",  action="store_true",
                        help="Re-index already-indexed URLs")
    args = parser.parse_args()

    embedder = Embedder()
    store    = QdrantStore()
    bm25     = BM25Index()
    bm25.load()

    total = index_browser_history(
        browser=args.browser,
        limit=args.limit,
        embedder=embedder,
        store=store,
        bm25=bm25,
    )

    pairs = store.scroll_text_chunks()
    if pairs:
        bm25.build_from(pairs)
        bm25.save()

    log.info("Done", total_stored=total, bm25_docs=bm25.doc_count)


if __name__ == "__main__":
    main()
