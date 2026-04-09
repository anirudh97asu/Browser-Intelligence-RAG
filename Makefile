# Browser RAG 2 - Makefile
# Requires: uv  https://docs.astral.sh/uv/getting-started/installation/
#   curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: help up down restart logs install setup server \
        index-pdf index-pdf-force index-pdf-folder index-url \
        query health clean reset

PDF    ?=
URL    ?=
QUERY  ?= What is RAG?
FOLDER ?=
SRC     = src

help:
	@echo ""
	@echo "  Browser RAG 2"
	@echo ""
	@echo "  Setup"
	@echo "    make install          uv sync"
	@echo "    make setup            install + CLIP + spaCy"
	@echo ""
	@echo "  Docker"
	@echo "    make up               docker compose up -d"
	@echo "    make down             docker compose down"
	@echo "    make restart          docker compose restart"
	@echo "    make logs             tail docker logs"
	@echo ""
	@echo "  Server"
	@echo "    make server           uvicorn on :8000 --reload"
	@echo ""
	@echo "  Indexing"
	@echo "    make index-pdf        PDF=/abs/path/to/doc.pdf"
	@echo "    make index-pdf-force  PDF=...  re-index even if exists"
	@echo "    make index-pdf-folder FOLDER=/abs/path/to/pdfs/"
	@echo "    make index-url        URL=https://example.com/article"
	@echo ""
	@echo "  Query"
	@echo "    make query            QUERY='your question here'"
	@echo "    make health           GET /health"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean            remove BM25 index + logs"
	@echo "    make reset            WIPE all docker volumes"
	@echo ""

# Docker
up:
	docker compose up -d
	@echo "Waiting for Qdrant + Redis..."
	@sleep 3
	@docker compose ps

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

# Python env
install:
	uv sync

setup: install
	@echo "Downloading CLIP model weights (~340MB, one-time)..."
	uv run python -c "import numpy as np; from sentence_transformers import SentenceTransformer; m = SentenceTransformer('clip-ViT-B-32'); dim = m.encode('test').shape[0]; print('CLIP ready - dim:', dim)"
	@echo "Downloading spaCy model..."
	uv run python -m spacy download en_core_web_sm
	@echo ""
	@echo ""
	@echo "Setup complete. Run: make up && make server"
	@echo "Note: Docling will download OCR models on first scanned PDF (~500MB)"

# Server
server:
	cd $(SRC) && uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Indexing
index-pdf:
	@test -n "$(PDF)" || (echo "Usage: make index-pdf PDF=/path/to/doc.pdf" && exit 1)
	cd $(SRC) && uv run python ingest_pdf.py --path "$(PDF)"

index-pdf-force:
	@test -n "$(PDF)" || (echo "Usage: make index-pdf-force PDF=/path/to/doc.pdf" && exit 1)
	cd $(SRC) && uv run python ingest_pdf.py --path "$(PDF)" --force

index-pdf-folder:
	@test -n "$(FOLDER)" || (echo "Usage: make index-pdf-folder FOLDER=/path/to/pdfs/" && exit 1)
	cd $(SRC) && uv run python ingest_pdf.py --folder "$(FOLDER)"

index-url:
	@test -n "$(URL)" || (echo "Usage: make index-url URL=https://example.com" && exit 1)
	cd $(SRC) && uv run python ingest_html.py --url "$(URL)"

index-browser:
	cd $(SRC) && uv run python ingest_browser.py --browser "$(BROWSER)" --limit "$(or $(LIMIT),200)"

index-claude:
	@test -n "$(PATH)" || (echo "Usage: make index-claude PATH=/path/to/conversations.json" && exit 1)
	cd $(SRC) && uv run python ingest_claude.py --path "$(PATH)" $(if $(LIMIT),--limit $(LIMIT),)

# Query
query:
	cd $(SRC) && uv run python query.py "$(QUERY)"

# Health
health:
	@curl -sf http://localhost:8000/health | uv run python -m json.tool

# Cleanup
wipe-collections:
	@echo "Deleting all rag_ collections from Qdrant..."
	uv run python -c "from qdrant_client import QdrantClient; q = QdrantClient(host='localhost', port=6333); deleted = [q.delete_collection(c.name) for c in q.get_collections().collections if c.name.startswith('rag_')]; print(f'Deleted {len(deleted)} collections')"
	@echo "Also clearing BM25 index..."
	rm -f data/bm25_index.pkl data/bm25_meta.json
	@echo "Done. Re-index with: make index-pdf PDF=..."

clean:
	rm -rf data/bm25_index.pkl data/bm25_meta.json logs/
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true

reset: down clean
	@echo "Wiping ALL docker volumes..."
	docker compose down -v
	@echo "Done. Run: make up && make server"
