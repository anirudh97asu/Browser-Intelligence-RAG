"""
logger.py — structured logging for the entire project.

Usage in any script:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Indexed page", page=3, chunks=7, elapsed_ms=120)
    log.error("Upsert failed", collection="rag_pdf__foo", error=str(e))

Output (console): timestamped, coloured by level
Output (file):    JSON-lines for machine parsing (one JSON object per line)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_LEVEL, LOG_FILE


# ── Colours (console only) ────────────────────────────────────────────────────

_RESET  = "\033[0m"
_GREY   = "\033[90m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BRED   = "\033[1;31m"

_LEVEL_COLOUR = {
    "DEBUG":    _GREY,
    "INFO":     _GREEN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _BRED,
}


# ── Console formatter ─────────────────────────────────────────────────────────

class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts    = datetime.now().strftime("%H:%M:%S")
        level = record.levelname
        col   = _LEVEL_COLOUR.get(level, "")
        name  = record.name.split(".")[-1]          # last part of module name
        msg   = record.getMessage()

        # Extra structured fields
        extras = {k: v for k, v in record.__dict__.items()
                  if k not in logging.LogRecord.__dict__ and
                     k not in ("message", "asctime", "args", "exc_info",
                               "exc_text", "stack_info", "taskName")}
        extra_str = "  " + "  ".join(f"{k}={v!r}" for k, v in extras.items()) if extras else ""

        return f"{_GREY}{ts}{_RESET}  {col}{level:<8}{_RESET}  {name}  {msg}{_GREY}{extra_str}{_RESET}"


# ── JSON file formatter ───────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "module":  record.name,
            "msg":     record.getMessage(),
        }
        # Include any extra fields passed to log.info("...", key=val)
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ and \
               k not in ("message", "asctime", "args", "exc_info",
                         "exc_text", "stack_info", "taskName"):
                payload[k] = v
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Logger factory ────────────────────────────────────────────────────────────

_initialised = False

def _init_root():
    global _initialised
    if _initialised:
        return
    _initialised = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ConsoleFormatter())
    root.addHandler(ch)

    # File handler (JSON lines)
    if LOG_FILE:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers",
                  "transformers", "huggingface_hub", "bm25s", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class StructuredLogger:
    """Thin wrapper that lets you pass keyword args as structured fields."""

    def __init__(self, name: str):
        _init_root()
        self._log = logging.getLogger(name)

    def _emit(self, level: int, msg: str, **kwargs):
        if self._log.isEnabledFor(level):
            record = self._log.makeRecord(
                self._log.name, level, "(unknown)", 0,
                msg, (), None,
            )
            for k, v in kwargs.items():
                setattr(record, k, v)
            self._log.handle(record)

    def debug(self,    msg: str, **kw): self._emit(logging.DEBUG,    msg, **kw)
    def info(self,     msg: str, **kw): self._emit(logging.INFO,     msg, **kw)
    def warning(self,  msg: str, **kw): self._emit(logging.WARNING,  msg, **kw)
    def error(self,    msg: str, **kw): self._emit(logging.ERROR,    msg, **kw)
    def critical(self, msg: str, **kw): self._emit(logging.CRITICAL, msg, **kw)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
