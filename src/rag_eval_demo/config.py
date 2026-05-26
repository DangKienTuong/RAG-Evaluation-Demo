from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    storage_dir: Path
    reports_dir: Path
    chat_model: str
    embedding_model: str
    top_k: int
    ocr_lang: str
    ocr_dpi: int
    embedding_provider: str
    max_output_tokens: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            storage_dir=Path(os.getenv("RAG_STORAGE_DIR", "storage")),
            reports_dir=Path(os.getenv("RAG_REPORTS_DIR", "reports")),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-5-mini"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large"),
            top_k=_int_env("RAG_TOP_K", 5),
            ocr_lang=os.getenv("RAG_OCR_LANG", "vie+eng"),
            ocr_dpi=_int_env("RAG_OCR_DPI", 220),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "openai"),
            max_output_tokens=_int_env("RAG_MAX_OUTPUT_TOKENS", 1200),
        )

    @property
    def ocr_dir(self) -> Path:
        return self.storage_dir / "ocr"

    @property
    def pages_path(self) -> Path:
        return self.ocr_dir / "pages.json"

    @property
    def index_path(self) -> Path:
        return self.storage_dir / "index.json"


def find_default_pdf(base_dir: Path | None = None) -> Path:
    base = base_dir or Path.cwd()
    pdfs = sorted(base.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Không tìm thấy file PDF trong {base}. Hãy truyền --pdf <path>."
        )
    if len(pdfs) > 1:
        names = ", ".join(path.name for path in pdfs)
        raise FileExistsError(
            f"Có nhiều file PDF trong {base}: {names}. Hãy truyền --pdf <path>."
        )
    return pdfs[0]
