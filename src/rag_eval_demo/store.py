from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Chunk, PageText


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def save_pages(path: Path, pdf_path: Path, pages: list[PageText]) -> None:
    write_json(
        path,
        {
            "metadata": {
                "pdf": str(pdf_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "page_count": len(pages),
            },
            "pages": [page.to_dict() for page in pages],
        },
    )


def load_pages(path: Path) -> list[PageText]:
    data = read_json(path)
    return [PageText.from_dict(item) for item in data.get("pages", [])]


def save_index(path: Path, metadata: dict[str, Any], chunks: list[Chunk]) -> None:
    write_json(
        path,
        {
            "metadata": {
                **metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(chunks),
            },
            "chunks": [chunk.to_dict(include_embedding=True) for chunk in chunks],
        },
    )


def load_index(path: Path) -> tuple[dict[str, Any], list[Chunk]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy index tại {path}. Hãy chạy lệnh ingest trước."
        )
    data = read_json(path)
    chunks = [Chunk.from_dict(item) for item in data.get("chunks", [])]
    return data.get("metadata", {}), chunks
