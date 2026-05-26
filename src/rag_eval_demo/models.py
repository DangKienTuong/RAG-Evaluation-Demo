from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str
    char_count: int
    source: str
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageText":
        return cls(
            page_number=int(data["page_number"]),
            text=str(data.get("text", "")),
            char_count=int(data.get("char_count", 0)),
            source=str(data.get("source", "unknown")),
            warnings=list(data.get("warnings", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int
    article: str | None = None
    title: str | None = None
    embedding: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            page_start=int(data["page_start"]),
            page_end=int(data.get("page_end", data["page_start"])),
            article=data.get("article"),
            title=data.get("title"),
            embedding=data.get("embedding"),
        )

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_embedding:
            data.pop("embedding", None)
        return data

    @property
    def citation(self) -> str:
        article = f", {self.article}" if self.article else ""
        if self.page_start == self.page_end:
            return f"Trang {self.page_start}{article}"
        return f"Trang {self.page_start}-{self.page_end}{article}"


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    semantic_score: float
    keyword_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "keyword_score": round(self.keyword_score, 6),
            "chunk": self.chunk.to_dict(include_embedding=False),
        }
