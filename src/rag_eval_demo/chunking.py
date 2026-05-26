from __future__ import annotations

import re

from .models import Chunk, PageText
from .text_utils import clean_ocr_text, normalize_for_search


ARTICLE_RE = re.compile(
    r"(?im)^\s*(?:điều|die[uú]|dieu|ðieu|ðiều)\s+([0-9]+[a-zA-Z]?)\s*[\.:]?\s*(.*)$"
)


def find_articles(text: str) -> list[tuple[str, str | None]]:
    articles: list[tuple[str, str | None]] = []
    for match in ARTICLE_RE.finditer(text):
        number = match.group(1)
        title = match.group(2).strip() or None
        articles.append((f"Điều {number}", title))
    return articles


def _split_long_text(value: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(value) <= max_chars:
        return [value]
    parts: list[str] = []
    start = 0
    while start < len(value):
        end = min(len(value), start + max_chars)
        parts.append(value[start:end].strip())
        if end >= len(value):
            break
        start = max(0, end - overlap_chars)
    return [part for part in parts if part]


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n|(?<=\.)\n", text)
    return [clean_ocr_text(part) for part in parts if clean_ocr_text(part)]


def build_chunks(
    pages: list[PageText],
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_article: str | None = None
    current_title: str | None = None

    for page in pages:
        text = clean_ocr_text(page.text)
        if not text:
            continue

        buffer: list[str] = []
        for para in _paragraphs(text):
            para_articles = find_articles(para)
            if para_articles:
                current_article, current_title = para_articles[-1]

            candidate = "\n\n".join([*buffer, para]).strip()
            if len(candidate) <= max_chars:
                buffer.append(para)
                continue

            if buffer:
                _emit_chunk(
                    chunks,
                    "\n\n".join(buffer),
                    page.page_number,
                    current_article,
                    current_title,
                    max_chars,
                    overlap_chars,
                )
            buffer = [para]

        if buffer:
            _emit_chunk(
                chunks,
                "\n\n".join(buffer),
                page.page_number,
                current_article,
                current_title,
                max_chars,
                overlap_chars,
            )

    return chunks


def _emit_chunk(
    chunks: list[Chunk],
    text: str,
    page_number: int,
    article: str | None,
    title: str | None,
    max_chars: int,
    overlap_chars: int,
) -> None:
    text = clean_ocr_text(text)
    detected = find_articles(text)
    if detected:
        article, title = detected[-1]

    for part in _split_long_text(text, max_chars=max_chars, overlap_chars=overlap_chars):
        chunk_id = f"p{page_number:03d}-c{len(chunks) + 1:04d}"
        chunks.append(
            Chunk(
                id=chunk_id,
                text=part,
                page_start=page_number,
                page_end=page_number,
                article=article,
                title=title,
            )
        )


def article_matches(text_or_article: str | None, expected_article: str) -> bool:
    if not text_or_article:
        return False
    return normalize_for_search(expected_article) in normalize_for_search(text_or_article)
