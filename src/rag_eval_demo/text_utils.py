from __future__ import annotations

import re
import unicodedata


TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+", re.UNICODE)


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def normalize_for_search(value: str) -> str:
    return strip_accents(value).lower()


def tokenize(value: str) -> list[str]:
    normalized = normalize_for_search(value)
    return TOKEN_RE.findall(normalized)


def clean_ocr_text(value: str) -> str:
    text = value.replace("\x0c", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s+", "", text)
    return text.strip()


def short_text(value: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
