from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

from .models import PageText
from .store import load_pages, save_pages
from .text_utils import clean_ocr_text


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def extract_pdf_text_layer(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = clean_ocr_text(page.extract_text() or "")
        pages.append(
            PageText(
                page_number=index,
                text=text,
                char_count=len(text),
                source="pdf_text_layer",
                warnings=[] if text else ["empty_text_layer"],
            )
        )
    return pages


def has_usable_text_layer(pages: list[PageText], min_average_chars: int = 300) -> bool:
    if not pages:
        return False
    average_chars = sum(page.char_count for page in pages) / len(pages)
    filled_pages = sum(1 for page in pages if page.char_count >= 100)
    return average_chars >= min_average_chars and filled_pages >= max(3, len(pages) // 2)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def _render_pdf_pages(pdf_path: Path, images_dir: Path, dpi: int, force: bool) -> list[Path]:
    if force and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("page-*.png"), key=_page_image_sort_key)
    if existing and not force:
        return existing

    if not command_exists("pdftoppm"):
        raise RuntimeError(
            "Không tìm thấy pdftoppm. Dockerfile đã cài poppler-utils; "
            "nếu chạy ngoài Docker, hãy cài poppler-utils trước."
        )

    prefix = images_dir / "page"
    _run(["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)])
    images = sorted(images_dir.glob("page-*.png"), key=_page_image_sort_key)
    if not images:
        raise RuntimeError("pdftoppm không tạo được ảnh trang nào từ PDF.")
    return images


def _page_image_sort_key(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def _ocr_image(image_path: Path, lang: str) -> str:
    if not command_exists("tesseract"):
        raise RuntimeError(
            "Không tìm thấy tesseract. Dockerfile đã cài tesseract-ocr-vie; "
            "nếu chạy ngoài Docker, hãy cài Tesseract và gói ngôn ngữ tiếng Việt."
        )
    completed = _run(["tesseract", str(image_path), "stdout", "-l", lang, "--psm", "6"])
    return clean_ocr_text(completed.stdout)


def load_or_extract_pages(
    pdf_path: Path,
    cache_path: Path,
    ocr_dir: Path,
    dpi: int,
    lang: str,
    force_ocr: bool = False,
    prefer_text_layer: bool = True,
) -> list[PageText]:
    if cache_path.exists() and not force_ocr:
        return load_pages(cache_path)

    if prefer_text_layer and not force_ocr:
        text_layer_pages = extract_pdf_text_layer(pdf_path)
        if has_usable_text_layer(text_layer_pages):
            save_pages(cache_path, pdf_path, text_layer_pages)
            return text_layer_pages

    images = _render_pdf_pages(pdf_path, ocr_dir / "images", dpi=dpi, force=force_ocr)
    pages: list[PageText] = []
    for index, image_path in enumerate(images, start=1):
        text = _ocr_image(image_path, lang=lang)
        warnings: list[str] = []
        if len(text) < 80:
            warnings.append("low_ocr_text")
        pages.append(
            PageText(
                page_number=index,
                text=text,
                char_count=len(text),
                source="tesseract_ocr",
                warnings=warnings,
            )
        )
    save_pages(cache_path, pdf_path, pages)
    return pages
