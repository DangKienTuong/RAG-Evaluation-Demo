from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .chunking import build_chunks
from .config import Settings, find_default_pdf
from .embeddings import make_embedder
from .evaluation import aggregate, load_cases, rule_checks, save_reports
from .models import Chunk
from .ocr import load_or_extract_pages
from .openai_client import RAGOpenAIClient, api_error_result
from .retrieval import format_context, search
from .store import load_index, save_index
from .text_utils import short_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval-demo",
        description="CLI RAG demo and evaluation suite for Vietnamese labor law.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="OCR PDF, chunk text, create embeddings.")
    ingest.add_argument("--pdf", type=Path, default=None, help="Path to labor-law PDF.")
    ingest.add_argument("--force-ocr", action="store_true", help="Rebuild OCR cache.")
    ingest.add_argument("--chunk-size", type=int, default=1800)
    ingest.add_argument("--overlap", type=int, default=250)
    ingest.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Use deterministic local embeddings for offline smoke tests.",
    )
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="Ask a question against the local RAG index.")
    ask.add_argument("question", help="Vietnamese question to ask.")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--show-context", action="store_true")
    ask.add_argument("--retrieve-only", action="store_true", help="Do not call the LLM.")
    ask.add_argument("--local-embeddings", action="store_true")
    ask.set_defaults(func=cmd_ask)

    evaluate = subparsers.add_parser("eval", help="Run compact RAG evaluation suite.")
    evaluate.add_argument("--cases", type=Path, default=Path("eval/test_cases.yaml"))
    evaluate.add_argument("--top-k", type=int, default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--skip-generation", action="store_true")
    evaluate.add_argument("--skip-judge", action="store_true")
    evaluate.add_argument("--local-embeddings", action="store_true")
    evaluate.add_argument("--report-prefix", default="eval")
    evaluate.set_defaults(func=cmd_eval)

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = (args.pdf or find_default_pdf()).resolve()
    provider = "local" if args.local_embeddings else settings.embedding_provider

    print(f"PDF: {pdf_path}")
    print(f"OCR cache: {settings.pages_path}")
    pages = load_or_extract_pages(
        pdf_path=pdf_path,
        cache_path=settings.pages_path,
        ocr_dir=settings.ocr_dir,
        dpi=settings.ocr_dpi,
        lang=settings.ocr_lang,
        force_ocr=args.force_ocr,
    )
    _print_ocr_summary(pages=[page.to_dict() for page in pages])

    chunks = build_chunks(pages, max_chars=args.chunk_size, overlap_chars=args.overlap)
    if not chunks:
        raise RuntimeError("Không tạo được chunk nào từ OCR text.")

    embedder = make_embedder(provider, settings.embedding_model)
    print(f"Embedding provider/model: {provider}/{embedder.model_name}")
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    metadata = {
        "pdf": str(pdf_path),
        "ocr_lang": settings.ocr_lang,
        "ocr_dpi": settings.ocr_dpi,
        "embedding_provider": provider,
        "embedding_model": embedder.model_name,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }
    save_index(settings.index_path, metadata=metadata, chunks=chunks)
    print(f"Saved index: {settings.index_path} ({len(chunks)} chunks)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    _, chunks = load_index(settings.index_path)
    provider = "local" if args.local_embeddings else settings.embedding_provider
    embedder = make_embedder(provider, settings.embedding_model)
    top_k = args.top_k or settings.top_k
    results = search(args.question, chunks, embedder, top_k=top_k)

    if args.show_context or args.retrieve_only:
        print("=== Retrieved context ===")
        print(format_context(results, max_chars_per_chunk=900))
        print()

    if args.retrieve_only:
        return 0

    client = RAGOpenAIClient(
        chat_model=settings.chat_model,
        max_output_tokens=settings.max_output_tokens,
    )
    answer = client.answer(args.question, results)
    print(answer)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    metadata, chunks = load_index(settings.index_path)
    provider = "local" if args.local_embeddings else metadata.get(
        "embedding_provider", settings.embedding_provider
    )
    embedder = make_embedder(provider, settings.embedding_model)
    top_k = args.top_k or settings.top_k
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError(f"Không có test case nào trong {args.cases}.")

    client = None
    if not args.skip_generation:
        client = RAGOpenAIClient(
            chat_model=settings.chat_model,
            max_output_tokens=settings.max_output_tokens,
        )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.id}: {case.question}")
        results = search(case.question, chunks, embedder, top_k=top_k)
        answer = ""
        judge = None
        if client:
            try:
                answer = client.answer(case.question, results)
            except Exception as exc:
                judge = api_error_result("answer_generation", exc)
            else:
                if not args.skip_judge:
                    try:
                        judge = client.judge(
                            case.question, case.expected_payload(), answer, results
                        )
                    except Exception as exc:
                        judge = api_error_result("llm_judge", exc)
        metrics = rule_checks(case, answer, results)
        rows.append(
            {
                "case": {
                    "id": case.id,
                    "question": case.question,
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "answerable": case.answerable,
                    "expected_citations": case.expected_citations,
                    "expected_keywords": case.expected_keywords,
                    "expected_answer_points": case.expected_answer_points,
                },
                "metrics": metrics,
                "judge": judge,
                "answer": answer,
                "retrieval": [result.to_dict() for result in results],
            }
        )
        _print_case_line(rows[-1])

    summary = aggregate(rows, top_k=top_k)
    json_path, md_path = save_reports(
        settings.reports_dir,
        rows=rows,
        summary=summary,
        top_k=top_k,
        prefix=args.report_prefix,
    )
    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nReports:\n- {json_path}\n- {md_path}")
    return 0


def _print_ocr_summary(pages: list[dict[str, Any]]) -> None:
    total_chars = sum(int(page.get("char_count", 0)) for page in pages)
    low_pages = [
        page["page_number"]
        for page in pages
        if "low_ocr_text" in page.get("warnings", [])
        or int(page.get("char_count", 0)) < 80
    ]
    print(f"Pages: {len(pages)} | OCR chars: {total_chars}")
    if low_pages:
        preview = ", ".join(str(page) for page in low_pages[:12])
        suffix = "..." if len(low_pages) > 12 else ""
        print(f"Warning: low OCR text on pages {preview}{suffix}")


def _print_case_line(row: dict[str, Any]) -> None:
    metrics = row["metrics"]
    retrieval = (
        f"rank={metrics['first_evidence_rank']}"
        if metrics["first_evidence_rank"]
        else "miss"
    )
    judge = row.get("judge") or {}
    verdict = judge.get("verdict", "no-judge")
    note = short_text(judge.get("notes", row.get("answer", "")), limit=100)
    print(f"  retrieval={retrieval} verdict={verdict} {note}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
