from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .chunking import article_matches
from .models import SearchResult
from .store import write_json
from .text_utils import normalize_for_search, short_text


@dataclass(slots=True)
class EvalCase:
    id: str
    question: str
    category: str
    difficulty: str
    answerable: bool
    expected_citations: list[dict[str, Any]]
    expected_keywords: list[str]
    expected_answer_points: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            category=str(data.get("category", "unknown")),
            difficulty=str(data.get("difficulty", "unknown")),
            answerable=bool(data.get("answerable", True)),
            expected_citations=list(data.get("expected_citations", [])),
            expected_keywords=list(data.get("expected_keywords", [])),
            expected_answer_points=list(data.get("expected_answer_points", [])),
        )

    def expected_payload(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "expected_citations": self.expected_citations,
            "expected_keywords": self.expected_keywords,
            "expected_answer_points": self.expected_answer_points,
        }


def load_cases(path: Path) -> list[EvalCase]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML chưa được cài. Hãy chạy qua Docker hoặc cài requirements.txt."
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [EvalCase.from_dict(item) for item in data.get("cases", [])]


def first_evidence_rank(results: list[SearchResult], expected: list[dict[str, Any]]) -> int | None:
    for result in results:
        if any(_evidence_matches(result, evidence) for evidence in expected):
            return result.rank
    return None


def _evidence_matches(result: SearchResult, evidence: dict[str, Any]) -> bool:
    checks: list[bool] = []
    page = evidence.get("page")
    if page is not None:
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            page_int = -1
        checks.append(result.chunk.page_start <= page_int <= result.chunk.page_end)

    article = evidence.get("article")
    if article:
        checks.append(
            article_matches(result.chunk.article, str(article))
            or article_matches(result.chunk.text, str(article))
        )

    return all(checks) if checks else False


def rule_checks(case: EvalCase, answer: str, results: list[SearchResult]) -> dict[str, Any]:
    normalized_answer = normalize_for_search(answer)
    keywords = [normalize_for_search(keyword) for keyword in case.expected_keywords]
    keyword_hits = [keyword for keyword in keywords if keyword in normalized_answer]
    citation_present = bool(re.search(r"\[(?:trang|page)\s+\d+", answer, re.IGNORECASE))
    refusal_phrases = [
        "khong tim thay can cu",
        "khong du can cu",
        "khong co thong tin",
        "ngoai pham vi",
    ]
    refusal = any(phrase in normalized_answer for phrase in refusal_phrases)
    first_rank = first_evidence_rank(results, case.expected_citations)
    return {
        "keyword_hit_count": len(keyword_hits),
        "keyword_total": len(keywords),
        "keyword_hit_ratio": len(keyword_hits) / len(keywords) if keywords else None,
        "citation_present": citation_present,
        "unanswerable_refusal": refusal if not case.answerable else None,
        "first_evidence_rank": first_rank,
        "retrieval_hit": first_rank is not None,
    }


def aggregate(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    answerable = [row for row in rows if row["case"]["answerable"]]
    retrieval_hits = [
        row for row in answerable if row["metrics"]["first_evidence_rank"] is not None
    ]
    reciprocal_ranks = [
        1.0 / row["metrics"]["first_evidence_rank"]
        for row in retrieval_hits
        if row["metrics"]["first_evidence_rank"]
    ]
    judged = [row["judge"] for row in rows if row.get("judge")]
    groundedness = [
        judge["groundedness"] for judge in judged if isinstance(judge.get("groundedness"), int)
    ]
    high_risk = [
        row
        for row in rows
        if (row.get("judge") or {}).get("hallucination_risk") == "high"
        or (
            bool(row.get("answer"))
            and
            not row["case"]["answerable"]
            and row["metrics"].get("unanswerable_refusal") is False
        )
    ]
    return {
        f"recall_at_{top_k}": len(retrieval_hits) / len(answerable) if answerable else None,
        "mrr": sum(reciprocal_ranks) / len(answerable) if answerable else None,
        "average_groundedness": sum(groundedness) / len(groundedness)
        if groundedness
        else None,
        "high_risk_count": len(high_risk),
        "case_count": len(rows),
        "answerable_case_count": len(answerable),
    }


def save_reports(
    reports_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    top_k: int,
    prefix: str = "eval",
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"{prefix}_{timestamp}.json"
    md_path = reports_dir / f"{prefix}_{timestamp}.md"
    write_json(json_path, {"summary": summary, "cases": rows})
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown_report(rows, summary, top_k), encoding="utf-8")
    return json_path, md_path


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_report(rows: list[dict[str, Any]], summary: dict[str, Any], top_k: int) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Answerable cases: {summary['answerable_case_count']}",
        f"- Recall@{top_k}: {_fmt_metric(summary.get(f'recall_at_{top_k}'))}",
        f"- MRR: {_fmt_metric(summary.get('mrr'))}",
        f"- Average groundedness: {_fmt_metric(summary.get('average_groundedness'))}",
        f"- High-risk cases: {summary.get('high_risk_count')}",
        "",
        "## Case Results",
        "",
        "| ID | Category | Retrieval | Judge | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = row["metrics"]
        judge = row.get("judge") or {}
        retrieval = (
            f"rank {metrics['first_evidence_rank']}"
            if metrics["first_evidence_rank"]
            else "miss"
        )
        judge_text = judge.get("verdict", "n/a")
        if judge:
            judge_text += f" / grounded {judge.get('groundedness', 'n/a')}"
        notes = short_text(judge.get("notes", row.get("answer", "")), limit=120).replace("|", "\\|")
        lines.append(
            f"| {row['case']['id']} | {row['case']['category']} | {retrieval} | {judge_text} | {notes} |"
        )
    lines.append("")
    lines.append("## Detailed Samples")
    lines.append("")
    for row in rows:
        lines.extend(
            [
                f"### {row['case']['id']} - {row['case']['question']}",
                "",
                f"- Answerable: {row['case']['answerable']}",
                f"- First evidence rank: {_fmt_metric(row['metrics']['first_evidence_rank'])}",
                f"- Keyword hit ratio: {_fmt_metric(row['metrics']['keyword_hit_ratio'])}",
                "",
                "**Top contexts:**",
                "",
            ]
        )
        for result in row["retrieval"]:
            chunk = result["chunk"]
            lines.append(
                f"- C{result['rank']} score={result['score']} citation={chunk.get('page_start')} / {chunk.get('article')}"
            )
        if row.get("answer"):
            lines.extend(["", "**Answer:**", "", row["answer"].strip(), ""])
    return "\n".join(lines).strip() + "\n"
