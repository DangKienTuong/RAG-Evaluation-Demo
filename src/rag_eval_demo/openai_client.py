from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any

from .models import SearchResult
from .retrieval import format_context


ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý RAG cho demo kiểm thử hệ thống hỏi đáp về Bộ luật Lao động Việt Nam.
Chỉ được sử dụng CONTEXT được cung cấp. Nếu context không đủ căn cứ, hãy nói rõ là không tìm thấy căn cứ trong tài liệu.
Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc. Mỗi ý quan trọng phải có citation dạng [Trang X, Điều Y, Cn].
Không tư vấn pháp lý ngoài phạm vi tài liệu."""


JUDGE_SYSTEM_PROMPT = """Bạn là evaluator cho demo kiểm thử RAG. Chấm điểm nghiêm khắc dựa trên câu hỏi, answer, expected evidence và retrieved context.
Không dùng kiến thức ngoài context. Trả về JSON đúng schema."""


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "correctness": {"type": "integer"},
        "groundedness": {"type": "integer"},
        "completeness": {"type": "integer"},
        "citation_support": {"type": "integer"},
        "hallucination_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "verdict": {"type": "string", "enum": ["pass", "borderline", "fail"]},
        "notes": {"type": "string"},
    },
    "required": [
        "correctness",
        "groundedness",
        "completeness",
        "citation_support",
        "hallucination_risk",
        "verdict",
        "notes",
    ],
}


def _judge_result_model():
    from pydantic import BaseModel

    class JudgeResult(BaseModel):
        correctness: int
        groundedness: int
        completeness: int
        citation_support: int
        hallucination_risk: str
        verdict: str
        notes: str

    return JudgeResult


class RAGOpenAIClient:
    def __init__(self, chat_model: str, max_output_tokens: int = 1200) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY chưa được cấu hình. Tạo .env từ .env.example "
                "trước khi chạy ask/eval có sinh câu trả lời."
            )
        from openai import OpenAI

        self.client = OpenAI()
        self.chat_model = chat_model
        self.max_output_tokens = max_output_tokens

    def answer(self, question: str, results: list[SearchResult]) -> str:
        context = format_context(results)
        user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

Yêu cầu:
- Nếu trả lời được, nêu câu trả lời và citation sau từng ý.
- Nếu không đủ căn cứ, nói: "Tôi không tìm thấy căn cứ đủ rõ trong tài liệu được cung cấp."
- Cuối câu trả lời thêm một dòng: "Lưu ý: Đây là demo kiểm thử RAG, không phải tư vấn pháp lý."
"""
        response = self.client.responses.create(
            model=self.chat_model,
            input=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=self.max_output_tokens,
        )
        return _output_text(response).strip()

    def judge(
        self,
        question: str,
        expected: dict[str, Any],
        answer: str,
        results: list[SearchResult],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "expected": expected,
            "answer": answer,
            "retrieved_context": [result.to_dict() for result in results],
        }
        input_messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Chấm JSON cho sample sau. Điểm 5 là tốt nhất, 1 là tệ nhất.\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ]
        max_judge_tokens = max(2000, self.max_output_tokens)

        try:
            response = self.client.responses.parse(
                model=self.chat_model,
                input=input_messages,
                text_format=_judge_result_model(),
                max_output_tokens=max_judge_tokens,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed is not None:
                return parsed.model_dump()

            text = _output_text(response).strip()
            return json.loads(text)
        except (JSONDecodeError, ValueError, TypeError) as exc:
            raw_text = ""
            response_obj = locals().get("response")
            if response_obj is not None:
                raw_text = _output_text(response_obj).strip()
            return _judge_error_result(exc, raw_text)


def _judge_error_result(exc: Exception, raw_text: str = "") -> dict[str, Any]:
    note = f"Judge output parse error: {exc}"
    if raw_text:
        excerpt = raw_text[:240].replace("\n", " ")
        note = f"{note}. Raw output excerpt: {excerpt}"
    return {
        "correctness": None,
        "groundedness": None,
        "completeness": None,
        "citation_support": None,
        "hallucination_risk": "medium",
        "verdict": "fail",
        "notes": note,
    }


def api_error_result(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "correctness": None,
        "groundedness": None,
        "completeness": None,
        "citation_support": None,
        "hallucination_risk": "medium",
        "verdict": "fail",
        "notes": f"{stage} error: {type(exc).__name__}: {exc}",
    }


def _output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)
