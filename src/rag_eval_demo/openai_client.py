from __future__ import annotations

import os
from typing import Any

from .models import SearchResult
from .retrieval import format_context


ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý RAG cho demo kiểm thử hệ thống hỏi đáp về Bộ luật Lao động Việt Nam.
Chỉ được sử dụng CONTEXT được cung cấp. Nếu context không đủ căn cứ, hãy nói rõ là không tìm thấy căn cứ trong tài liệu.
Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc. Mỗi ý quan trọng phải có citation dạng [Trang X, Điều Y, Cn].
Không tư vấn pháp lý ngoài phạm vi tài liệu."""


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
