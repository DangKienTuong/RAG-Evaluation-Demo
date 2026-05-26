import unittest

from rag_eval_demo.embeddings import LocalHashEmbedder
from rag_eval_demo.evaluation import EvalCase, first_evidence_rank, rule_checks
from rag_eval_demo.models import Chunk
from rag_eval_demo.retrieval import search


class RetrievalTests(unittest.TestCase):
    def test_hybrid_search_finds_expected_article(self):
        chunks = [
            Chunk(
                id="c1",
                text="Điều 20. Loại hợp đồng lao động gồm hợp đồng không xác định thời hạn và hợp đồng xác định thời hạn.",
                page_start=10,
                page_end=10,
                article="Điều 20",
            ),
            Chunk(
                id="c2",
                text="Điều 105. Thời giờ làm việc bình thường không quá 08 giờ trong một ngày.",
                page_start=50,
                page_end=50,
                article="Điều 105",
            ),
        ]
        embedder = LocalHashEmbedder()
        for chunk in chunks:
            chunk.embedding = embedder.embed_query(chunk.text)

        results = search("Có những loại hợp đồng lao động nào?", chunks, embedder, top_k=2)

        self.assertEqual(results[0].chunk.article, "Điều 20")

    def test_eval_evidence_rank_can_match_article_only(self):
        result = search_result_for_article("Điều 35")
        rank = first_evidence_rank([result], [{"article": "Điều 35"}])

        self.assertEqual(rank, 1)

    def test_rule_checks_detect_unanswerable_refusal(self):
        result = search_result_for_article("Điều 35")
        case = EvalCase(
            id="TCX",
            question="Hỏi ngoài phạm vi",
            category="out_of_scope",
            difficulty="easy",
            answerable=False,
            expected_citations=[],
            expected_keywords=["không tìm thấy căn cứ"],
            expected_answer_points=[],
        )

        metrics = rule_checks(
            case,
            "Tôi không tìm thấy căn cứ đủ rõ trong tài liệu được cung cấp.",
            [result],
        )

        self.assertTrue(metrics["unanswerable_refusal"])


def search_result_for_article(article):
    from rag_eval_demo.models import SearchResult

    return SearchResult(
        chunk=Chunk(
            id="c1",
            text=f"{article}. Người lao động có quyền đơn phương chấm dứt hợp đồng.",
            page_start=12,
            page_end=12,
            article=article,
        ),
        score=1.0,
        semantic_score=1.0,
        keyword_score=1.0,
        rank=1,
    )


if __name__ == "__main__":
    unittest.main()
