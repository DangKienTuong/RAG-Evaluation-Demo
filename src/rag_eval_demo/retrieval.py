from __future__ import annotations

import math
from collections import Counter

from .embeddings import Embedder, cosine_similarity
from .models import Chunk, SearchResult
from .text_utils import tokenize


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk.text) for chunk in chunks]
        self.doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            self.doc_freq.update(set(doc))
        self.doc_lens = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.total_docs = len(self.docs)

    def score(self, query: str, doc_index: int) -> float:
        query_terms = tokenize(query)
        if not query_terms or not self.docs or not self.avgdl:
            return 0.0
        score = 0.0
        doc_len = self.doc_lens[doc_index]
        freqs = self.term_freqs[doc_index]
        for term in query_terms:
            if term not in freqs:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def search(
    question: str,
    chunks: list[Chunk],
    embedder: Embedder,
    top_k: int = 5,
    semantic_weight: float = 0.7,
) -> list[SearchResult]:
    if not chunks:
        return []

    query_embedding = embedder.embed_query(question)
    semantic_raw = [
        cosine_similarity(query_embedding, chunk.embedding or []) for chunk in chunks
    ]
    bm25 = BM25(chunks)
    keyword_raw = [bm25.score(question, index) for index in range(len(chunks))]

    semantic_scores = _minmax(semantic_raw)
    keyword_scores = _minmax(keyword_raw)
    keyword_weight = 1.0 - semantic_weight

    scored: list[SearchResult] = []
    for index, chunk in enumerate(chunks):
        score = semantic_scores[index] * semantic_weight + keyword_scores[index] * keyword_weight
        scored.append(
            SearchResult(
                chunk=chunk,
                score=score,
                semantic_score=semantic_scores[index],
                keyword_score=keyword_scores[index],
                rank=0,
            )
        )

    scored.sort(key=lambda result: result.score, reverse=True)
    top = scored[:top_k]
    for rank, result in enumerate(top, start=1):
        result.rank = rank
    return top


def format_context(results: list[SearchResult], max_chars_per_chunk: int = 1800) -> str:
    blocks: list[str] = []
    for result in results:
        text = result.chunk.text[:max_chars_per_chunk].strip()
        blocks.append(
            "\n".join(
                [
                    f"[C{result.rank}] {result.chunk.citation}",
                    f"score={result.score:.3f} semantic={result.semantic_score:.3f} keyword={result.keyword_score:.3f}",
                    text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)
