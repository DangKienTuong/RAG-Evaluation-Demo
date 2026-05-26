# RAG Evaluation Guide

## Evaluation Goal

This demo does not only ask whether the final answer is good. It separates quality into testable layers:

1. Whether the document was OCRed correctly.
2. Whether chunks preserve the necessary evidence.
3. Whether retrieval finds the correct article/page.
4. Whether generation produces a correct answer with citations.
5. Whether the system refuses to answer when the question is outside the document scope.

For a Software Testing team, this layered approach makes failures easier to diagnose: the issue may come from OCR, the retriever, or the LLM generation step.

## Test Data

The test suite lives in `eval/test_cases.yaml` and contains 25 compact cases:

- `direct_fact`: a question that should be answered directly from one article.
- `article_specific`: a question that must retrieve the correct article and summarize it accurately.
- `broad_article`: a broader question that may require several points from the same article.
- `ambiguous`: a question with missing situation details where the system should avoid overclaiming.
- `out_of_scope`: a question outside the Vietnamese Labor Code.

Each case has these fields:

- `id`: test case identifier.
- `question`: user question.
- `category`: risk or behavior category.
- `difficulty`: easy, medium, or hard.
- `answerable`: true when the document contains enough evidence to answer.
- `expected_citations`: expected article/page evidence. Because the PDF is scanned, article is the primary evidence target; page can be added after OCR if stricter checks are needed.
- `expected_keywords`: keywords used by lightweight rule checks.
- `expected_answer_points`: main answer points expected from the system.

Because the source PDF is scanned, the default test data uses article-level evidence. Page numbers depend on OCR quality and chunking behavior. After the first ingestion run, you can inspect `storage/index.json`, find the chunk for each article, and add `page` to test cases for stricter evaluation.

## Retrieval Metrics

The `eval` command retrieves top-k chunks for each question.

Main metrics:

- `Recall@5`: for answerable cases, whether expected evidence appears in the top 5 retrieved chunks.
- `MRR`: reciprocal rank of the first correct evidence. Evidence at rank 1 is better than evidence at rank 5.
- `first_evidence_rank`: the first rank that matches the expected article/page.

Evidence matching rules:

- If a case has `article`, a chunk matches when either the chunk metadata or chunk text contains that article.
- If a case has `page`, a chunk matches when the page is between `page_start` and `page_end`.
- If both article and page are provided, both must match.

Demo thresholds:

- `Recall@5 >= 0.80`.
- Higher MRR is better; for this small demo, expect `> 0.60` after OCR is stable.

## Answer Metrics

When `--skip-generation` is not used, the system calls an LLM to generate the answer. When `--skip-judge` is not used, it calls an LLM-as-judge with Structured Outputs and expects JSON.

The judge scores:

- `correctness`: whether the answer matches the expected answer points.
- `groundedness`: whether the answer is supported by retrieved context.
- `completeness`: whether the main expected points are covered.
- `citation_support`: whether citations support the claims they accompany.
- `hallucination_risk`: low, medium, or high.
- `verdict`: pass, borderline, or fail.
- `notes`: short explanation.

Additional rule checks:

- Answerable cases should include a citation such as `[Trang X, ...]`.
- Unanswerable cases should include a refusal phrase such as "không tìm thấy căn cứ" or "không đủ căn cứ".
- Keyword hit ratio shows whether the answer mentions important expected terms.

Demo thresholds:

- Average `groundedness >= 4/5`.
- No out-of-scope case should receive `hallucination_risk=high`.
- Ambiguous cases should not be answered with excessive certainty when key details are missing.

## Evaluation Commands

Run full evaluation:

```bash
docker compose run --rm rag eval
```

Run quickly and skip LLM-as-judge:

```bash
docker compose run --rm rag eval --skip-judge
```

Test retrieval only, without calling the LLM:

```bash
docker compose run --rm rag eval --skip-generation
```

Run the first 5 cases as a smoke test:

```bash
docker compose run --rm rag eval --limit 5 --skip-judge
```

Reports are written to:

- `reports/eval_<timestamp>.json`: detailed machine-readable results.
- `reports/eval_<timestamp>.md`: human-readable report for review and sharing.

## How To Read The Report

In the summary:

- Low recall: inspect OCR and chunking first, then tune retrieval.
- Low MRR but acceptable recall: evidence is retrieved but ranked too low; consider increasing keyword weight or changing chunking.
- Low groundedness: answer generation prompt or context ranking may be the problem.
- High-risk case count > 0: inspect out-of-scope and ambiguous cases immediately, because hallucination risk is the most important demo failure mode.

In each case:

- Retrieval miss but correct answer: the model may be relying on outside knowledge, which is a RAG risk.
- Retrieval hit but wrong answer: the issue is likely generation, prompt behavior, or citation use.
- Retrieval hit but refusal: the context may be incomplete or the prompt may be too conservative.

## Improvement Workflow

1. Run `ingest` and inspect OCR warnings.
2. Run `eval --skip-generation` to measure retrieval separately.
3. Improve OCR, chunking, or retrieval until Recall@5 reaches the threshold.
4. Run `eval --skip-judge` to inspect answers and citations.
5. Run full `eval` to produce judge JSON and the markdown report.
6. Add new test cases for failures that actually appear during the demo.

## Extending The Test Data

When adding new test cases, prioritize:

- Questions that use terminology different from the source text, to test semantic search.
- Questions involving multiple related articles, to test multi-hop behavior.
- Questions with numbers, dates, hours, or limits, to test precision.
- Questions outside labor law, to test refusal behavior.
- Ambiguous questions with missing situation details, to test over-answering.
