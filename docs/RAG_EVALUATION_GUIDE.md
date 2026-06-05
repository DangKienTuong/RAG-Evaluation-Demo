# RAG Evaluation Scaffold

## Goal

Repo nay chi cung cap RAG system va mot scaffold danh gia nhe de hoc vien co du dau vao tu xay bo cham bang LLM.

Scaffold hien tai tach cac lop can quan sat:

1. OCR co trich xuat duoc text tu PDF hay khong.
2. Chunk co giu du bang chung can thiet hay khong.
3. Retrieval co dua dung dieu/trang vao top-k hay khong.
4. Answer generation co sinh cau tra loi kem citation hay khong.
5. Cau hoi ngoai pham vi co duoc tu choi hay khong.

## Test Data

Bo test nam o `eval/test_cases.yaml`.

Moi case co cac field:

- `id`: ma test case.
- `question`: cau hoi nguoi dung.
- `category`: nhom rui ro hoac hanh vi.
- `difficulty`: easy, medium, hard.
- `answerable`: true neu tai lieu co du bang chung de tra loi.
- `expected_citations`: dieu/trang ky vong dung lam bang chung.
- `expected_keywords`: keyword cho rule check nhe.
- `expected_answer_points`: cac y chinh de hoc vien co the dung khi thiet ke evaluator rieng.

Vi PDF nguon la file scan, test data mac dinh uu tien evidence theo dieu luat. Sau khi chay `ingest`, co the mo `storage/index.json` de bo sung `page` neu muon check chat hon.

## Retrieval Metrics

Lenh `eval` retrieve top-k chunks cho tung cau hoi.

Metric chinh:

- `Recall@5`: voi cac case answerable, expected evidence co xuat hien trong top 5 chunk hay khong.
- `MRR`: reciprocal rank cua evidence dau tien dung.
- `first_evidence_rank`: rank dau tien match expected article/page.

Evidence matching:

- Neu case co `article`, chunk match khi metadata hoac text cua chunk chua article do.
- Neu case co `page`, chunk match khi page nam trong `page_start` va `page_end`.
- Neu co ca article va page, ca hai dieu kien deu phai dung.

## Lightweight Answer Checks

Khi khong dung `--skip-generation`, he thong sinh cau tra loi RAG va ghi vao report. Scaffold chi chay cac rule check nhe:

- `keyword_hit_ratio`: ti le keyword ky vong xuat hien trong answer.
- `citation_present`: answer co citation dang `[Trang X, ...]` hay khong.
- `unanswerable_refusal`: voi case ngoai pham vi, answer co tu choi tra loi hay khong.

Nhung check nay khong thay the bo cham bang LLM. Chung chi giup hoc vien co baseline de so sanh khi tu xay evaluator.

## Commands

Chay report retrieval + answer checks:

```bash
docker compose run --rm rag eval
```

Chi test retrieval, khong goi LLM sinh answer:

```bash
docker compose run --rm rag eval --skip-generation
```

Chay nhanh 5 case dau:

```bash
docker compose run --rm rag eval --limit 5
```

Report duoc ghi vao:

- `reports/eval_<timestamp>.json`: ket qua chi tiet dang JSON.
- `reports/eval_<timestamp>.md`: report de doc nhanh.

## Reading The Report

- Low recall: uu tien kiem tra OCR, chunking va query/retrieval.
- Low MRR nhung recall tot: evidence co duoc retrieve nhung rank thap; can tune keyword weight, embedding, chunk size hoac query.
- Citation missing: answer generation prompt co the chua ep model cite du manh.
- Unanswerable non-refusal > 0: out-of-scope case dang bi tra loi qua tay.

## Suggested Student Exercise

Hoc vien co the tu them mot module evaluator rieng de doc `question`, `expected_*`, `answer`, va `retrieval`, sau do tu thiet ke cach cham va cach giai thich ket qua cua nhom. Repo hien tai co chu y khong cung cap implementation do.
