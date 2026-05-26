# RAG Evaluation Demo - Luat Lao Dong Viet Nam

Demo CLI RAG cho chu de evaluation mot ung dung hoi dap ve Bo luat Lao dong Viet Nam.

## Quick Start

1. Tao file `.env` tu `.env.example` va dien `OPENAI_API_KEY`.
2. Build image:

```bash
docker compose build
```

3. OCR PDF, chunk va tao index:

```bash
docker compose run --rm rag ingest
```

4. Hoi dap tren command line:

```bash
docker compose run --rm rag ask "Nguoi lao dong lam hop dong khong xac dinh thoi han muon nghi viec phai bao truoc bao lau?" --show-context
```

5. Chay evaluation compact:

```bash
docker compose run --rm rag eval --skip-judge
docker compose run --rm rag eval
```

Bao cao duoc ghi vao `reports/`.

## Offline Smoke Test

Neu chua co API key, co the test luong code bang local hash embeddings va bo qua generation:

```bash
docker compose run --rm rag ingest --local-embeddings
docker compose run --rm rag eval --local-embeddings --skip-generation
```

Ket qua offline chi de smoke test pipeline, khong phai metric chat luong RAG that.

## Documents

- `docs/RAG_SYSTEM_DESIGN.md`: cach he thong RAG duoc xay dung.
- `docs/RAG_EVALUATION_GUIDE.md`: cach tao test data va danh gia chat luong.
