# RAG System Design

## Goal

This demo builds a small command-line RAG system for question answering over the Vietnamese Labor Code PDF file `Luật Lao Động Việt Nam Hiện Hành.signed.pdf`. The primary audience is a Software Testing team, so the design favors transparency, debuggability, and clear separation between quality layers: OCR, chunking, retrieval, and answer generation.

This is a RAG testing demo, not a legal advice tool. Answers must be grounded only in the context retrieved from the provided PDF.

## Architecture

The pipeline has 5 stages:

1. **PDF ingestion**
   - If `--pdf` is not provided, the CLI looks for the single PDF file in the workspace.
   - The app first tries to read a text layer with `pypdf`.
   - The current PDF is scanned and has an empty text layer, so the pipeline falls back to OCR.

2. **OCR**
   - The Docker image installs `poppler-utils`, `tesseract-ocr`, and `tesseract-ocr-vie`.
   - `pdftoppm` renders each PDF page as a PNG image.
   - `tesseract -l vie+eng` extracts text from each rendered page.
   - OCR output is cached in `storage/ocr/pages.json`.
   - Pages with very little extracted text are marked with the `low_ocr_text` warning so they can be tested as data quality risks.

3. **Chunking**
   - Page text is normalized by cleaning whitespace.
   - Default chunk size is about 1800 characters with 250 characters of overlap.
   - The system uses a regex to detect `Điều N` article labels and attach article metadata to chunks.
   - Each chunk stores `id`, `text`, `page_start`, `page_end`, `article`, and `title`.

4. **Indexing and retrieval**
   - The default embedding provider is OpenAI with model `text-embedding-3-large`.
   - The local index is stored at `storage/index.json`.
   - Retrieval uses hybrid search:
     - Semantic score: cosine similarity over embeddings.
     - Keyword score: BM25 over normalized Vietnamese tokens with accents stripped.
     - Combined score: 70% semantic, 30% keyword by default.
   - The CLI returns top-k chunks; default is `RAG_TOP_K=5`.

5. **Answer generation**
   - `ask` calls the OpenAI Responses API with default model `gpt-5-mini`.
   - The prompt requires the model to answer only from retrieved context.
   - If the context is insufficient, the model must explicitly say that the provided document does not contain enough evidence.
   - Important claims should include citations in the format `[Trang X, Điều Y, Cn]`.

## How To Run

Create `.env`:

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY`, then build the Docker image:

```bash
docker compose build
```

Run ingestion:

```bash
docker compose run --rm rag ingest
```

Ask a question:

```bash
docker compose run --rm rag ask "What is the maximum normal working time?" --show-context
```

Retrieve context only, without calling the LLM:

```bash
docker compose run --rm rag ask "How many types of labor contracts are defined?" --retrieve-only
```

Re-run OCR from scratch:

```bash
docker compose run --rm rag ingest --force-ocr
```

## File Structure

- `src/rag_eval_demo/ocr.py`: renders the PDF and runs OCR.
- `src/rag_eval_demo/chunking.py`: chunks text and attaches article metadata.
- `src/rag_eval_demo/embeddings.py`: OpenAI embeddings and local hash embeddings for smoke tests.
- `src/rag_eval_demo/retrieval.py`: BM25, cosine similarity, and hybrid ranking.
- `src/rag_eval_demo/openai_client.py`: answer generation.
- `src/rag_eval_demo/evaluation.py`: retrieval metrics, lightweight answer checks, and report generation.
- `eval/test_cases.yaml`: compact evaluation suite.
- `reports/`: generated evaluation reports.

## What To Test Separately

- **OCR quality**: pages with too little text, incorrect Vietnamese accents, or missing articles/clauses.
- **Chunk boundaries**: whether chunks split important legal meaning or attach incorrect article metadata.
- **Retrieval quality**: whether the expected article/page appears in top-k results.
- **Answer grounding**: whether the answer is supported by retrieved context.
- **Citation quality**: whether citations point to the correct chunk/page/article.
- **Negative handling**: whether out-of-scope questions trigger hallucinated answers.

## Common Issues

- `OPENAI_API_KEY chưa được cấu hình`: create `.env` and fill in the API key.
- `Không tìm thấy pdftoppm`: run through Docker or install `poppler-utils`.
- `Không tìm thấy tesseract`: run through Docker or install Tesseract plus the Vietnamese language pack.
- OCR is slow: this is expected for an 83-page scanned PDF; results are cached after the first run.
- OCR quality is poor: increase `RAG_OCR_DPI`, run `ingest --force-ocr`, then compare retrieval metrics again.

## Environment Variables

- `OPENAI_API_KEY`: required for OpenAI embeddings and generation.
- `RAG_CHAT_MODEL`: answer generation model, default `gpt-5-mini`.
- `RAG_EMBEDDING_MODEL`: embedding model, default `text-embedding-3-large`.
- `RAG_TOP_K`: number of chunks to retrieve, default `5`.
- `RAG_OCR_LANG`: OCR language setting, default `vie+eng`.
- `RAG_OCR_DPI`: PDF rendering DPI, default `220`.

## Offline Smoke Test

Offline mode uses deterministic local hash embeddings only to verify the code path when no API key is available:

```bash
docker compose run --rm rag ingest --local-embeddings
docker compose run --rm rag eval --local-embeddings --skip-generation
```

Do not use offline smoke-test metrics as final RAG quality evidence.
