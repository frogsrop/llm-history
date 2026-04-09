# Project Instructions

## Project Description

Interactive web app explaining the evolution of language models (N-gram → RNN/LSTM → Word2Vec → LLM era).
Built with FastAPI + Jinja2 + vanilla HTML/JS. Runs locally, designed for presenter use on a projector.

**Target audience:** IT company employees without ML background (HR, accountants, artists, developers).
**Role of the app:** visual aid for the presenter, not a standalone textbook.

A **single 10-sentence Russian corpus** (кот и дождь theme) runs through all modules — same text, different approaches, quality difference is obvious.

Historical timeline covered:
- 1990s–2000s: N-gram / Markov chains
- 1986–1997: RNN → LSTM
- 2013: Word2Vec (embeddings)
- 2017+: LLM era (Seq2Seq → Attention → Transformer → BERT/GPT)

## Language
All code (variable names, function names, comments, string literals in logic) must be in English only.
Russian is allowed only in user-facing content: corpus text, HTML templates, UI labels.

## Training
Use the fastest available device for each model type:
- **TinyRNN / TinyLSTM** (PyTorch, tiny vocab) — use `device = torch.device("cpu")` explicitly. GPU overhead exceeds compute for these models (CPU: ~30s, GPU: ~130s on RTX 5080).
- **Qwen2.5-7B-AWQ / large Transformer models** — use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move all tensors/models to that device.

## PyTorch / CUDA
RTX 5080 (Blackwell, sm_120) requires PyTorch nightly with CUDA 12.8 — stable releases do not support this GPU.
Install: `pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128`

## Tokenizer & Cyrillic
Qwen2.5 uses a SentencePiece tokenizer with native Unicode support — Cyrillic works correctly.
For consistent token display, use `tokenizer.decode([token_id]).strip()` per token.

## Project Structure

```
├── CLAUDE.md                — project instructions (this file)
├── .gitignore               — git ignore rules
├── PLAN.md                  — implementation plan, steps with checkboxes
├── main.py                  — FastAPI app, routes, startup hooks
├── corpus.py                — unified training corpus (single source of truth)
├── pretrain_models.py       — one-time RNN/LSTM training, saves models/rnn_models.pt
├── download_models.py       — one-time download: Qwen2.5-7B-AWQ + fastText cc.ru.300.bin
├── requirements.txt         — pip dependencies
├── pytest.ini               — pytest config
├── routers/
│   ├── ngram.py             — N-gram / Markov chain API
│   ├── rnn.py               — TinyRNN + TinyLSTM (PyTorch), pretraining cache, /vocab endpoint
│   ├── embeddings.py        — fastText word vectors, analogy API
│   └── llm_era.py           — Qwen2.5-7B-AWQ attention + generation
├── static/
│   ├── style.css            — dark theme, epoch color coding
│   ├── nav.js               — sidebar, prev/next, localStorage progress
│   ├── utils.js             — animateFlow(), fetch wrappers
│   └── tooltip.js           — universal ? hint button component
├── templates/
│   ├── index.html           — main page + interactive timeline
│   ├── module-1-ngram.html  — N-gram / Markov chains
│   ├── module-2-rnn-lstm.html — RNN + LSTM
│   ├── module-3-embeddings.html — Word2Vec / embeddings
│   ├── module-4-llm-era.html — LLM era: improvement constructor
│   └── module-5-compare.html — final comparison
├── models/
│   ├── rnn_models.pt        — pretrained RNN/LSTM weights (regenerate with pretrain_models.py)
│   ├── qwen2.5-7b-awq/      — Qwen2.5-7B-AWQ weights (Qwen/Qwen2.5-7B-AWQ)
│   └── cc.ru.300.bin        — fastText Russian vectors (cc.ru.300.bin, ~2.6GB)
├── data/
│   └── corpus.txt           — raw corpus sentences (one per line)
└── tests/
    ├── conftest.py          — fixtures: FastAPI subprocess + Playwright browser
    ├── test_smoke.py        — pages return 200, browser opens
    ├── test_api.py          — API endpoint tests (httpx)
    ├── test_corpus.py       — corpus quality checks
    └── test_ui.py           — UI tests: clicks, sliders, animations (Playwright)
```

## Progress

Completed steps (see PLAN.md for full checklist):
- **Step 1** — App skeleton: environment, FastAPI, corpus stub, CSS/JS, tests, model download script
- **Step 2** — Corpus selection: 3 Russian sentences in `corpus.py`
- **Step 3** — `index.html`: interactive timeline, module cards, progress bar
- **Step 4** — `module-1-ngram.html`: N-gram table, generator, seed control
- **Step 5** — `module-2-rnn-lstm.html`: TinyRNN + TinyLSTM (PyTorch), hidden size slider, SVG animation (corpus-adaptive via /api/rnn/vocab)
- **Step 6** — `module-3-embeddings.html`: fastText vectors, 2D PCA word map, analogy arithmetic, before/after comparison
- **Step 7** — `module-4-llm-era.html`: Seq2Seq animation, attention heatmap (Qwen2.5-7B-AWQ, per-head view), cumulative toggles, temperature-controlled generation

Pending:
- **Step 8** — `module-5-compare.html` (side-by-side comparison)
- **Steps 9–10** — corpus tuning, tooltip texts

## Running

```bash
uv sync                          # install all dependencies (creates .venv automatically)
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128  # PyTorch for RTX 5080

uv run uvicorn main:app --reload # start server → http://localhost:8000
uv run python pretrain_models.py # retrain RNN/LSTM after corpus changes
uv run python download_models.py # one-time model download
uv run pytest tests/ -v          # run all tests
```
