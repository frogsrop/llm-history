# Project Instructions

## Language
All code (variable names, function names, comments, string literals in logic) must be in English only.
Russian is allowed only in user-facing content: corpus text, HTML templates, UI labels.

## Training
Always train models on GPU. Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move all tensors/models to that device.
For numpy-based models (TinyRNN, TinyLSTM) — GPU is not applicable, they stay on CPU.

## Project Structure

```
research/
  ├── PLAN.md                  — implementation plan, steps with checkboxes
  ├── main.py                  — FastAPI app, routes, startup hooks
  ├── corpus.py                — unified training corpus (single source of truth)
  ├── pretrain_models.py       — one-time RNN/LSTM training, saves models/rnn_models.pkl
  ├── download_models.py       — one-time download: ruGPT-3 large + fastText cc.ru.300.bin
  ├── environment.yml          — conda environment (use conda, not pip)
  ├── routers/
  │   ├── ngram.py             — N-gram / Markov chain API
  │   ├── rnn.py               — TinyRNN + TinyLSTM (numpy), pretraining cache
  │   ├── embeddings.py        — fastText word vectors, analogy API (not yet implemented)
  │   └── llm_era.py           — ruGPT-3 large attention + generation (not yet implemented)
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
  │   ├── rnn_models.pkl       — pretrained RNN/LSTM cache (regenerate with pretrain_models.py)
  │   ├── rugpt3large/         — ruGPT-3 large weights (HuggingFace)
  │   └── cc.ru.300.bin        — fastText Russian vectors
  ├── data/
  │   └── corpus.txt           — raw corpus sentences (one per line)
  └── tests/
      ├── conftest.py          — fixtures: FastAPI subprocess + Playwright browser
      ├── test_smoke.py        — pages return 200, browser opens
      ├── test_api.py          — API endpoint tests (httpx)
      ├── test_corpus.py       — corpus quality checks
      └── test_ui.py           — UI tests: clicks, sliders, animations (Playwright)
```

## Running

```bash
conda activate llm-explainer
uvicorn main:app --reload        # start server → http://localhost:8000
python pretrain_models.py        # retrain RNN/LSTM after corpus changes
python download_models.py        # one-time model download
pytest tests/ -v                 # run all tests
```
