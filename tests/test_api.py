"""
test_api.py — API endpoint tests (httpx).
Uncommented as steps 4–7 are implemented.
"""

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --- N-gram API (Step 4) ---
def test_ngram_table_bigram(client):
    r = client.get("/api/ngram/table?n=2")
    assert r.status_code == 200
    data = r.json()
    assert "table" in data
    assert isinstance(data["table"], dict)
    assert len(data["table"]) > 0

def test_ngram_table_unigram(client):
    r = client.get("/api/ngram/table?n=1")
    assert r.status_code == 200
    data = r.json()
    assert "table" in data

def test_ngram_table_trigram(client):
    r = client.get("/api/ngram/table?n=3")
    assert r.status_code == 200
    data = r.json()
    assert "table" in data

def test_ngram_generate_returns_words(client):
    r = client.get("/api/ngram/generate?order=2&start=кот&words=5")
    assert r.status_code == 200
    data = r.json()
    assert "words" in data
    assert isinstance(data["words"], list)
    assert len(data["words"]) >= 1

def test_ngram_generate_unknown_start(client):
    r = client.get("/api/ngram/generate?order=2&start=zzznonsense&words=3")
    assert r.status_code == 200
    data = r.json()
    # should return empty list or fallback
    assert "words" in data

def test_ngram_table_probabilities_sum(client):
    r = client.get("/api/ngram/table?n=1")
    data = r.json()
    table = data["table"]
    assert len(table) > 0
    # unigram probabilities should sum to ~1
    probs = list(table["*"].values())
    assert abs(sum(probs) - 1.0) < 0.01

def test_ngram_table_4gram(client):
    r = client.get("/api/ngram/table?n=4")
    assert r.status_code == 200
    assert "table" in r.json()

def test_ngram_table_5gram(client):
    r = client.get("/api/ngram/table?n=5")
    assert r.status_code == 200
    assert "table" in r.json()

def test_ngram_table_out_of_range(client):
    """n=6 should return 422 (above maximum of 5)."""
    r = client.get("/api/ngram/table?n=6")
    assert r.status_code == 422

def test_ngram_seed_fixed_reproducible(client):
    """Fixed seed produces the same result on repeated calls."""
    r1 = client.get("/api/ngram/generate?order=2&start=кот&words=5&seed=42")
    r2 = client.get("/api/ngram/generate?order=2&start=кот&words=5&seed=42")
    assert r1.json()["words"] == r2.json()["words"]

def test_ngram_seed_minus1_random(client):
    """seed=-1 produces different results across calls (on average)."""
    results = set()
    for _ in range(5):
        r = client.get("/api/ngram/generate?order=1&start=кот&words=5&seed=-1")
        results.add(tuple(r.json()["words"]))
    # at least 2 distinct results out of 5 attempts
    assert len(results) >= 2, "seed=-1 should produce different results"

def test_ngram_words_max_10(client):
    """Generate up to 10 words (may stop early at sentence boundary)."""
    r = client.get("/api/ngram/generate?order=2&start=кот&words=10&seed=42")
    assert r.status_code == 200
    assert 1 <= len(r.json()["words"]) <= 10

def test_ngram_words_min_1(client):
    """Generate 1 word."""
    r = client.get("/api/ngram/generate?order=2&start=кот&words=1&seed=42")
    assert r.status_code == 200
    assert len(r.json()["words"]) == 1


# --- RNN API (Step 5) ---
def test_rnn_generate_returns_words(client):
    r = client.get("/api/rnn/generate?hidden_size=8&start=кот&words=5")
    assert r.status_code == 200
    data = r.json()
    assert "words" in data
    assert isinstance(data["words"], list)
    assert len(data["words"]) == 5

def test_lstm_generate_returns_words(client):
    r = client.get("/api/rnn/lstm/generate?hidden_size=8&start=кот&words=5")
    assert r.status_code == 200
    data = r.json()
    assert "words" in data
    assert len(data["words"]) == 5

def test_rnn_hidden_sizes(client):
    for hs in [4, 8, 16, 32]:
        r = client.get(f"/api/rnn/generate?hidden_size={hs}&start=кот&words=3")
        assert r.status_code == 200, f"hidden_size={hs} returned {r.status_code}"

def test_lstm_hidden_sizes(client):
    for hs in [4, 8, 16, 32]:
        r = client.get(f"/api/rnn/lstm/generate?hidden_size={hs}&start=кот&words=3")
        assert r.status_code == 200, f"hidden_size={hs} returned {r.status_code}"

def test_rnn_words_in_vocab(client):
    """Generated words belong to the corpus vocabulary."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from corpus import TOKENS
    vocab = set(TOKENS)
    r = client.get("/api/rnn/generate?hidden_size=8&start=кот&words=5")
    for w in r.json()["words"]:
        assert w in vocab, f"Word '{w}' is not from the corpus"

def test_rnn_unknown_start_fallback(client):
    r = client.get("/api/rnn/generate?hidden_size=8&start=zzznonsense&words=3")
    assert r.status_code == 200
    data = r.json()
    assert "words" in data
    assert data.get("fallback_used") is True


# --- Embeddings API (Step 6) ---
def test_embeddings_map(client):
    r = client.get("/api/embeddings/map")
    assert r.status_code == 200
    data = r.json()
    assert "words" in data
    assert len(data["words"]) > 20
    # Each word has name, x, y, group
    w = data["words"][0]
    assert "word" in w
    assert "x" in w
    assert "y" in w
    assert "group" in w

def test_embeddings_analogy(client):
    r = client.post("/api/embeddings/analogy", json={"expression": "король - мужчина + женщина"})
    assert r.status_code == 200
    data = r.json()
    assert "result" in data
    assert "word" in data["result"]

def test_embeddings_analogy_empty_expression(client):
    r = client.post("/api/embeddings/analogy", json={"expression": ""})
    assert r.status_code == 200
    data = r.json()
    assert "error" in data

def test_embeddings_analogy_simple_addition(client):
    r = client.post("/api/embeddings/analogy", json={"expression": "кот + собака"})
    assert r.status_code == 200
    data = r.json()
    assert "result" in data

def test_embeddings_status(client):
    r = client.get("/api/embeddings/status")
    assert r.status_code == 200
    data = r.json()
    assert "ready" in data


# --- LLM-era API (Step 7) ---
def test_llm_status(client):
    r = client.get("/api/llm-era/status")
    assert r.status_code == 200
    data = r.json()
    assert "ready" in data
    assert "seq2seq_ready" in data
    assert "bert_ready" not in data  # BERT removed, attention uses GPT now

def test_llm_seq2seq_generate(client):
    r = client.get("/api/llm-era/seq2seq/generate?start=кот&words=5")
    assert r.status_code == 200
    data = r.json()
    assert "input" in data
    assert "words" in data
    assert isinstance(data["words"], list)
    assert len(data["words"]) >= 1
    assert data["bottleneck_size"] == 64

def test_llm_seq2seq_generate_default(client):
    r = client.get("/api/llm-era/seq2seq/generate")
    assert r.status_code == 200
    data = r.json()
    assert "words" in data
    assert len(data["words"]) >= 1

def test_llm_seq2seq_vocab(client):
    r = client.get("/api/llm-era/seq2seq/vocab")
    assert r.status_code == 200
    data = r.json()
    assert "vocab" in data
    assert len(data["vocab"]) > 10

def test_llm_attention(client):
    r = client.get("/api/llm-era/attention?sentence=кот сидел у окна и смотрел на дождь")
    assert r.status_code == 200
    data = r.json()
    assert "tokens" in data
    assert "weights" in data
    assert isinstance(data["weights"], list)
    n = len(data["tokens"])
    assert len(data["weights"]) == n
    assert len(data["weights"][0]) == n
    assert data["model"] == "YandexGPT-5-Lite-8B"
    assert data["causal"] is True
    assert data["total_heads"] > 0
    assert data["total_layers"] > 0

def test_llm_attention_default_sentence(client):
    """Attention endpoint works with no sentence (uses corpus default)."""
    r = client.get("/api/llm-era/attention")
    assert r.status_code == 200
    data = r.json()
    assert "tokens" in data
    assert "weights" in data

def test_llm_attention_specific_head(client):
    """Requesting a specific head returns non-averaged weights."""
    r = client.get("/api/llm-era/attention?sentence=кот сидел&head=1")
    assert r.status_code == 200
    data = r.json()
    assert data["head"] == 1
    assert len(data["tokens"]) > 0

def test_llm_attention_avg_heads(client):
    """head=0 returns averaged weights."""
    r = client.get("/api/llm-era/attention?sentence=кот сидел&head=0")
    assert r.status_code == 200
    data = r.json()
    assert data["head"] == 0

def test_llm_generate(client):
    r = client.post("/api/llm-era/generate", json={"prompt": "кот сидел у окна", "temperature": 0.7, "max_tokens": 20})
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    assert len(data["text"]) > 0

def test_llm_generate_temperature(client):
    """Different temperatures produce results."""
    for temp in [0.3, 0.7, 1.2]:
        r = client.post("/api/llm-era/generate", json={"prompt": "кот", "temperature": temp, "max_tokens": 10})
        assert r.status_code == 200
        assert "text" in r.json()

def test_llm_generate_no_fewshot_leak(client):
    """Few-shot prefix markers must not appear in the returned text."""
    r = client.post("/api/llm-era/generate", json={
        "prompt": "джон стрелял по замку из пистолета, чтобы",
        "temperature": 0.5,
        "max_tokens": 20,
    })
    assert r.status_code == 200
    data = r.json()
    assert "Начало:" not in data["text"]
    assert "Продолжение:" not in data["text"]
    assert data["text"].startswith(data["prompt"])
