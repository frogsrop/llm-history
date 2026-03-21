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
# def test_llm_attention(client):
#     r = client.get("/api/llm-era/attention?sentence=the cat sat on the mat")
#     assert r.status_code == 200

# def test_llm_generate(client):
#     r = client.post("/api/llm-era/generate", json={"prompt": "the cat sat"})
#     assert r.status_code == 200
