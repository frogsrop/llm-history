"""
test_corpus.py — training corpus quality checks.

Criteria (from PLAN.md):
  - ~8–12 words per sentence
  - Repeated words (N-gram can build meaningful tables)
  - Long-range dependencies (RNN will "forget" the start)
  - Words with similar meaning / polysemous words
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus import SENTENCES, TOKENS


def test_three_sentences():
    assert len(SENTENCES) == 3, "Corpus must contain exactly 3 sentences"


def test_sentence_length():
    for s in SENTENCES:
        words = s.split()
        assert 8 <= len(words) <= 14, (
            f"Sentence '{s}' contains {len(words)} words, expected 8–14"
        )


def test_repeated_words():
    """At least one word appears 3+ times in the tokens (required for N-gram tables)."""
    from collections import Counter
    counts = Counter(TOKENS)
    max_count = max(counts.values())
    assert max_count >= 3, (
        f"No words with frequency ≥3. Maximum: {max_count}. "
        "N-gram tables will be degenerate."
    )


def test_long_range_dependency():
    """Each sentence has words separated by ≥5 positions that are semantically
    related (sentence length is used as an indirect criterion)."""
    for s in SENTENCES:
        words = s.split()
        assert len(words) >= 9, (
            f"Sentence is too short for long-range dependencies: '{s}'"
        )


def test_vocabulary_size():
    """Vocabulary must not be too small (otherwise N-gram is trivial)."""
    vocab = set(TOKENS)
    assert len(vocab) >= 10, f"Vocabulary is too small: {len(vocab)} unique words"


def test_corpus_is_russian():
    """All sentences contain Cyrillic characters."""
    for s in SENTENCES:
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in s)
        assert has_cyrillic, f"Sentence is not in Russian: '{s}'"
