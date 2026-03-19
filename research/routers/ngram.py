"""
routers/ngram.py — N-gram / Markov Chain API

Endpoints:
  GET /api/ngram/table?n=1|2|3   → transition table from corpus
  GET /api/ngram/generate?order=2&start=word&words=5 → word prediction

Sentence boundaries: each sentence is wrapped with (order-1) <s> tokens
at the start and one </s> token at the end, so N-gram contexts never
cross sentence boundaries.
"""

import random
from collections import defaultdict
from typing import Dict, List

from fastapi import APIRouter, Query

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus import SENTENCES

router = APIRouter()

BOS = "<s>"
EOS = "</s>"


def tokens_for_order(order: int) -> List[str]:
    """Build flat token list with sentence boundary tokens.

    Each sentence gets (order-1) BOS tokens prepended and one EOS appended,
    so N-gram contexts never leak across sentence boundaries.
    """
    result = []
    for sent in SENTENCES:
        result.extend([BOS] * max(0, order - 1))
        result.extend(sent.split())
        result.append(EOS)
    return result


def build_table(tokens: List[str], n: int) -> Dict[str, Dict[str, float]]:
    """Builds an N-gram transition table."""
    if n == 1:
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        total = sum(counts.values())
        return {"*": {w: round(c / total, 4) for w, c in counts.items()}}

    context_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - n + 1):
        context = " ".join(tokens[i: i + n - 1])
        next_word = tokens[i + n - 1]
        context_counts[context][next_word] += 1

    table = {}
    for context, nexts in context_counts.items():
        total = sum(nexts.values())
        table[context] = {w: round(c / total, 4) for w, c in sorted(
            nexts.items(), key=lambda x: -x[1]
        )}
    return table


def generate_words(tokens: List[str], order: int, start: str, n_words: int) -> List[str]:
    """Generates up to n_words words; stops early at EOS."""
    table = build_table(tokens, order)
    result = []
    regular = [t for t in tokens if t not in (BOS, EOS)]

    if order == 1:
        dist = table.get("*", {})
        if not dist:
            return []
        words_pool = list(dist.keys())
        weights = list(dist.values())
        for _ in range(n_words):
            w = random.choices(words_pool, weights=weights, k=1)[0]
            result.append(w)
            if w == EOS:
                break
        return result

    context_words = start.split()
    context = " ".join(context_words[-(order - 1):])

    for _ in range(n_words):
        nexts = table.get(context, {})
        if not nexts:
            next_word = random.choice(regular) if regular else BOS
        else:
            next_word = random.choices(
                list(nexts.keys()), weights=list(nexts.values()), k=1
            )[0]
        result.append(next_word)
        context_words.append(next_word)
        context = " ".join(context_words[-(order - 1):])
        if next_word == EOS:
            break

    return result


@router.get("/table")
def ngram_table(n: int = Query(2, ge=1, le=5)):
    """Returns the N-gram transition table built from the corpus."""
    tokens = tokens_for_order(n)
    table = build_table(tokens, n)
    return {"n": n, "table": table, "vocab_size": len(set(tokens)), "token_count": len(tokens)}


@router.get("/generate")
def ngram_generate(
    order: int = Query(2, ge=1, le=5),
    start: str = Query("кот"),
    words: int = Query(5, ge=1, le=20),
    seed: int = Query(42),
):
    """Generates a continuation from the start word. seed=-1 → random."""
    if seed == -1:
        random.seed(None)
    else:
        random.seed(seed)

    tokens = tokens_for_order(order)
    regular = [t for t in tokens if t not in (BOS, EOS)]

    if start not in tokens:
        start = random.choice(regular) if regular else tokens[0]
        fallback = True
    else:
        fallback = False

    generated = generate_words(tokens, order, start, words)
    return {
        "start": start,
        "words": generated,
        "order": order,
        "fallback_used": fallback,
    }
