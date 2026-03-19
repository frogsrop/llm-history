"""
routers/ngram.py — N-gram / Markov Chain API

Endpoints:
  GET /api/ngram/table?n=1|2|3   → transition table from corpus
  GET /api/ngram/generate?order=2&start=word&words=5 → word prediction
"""

import random
from collections import defaultdict
from typing import Dict, List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus import TOKENS

router = APIRouter()


def build_table(tokens: List[str], n: int) -> Dict[str, Dict[str, float]]:
    """
    Builds an N-gram transition table.
    Key: context (tuple of n-1 words; for n=1 each word stands alone).
    Value: dict {next_word: probability}.
    """
    if n == 1:
        # Unigram: simple word frequencies
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        total = sum(counts.values())
        return {"*": {w: round(c / total, 4) for w, c in counts.items()}}

    # Bigram / Trigram
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
    """
    Generates n_words words starting from start using N-gram of given order.
    """
    table = build_table(tokens, order)
    result = []

    if order == 1:
        dist = table.get("*", {})
        if not dist:
            return []
        words_pool = list(dist.keys())
        weights = list(dist.values())
        for _ in range(n_words):
            result.append(random.choices(words_pool, weights=weights, k=1)[0])
        return result

    # bigram / trigram
    context_words = start.split()
    context = " ".join(context_words[-(order - 1):])

    for _ in range(n_words):
        nexts = table.get(context, {})
        if not nexts:
            # fallback: random from all tokens
            result.append(random.choice(tokens))
            context_words.append(result[-1])
        else:
            next_word = random.choices(list(nexts.keys()), weights=list(nexts.values()), k=1)[0]
            result.append(next_word)
            context_words.append(next_word)
        context = " ".join(context_words[-(order - 1):])

    return result


@router.get("/table")
def ngram_table(n: int = Query(2, ge=1, le=5)):
    """Returns the N-gram transition table built from the corpus."""
    table = build_table(TOKENS, n)
    return {"n": n, "table": table, "vocab_size": len(set(TOKENS)), "token_count": len(TOKENS)}


@router.get("/generate")
def ngram_generate(
    order: int = Query(2, ge=1, le=5),
    start: str = Query("железо"),
    words: int = Query(5, ge=1, le=20),
    seed: int = Query(42),
):
    """Generates a continuation from the start word. seed=-1 → random."""
    if seed == -1:
        random.seed(None)  # system random source
    else:
        random.seed(seed)

    # If start is not in corpus — fallback to a random token
    if start not in TOKENS:
        start = random.choice(TOKENS)
        fallback = True
    else:
        fallback = False

    generated = generate_words(TOKENS, order, start, words)
    return {
        "start": start,
        "words": generated,
        "order": order,
        "fallback_used": fallback,
    }
