"""
utils.py — shared corpus utilities.

Sentence boundary tokens and token-list builder used by all routers.
"""

from corpus import SENTENCES

BOS = "<s>"
EOS = "</s>"


def make_tokens(order: int = 2) -> list[str]:
    """Return flat token list with sentence boundary markers.

    Each sentence gets (order-1) BOS tokens prepended and one EOS appended,
    so N-gram contexts and RNN training never cross sentence boundaries.

    For RNN (order=1): only EOS is added between sentences.
    """
    result = []
    for sent in SENTENCES:
        result.extend([BOS] * max(0, order - 1))
        result.extend(sent.split())
        result.append(EOS)
    return result
