# Unified training corpus
#
# Cat-and-rain corpus (10 sentences) chosen by these criteria:
#   - the words "кот" and "дождь" repeat across all sentences → N-gram builds non-trivial tables
#   - long-range dependencies (subject at start, verb at end) → RNN forgets context
#   - contrastive pairs: дремал/прыгнул, тёплый/дождь → useful for embeddings
#   - unified semantic group: кот/окно/печь/дождь → clear Word2Vec clusters

from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "corpus.txt"

SENTENCES = [
    line.strip()
    for line in _DATA_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# All sentences joined as one corpus
CORPUS = " ".join(SENTENCES)

# Corpus tokens (words)
TOKENS = CORPUS.split()
