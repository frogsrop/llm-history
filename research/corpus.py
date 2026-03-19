# Unified training corpus
#
# Variant B (blacksmith/iron) chosen by these criteria:
#   - the word "iron" appears 5 times → N-gram builds non-trivial tables
#   - hot/cold iron long-range dependency: RNN forgets the earlier adjective
#     by the time it predicts the verb
#   - hot vs cold, bends vs breaks → clear contrastive signal for embeddings
#   - blacksmith/iron/hammer — unified semantic group for Word2Vec

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
