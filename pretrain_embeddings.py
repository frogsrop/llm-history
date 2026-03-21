"""
pretrain_embeddings.py — one-time UMAP projection of fastText word vectors.

Usage:
  python pretrain_embeddings.py

Loads cc.ru.300.bin, picks top 50k Cyrillic words, runs UMAP to 2D,
saves result to models/umap_coords.npz.
Re-running overwrites the file.
"""

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from gensim.models.fasttext import load_facebook_vectors
from routers.embeddings import compute_umap, save_umap_cache

MODELS_DIR = Path(__file__).parent / "models"
FASTTEXT_PATH = MODELS_DIR / "cc.ru.300.bin"


def main():
    print(f"[pretrain_embeddings] Loading fastText from {FASTTEXT_PATH}...")
    t0 = time.time()
    model = load_facebook_vectors(str(FASTTEXT_PATH))
    print(f"[pretrain_embeddings] Model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    words, groups, coords, reducer = compute_umap(model)
    print(f"[pretrain_embeddings] UMAP computed in {time.time() - t1:.1f}s")

    save_umap_cache(words, coords)

    cache_path = MODELS_DIR / "umap_coords.npz"
    print(f"[pretrain_embeddings] Done! File size: {cache_path.stat().st_size // 1024} KB")
    print(f"[pretrain_embeddings] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
