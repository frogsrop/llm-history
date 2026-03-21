"""
download_models.py — one-time model download script.

Run once before first launch:
  python download_models.py

Downloads:
  1. ruGPT-3 XL (~5GB) from HuggingFace: ai-forever/rugpt3xl
  2. fastText Russian model cc.ru.300.bin.gz (~2.6GB)

Models are cached in the models/ directory next to this script.
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RUGPT3_MODEL_ID = "ai-forever/rugpt3xl"
FASTTEXT_RU_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz"
FASTTEXT_RU_GZ  = MODELS_DIR / "cc.ru.300.bin.gz"
FASTTEXT_RU_BIN = MODELS_DIR / "cc.ru.300.bin"

FASTTEXT_EN_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
FASTTEXT_EN_GZ  = MODELS_DIR / "cc.en.300.bin.gz"
FASTTEXT_EN_BIN = MODELS_DIR / "cc.en.300.bin"

# Active fastText model (used by embeddings router)
FASTTEXT_URL = FASTTEXT_EN_URL
FASTTEXT_GZ  = FASTTEXT_EN_GZ
FASTTEXT_BIN = FASTTEXT_EN_BIN


def download_rugpt3():
    """Downloads ruGPT-3 XL via HuggingFace transformers."""
    print("=" * 60)
    print(f"Downloading ruGPT-3 XL ({RUGPT3_MODEL_ID}) ...")
    print("Size: ~5 GB. This may take several minutes.")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: install transformers: pip install transformers")
        sys.exit(1)

    cache_dir = str(MODELS_DIR / "rugpt3xl")

    tokenizer = AutoTokenizer.from_pretrained(
        RUGPT3_MODEL_ID,
        cache_dir=cache_dir,
    )
    print("Tokenizer loaded.")

    model = AutoModelForCausalLM.from_pretrained(
        RUGPT3_MODEL_ID,
        cache_dir=cache_dir,
    )
    print("Model loaded.")

    # Quick smoke test
    import torch
    inputs = tokenizer("Hello,", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Generation test: '{decoded}'")
    print("ruGPT-3 XL — OK\n")


def download_fasttext():
    """Downloads fastText cc.ru.300.bin."""
    print("=" * 60)
    print("Downloading fastText cc.ru.300.bin.gz ...")
    print("Size: ~2.6 GB. This may take several minutes.")
    print("=" * 60)

    if FASTTEXT_BIN.exists():
        print(f"Already downloaded: {FASTTEXT_BIN}")
        _verify_fasttext()
        return

    import urllib.request

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            mb = count * block_size / 1_000_000
            print(f"\r  {pct:.1f}% ({mb:.0f} MB)", end="", flush=True)

    print(f"URL: {FASTTEXT_URL}")
    urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_GZ, reporthook)
    print()

    print("Extracting ...")
    import gzip
    import shutil
    with gzip.open(FASTTEXT_GZ, "rb") as f_in, open(FASTTEXT_BIN, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    FASTTEXT_GZ.unlink()  # remove archive after extraction
    print(f"Extracted: {FASTTEXT_BIN}")

    _verify_fasttext()


def _verify_fasttext():
    """Verifies that the fastText model loads correctly."""
    print("Verifying fastText model ...")
    try:
        import gensim.models.fasttext as ft
        model = ft.load_facebook_vectors(str(FASTTEXT_BIN))
        test_word = list(model.key_to_index.keys())[0]
        vec = model[test_word]
        print(f"Vector dim={len(vec)} for first vocab word")
        similar = model.most_similar(test_word, topn=3)
        print(f"Similar words: {[w for w, _ in similar]}")
        print("fastText — OK\n")
    except Exception as e:
        print(f"WARNING: {e}")
        print("Model downloaded but verification failed. Check manually.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download models for LLM Explainer")
    parser.add_argument("--rugpt3-only", action="store_true", help="Download ruGPT-3 XL only")
    parser.add_argument("--fasttext-only", action="store_true", help="Download fastText only")
    args = parser.parse_args()

    if args.rugpt3_only:
        download_rugpt3()
    elif args.fasttext_only:
        download_fasttext()
    else:
        download_rugpt3()
        download_fasttext()

    print("All models downloaded. Run: uvicorn main:app --reload")
