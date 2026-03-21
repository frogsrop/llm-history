import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# HuggingFace token — set via environment variable HF_TOKEN before running
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN environment variable before running this script")

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── ruGPT-3 large (~1.5 GB, standard HF format) ─────────────
# rugpt3xl uses DeepSpeed format (mp_rank_00_model_states.pt)
# and is incompatible with AutoModelForCausalLM — using large instead.
RUGPT3_ID    = "ai-forever/rugpt3large_based_on_gpt2"
RUGPT3_CACHE = str(MODELS_DIR / "rugpt3large")

print("=== Downloading ruGPT-3 large ===")
print(f"Model : {RUGPT3_ID}")
print(f"Cache : {RUGPT3_CACHE}")
print("Size  : ~1.5 GB. Please wait...")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(
    RUGPT3_ID, cache_dir=RUGPT3_CACHE, token=HF_TOKEN
)
print("[OK] tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    RUGPT3_ID, cache_dir=RUGPT3_CACHE, token=HF_TOKEN
)
print("[OK] model loaded")

# quick smoke test
inputs = tokenizer("Koshka", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
decoded = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"[OK] generation test: '{decoded}'")
print("=== ruGPT-3 large: DONE ===\n")

# ── fastText cc.ru.300.bin (~2.6 GB) ─────────────────────────
import urllib.request, gzip, shutil

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz"
FASTTEXT_GZ  = MODELS_DIR / "cc.ru.300.bin.gz"
FASTTEXT_BIN = MODELS_DIR / "cc.ru.300.bin"

print("=== Downloading fastText cc.ru.300.bin ===")

if FASTTEXT_BIN.exists():
    print(f"[SKIP] already exists: {FASTTEXT_BIN}")
else:
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            mb  = count * block_size / 1_000_000
            print(f"\r  {pct:.1f}%  ({mb:.0f} MB)   ", end="", flush=True)

    print(f"URL: {FASTTEXT_URL}")
    urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_GZ, reporthook)
    print()

    print("Extracting...")
    with gzip.open(FASTTEXT_GZ, "rb") as f_in, open(FASTTEXT_BIN, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    FASTTEXT_GZ.unlink()
    print(f"[OK] extracted: {FASTTEXT_BIN}")

print("Verifying fastText...")
from gensim.models.fasttext import load_facebook_vectors
ft = load_facebook_vectors(str(FASTTEXT_BIN))
vec = ft["kot"]
similar = ft.most_similar("kot", topn=3)
print(f"[OK] dim={len(vec)}, similar to 'kot': {[w for w,_ in similar]}")
print("=== fastText: DONE ===")
print("\nAll models ready. Run: uvicorn main:app --reload")
