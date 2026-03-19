"""
pretrain_models.py — one-time training of RNN/LSTM models.

Usage:
  python pretrain_models.py

Saves trained models to models/rnn_models.pkl (~1 MB).
Re-running overwrites the file.
"""

import pickle
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from corpus import TOKENS
from routers.rnn import TinyRNN, TinyLSTM, HIDDEN_SIZES, N_EPOCHS, W2I

MODELS_DIR = Path(__file__).parent / "models"
MODELS_FILE = MODELS_DIR / "rnn_models.pkl"


def pretrain():
    MODELS_DIR.mkdir(exist_ok=True)
    rnn_cache = {}
    lstm_cache = {}

    print(f"[pretrain] Training {len(HIDDEN_SIZES)} RNN models, {N_EPOCHS} epochs each...")
    t0 = time.time()
    for hs in HIDDEN_SIZES:
        t = time.time()
        m = TinyRNN(hs)
        m.train(TOKENS, N_EPOCHS)
        rnn_cache[hs] = m
        print(f"  RNN hidden={hs} done ({time.time()-t:.1f}s)")

    print(f"[pretrain] Training {len(HIDDEN_SIZES)} LSTM models...")
    for hs in HIDDEN_SIZES:
        t = time.time()
        m = TinyLSTM(hs)
        m.train(TOKENS, N_EPOCHS)
        lstm_cache[hs] = m
        print(f"  LSTM hidden={hs} done ({time.time()-t:.1f}s)")

    print(f"[pretrain] Total training time: {time.time()-t0:.1f}s")
    print(f"[pretrain] Saving to {MODELS_FILE}...")

    with open(MODELS_FILE, "wb") as f:
        pickle.dump({"rnn": rnn_cache, "lstm": lstm_cache}, f)

    print(f"[pretrain] Done! File size: {MODELS_FILE.stat().st_size // 1024} KB")


if __name__ == "__main__":
    pretrain()
