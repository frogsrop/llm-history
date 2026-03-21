"""
pretrain_models.py — one-time training of RNN/LSTM models (PyTorch).

Usage:
  python pretrain_models.py

Saves trained models to models/rnn_models.pt.
Re-running overwrites the file.
"""

import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from routers.rnn import TinyRNN, TinyLSTM, HIDDEN_SIZES, N_EPOCHS, W2I, DEVICE, TOKENS

MODELS_DIR = Path(__file__).parent / "models"
MODELS_FILE = MODELS_DIR / "rnn_models.pt"


def pretrain():
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"[pretrain] Device: {DEVICE}")

    rnn_cache = {}
    lstm_cache = {}

    print(f"[pretrain] Training {len(HIDDEN_SIZES)} RNN models, {N_EPOCHS} epochs each...")
    t0 = time.time()
    for hs in HIDDEN_SIZES:
        t = time.time()
        m = TinyRNN(hs).to(DEVICE)
        m.train_model(TOKENS, N_EPOCHS)
        rnn_cache[hs] = m.state_dict()
        print(f"  RNN hidden={hs} done ({time.time()-t:.1f}s)")

    print(f"[pretrain] Training {len(HIDDEN_SIZES)} LSTM models...")
    for hs in HIDDEN_SIZES:
        t = time.time()
        m = TinyLSTM(hs).to(DEVICE)
        m.train_model(TOKENS, N_EPOCHS)
        lstm_cache[hs] = m.state_dict()
        print(f"  LSTM hidden={hs} done ({time.time()-t:.1f}s)")

    print(f"[pretrain] Total training time: {time.time()-t0:.1f}s")
    print(f"[pretrain] Saving to {MODELS_FILE}...")

    torch.save({"rnn": rnn_cache, "lstm": lstm_cache}, MODELS_FILE)

    print(f"[pretrain] Done! File size: {MODELS_FILE.stat().st_size // 1024} KB")


if __name__ == "__main__":
    pretrain()
