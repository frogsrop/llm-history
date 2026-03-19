"""
routers/rnn.py — TinyRNN and TinyLSTM implemented in PyTorch.

Architecture: word-level language model trained on the corpus.
Pre-training: 500 epochs, seed=42, all 4 hidden_size variants cached at startup.

Endpoints:
  GET /api/rnn/generate?hidden_size=8&start=word&words=5
  GET /api/rnn/lstm/generate?hidden_size=8&start=word&words=5
  GET /api/rnn/status
"""

import threading
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import APIRouter, Query

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus import TOKENS

router = APIRouter()

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB = sorted(set(TOKENS))
W2I = {w: i for i, w in enumerate(VOCAB)}
I2W = {i: w for w, i in W2I.items()}
V = len(VOCAB)

HIDDEN_SIZES = [4, 8, 16, 32]
N_EPOCHS = 500

# ── Model cache ───────────────────────────────────────────────────────────────
_rnn_cache: dict = {}
_lstm_cache: dict = {}


# ── TinyRNN ───────────────────────────────────────────────────────────────────
class TinyRNN(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.H    = hidden_size
        self.cell = nn.RNNCell(V, hidden_size)
        self.out  = nn.Linear(hidden_size, V)

    def train_model(self, tokens: list[str], n_epochs: int):
        idxs = torch.tensor(
            [W2I[t] for t in tokens if t in W2I], dtype=torch.long, device=DEVICE
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            h = torch.zeros(1, self.H, device=DEVICE)
            loss = torch.tensor(0.0, device=DEVICE)
            for t in range(len(idxs) - 1):
                x = F.one_hot(idxs[t].unsqueeze(0), V).float()
                h = self.cell(x, h)
                logits = self.out(h)
                loss = loss + criterion(logits, idxs[t + 1].unsqueeze(0))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
        self.eval()

    def generate(self, start_idx: int, n: int, seed: int) -> list[int]:
        torch.manual_seed(seed)
        result = []
        with torch.no_grad():
            h = torch.zeros(1, self.H, device=DEVICE)
            x = F.one_hot(torch.tensor([start_idx], device=DEVICE), V).float()
            for _ in range(n):
                h = self.cell(x, h)
                logits = self.out(h)
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                result.append(idx)
                x = F.one_hot(torch.tensor([idx], device=DEVICE), V).float()
        return result


# ── TinyLSTM ──────────────────────────────────────────────────────────────────
class TinyLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.H    = hidden_size
        self.cell = nn.LSTMCell(V, hidden_size)
        self.out  = nn.Linear(hidden_size, V)

    def train_model(self, tokens: list[str], n_epochs: int):
        idxs = torch.tensor(
            [W2I[t] for t in tokens if t in W2I], dtype=torch.long, device=DEVICE
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            h = torch.zeros(1, self.H, device=DEVICE)
            c = torch.zeros(1, self.H, device=DEVICE)
            loss = torch.tensor(0.0, device=DEVICE)
            for t in range(len(idxs) - 1):
                x = F.one_hot(idxs[t].unsqueeze(0), V).float()
                h, c = self.cell(x, (h, c))
                logits = self.out(h)
                loss = loss + criterion(logits, idxs[t + 1].unsqueeze(0))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
        self.eval()

    def generate(self, start_idx: int, n: int, seed: int) -> list[int]:
        torch.manual_seed(seed)
        result = []
        with torch.no_grad():
            h = torch.zeros(1, self.H, device=DEVICE)
            c = torch.zeros(1, self.H, device=DEVICE)
            x = F.one_hot(torch.tensor([start_idx], device=DEVICE), V).float()
            for _ in range(n):
                h, c = self.cell(x, (h, c))
                logits = self.out(h)
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                result.append(idx)
                x = F.one_hot(torch.tensor([idx], device=DEVICE), V).float()
        return result


# ── Model loading / pre-training ──────────────────────────────────────────────
_models_ready = threading.Event()
_MODELS_FILE = Path(__file__).parent.parent / "models" / "rnn_models.pt"


def _load_from_file():
    data = torch.load(_MODELS_FILE, map_location=DEVICE, weights_only=True)
    for hs, sd in data["rnn"].items():
        m = TinyRNN(hs).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        _rnn_cache[hs] = m
    for hs, sd in data["lstm"].items():
        m = TinyLSTM(hs).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        _lstm_cache[hs] = m
    print(f"[RNN/LSTM] Loaded pretrained models from {_MODELS_FILE} (device={DEVICE})")
    _models_ready.set()


def _pretrain_all():
    print(f"[RNN/LSTM] Training from scratch on device={DEVICE}...")
    for hs in HIDDEN_SIZES:
        m = TinyRNN(hs).to(DEVICE)
        m.train_model(TOKENS, N_EPOCHS)
        _rnn_cache[hs] = m
        print(f"  RNN hidden={hs} done")
    for hs in HIDDEN_SIZES:
        m = TinyLSTM(hs).to(DEVICE)
        m.train_model(TOKENS, N_EPOCHS)
        _lstm_cache[hs] = m
        print(f"  LSTM hidden={hs} done")
    print("[RNN/LSTM] All models ready.")
    _models_ready.set()


def start_pretraining():
    """Load from .pt file or train in background. Called from FastAPI startup."""
    if _MODELS_FILE.exists():
        _load_from_file()
    else:
        print(f"[RNN/LSTM] No cache at {_MODELS_FILE}, training in background...")
        print("[RNN/LSTM] Run 'python pretrain_models.py' to pre-train once.")
        t = threading.Thread(target=_pretrain_all, daemon=True)
        t.start()


# ── Endpoints ─────────────────────────────────────────────────────────────────
def _generate_response(cache, hidden_size, start, words, seed):
    if not _models_ready.is_set():
        return {"error": "Models are still training, please wait"}

    if start not in W2I:
        import random
        start = random.choice(VOCAB)
        fallback = True
    else:
        fallback = False

    model = cache.get(hidden_size)
    if model is None:
        return {"error": f"hidden_size={hidden_size} not cached"}

    indices = model.generate(W2I[start], words, seed)
    generated = [I2W[i] for i in indices]

    return {
        "start": start,
        "words": generated,
        "hidden_size": hidden_size,
        "fallback_used": fallback,
    }


@router.get("/status")
def rnn_status():
    return {"ready": _models_ready.is_set()}


@router.get("/generate")
def rnn_generate(
    hidden_size: int = Query(8, ge=4, le=32),
    start: str = Query("кот"),
    words: int = Query(5, ge=1, le=20),
    seed: int = Query(42),
):
    return _generate_response(_rnn_cache, hidden_size, start, words, seed)


@router.get("/lstm/generate")
def lstm_generate(
    hidden_size: int = Query(8, ge=4, le=32),
    start: str = Query("кот"),
    words: int = Query(5, ge=1, le=20),
    seed: int = Query(42),
):
    return _generate_response(_lstm_cache, hidden_size, start, words, seed)
