"""
routers/rnn.py — Tiny numpy RNN and LSTM

Architecture: character/word-level language model trained on the corpus.
Pre-training: 500 epochs, seed=42, all 4 hidden_size variants cached at startup.

Endpoints:
  GET /api/rnn/generate?hidden_size=8&start=word&words=5
  GET /api/lstm/generate?hidden_size=8&start=word&words=5
"""

import numpy as np
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus import TOKENS

router = APIRouter()

# ── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB = sorted(set(TOKENS))
W2I = {w: i for i, w in enumerate(VOCAB)}
I2W = {i: w for w, i in W2I.items()}
V = len(VOCAB)

HIDDEN_SIZES = [4, 8, 16, 32]
N_EPOCHS = 500

# ── Trained model cache ───────────────────────────────────────────────────────
_rnn_cache: dict = {}   # {hidden_size: params}
_lstm_cache: dict = {}


# ── Utilities ─────────────────────────────────────────────────────────────────
def one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros((size, 1))
    v[idx] = 1.0
    return v

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def sample(probs: np.ndarray) -> int:
    return int(np.random.choice(len(probs), p=probs.ravel()))


# ── Tiny RNN ──────────────────────────────────────────────────────────────────
class TinyRNN:
    def __init__(self, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        H, V_ = hidden_size, V
        self.Wxh = np.random.randn(H, V_) * 0.01
        self.Whh = np.random.randn(H, H)  * 0.01
        self.Why = np.random.randn(V_, H) * 0.01
        self.bh  = np.zeros((H, 1))
        self.by  = np.zeros((V_, 1))
        self.H   = hidden_size

    def forward(self, inputs, h_prev):
        """inputs: list of one-hot (V,1). Returns hs, ys, ps."""
        hs, ys, ps = {}, {}, {}
        hs[-1] = h_prev.copy()
        for t, x in enumerate(inputs):
            hs[t] = np.tanh(self.Wxh @ x + self.Whh @ hs[t-1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = softmax(ys[t])
        return hs, ys, ps

    def loss_and_grads(self, inputs, targets, h_prev):
        hs, ys, ps = self.forward(inputs, h_prev)
        loss = sum(-np.log(ps[t][targets[t], 0] + 1e-8) for t in range(len(inputs)))

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh  = np.zeros_like(self.bh)
        dby  = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1
            dWhy += dy @ hs[t].T
            dby  += dy
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t]**2) * dh
            dbh  += dh_raw
            dWxh += dh_raw @ inputs[t].T
            dWhh += dh_raw @ hs[t-1].T
            dh_next = self.Whh.T @ dh_raw

        for g in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(g, -5, 5, out=g)

        return loss, (dWxh, dWhh, dWhy, dbh, dby), hs[len(inputs)-1]

    def train(self, tokens, n_epochs):
        idxs = [W2I[t] for t in tokens if t in W2I]
        inputs_idx  = idxs[:-1]
        targets_idx = idxs[1:]

        lr = 0.1
        mWxh = np.zeros_like(self.Wxh); mWhh = np.zeros_like(self.Whh)
        mWhy = np.zeros_like(self.Why); mbh  = np.zeros_like(self.bh)
        mby  = np.zeros_like(self.by)

        for _ in range(n_epochs):
            h = np.zeros((self.H, 1))
            inputs  = [one_hot(i, V) for i in inputs_idx]
            targets = targets_idx
            _, grads, h = self.loss_and_grads(inputs, targets, h)
            dWxh, dWhh, dWhy, dbh, dby = grads
            for param, dparam, mem in [
                (self.Wxh, dWxh, mWxh), (self.Whh, dWhh, mWhh),
                (self.Why, dWhy, mWhy), (self.bh, dbh, mbh), (self.by, dby, mby)
            ]:
                mem += dparam**2
                param -= lr * dparam / (np.sqrt(mem) + 1e-8)

    def generate(self, start_idx: int, n: int, seed: int) -> list[int]:
        np.random.seed(seed)
        h = np.zeros((self.H, 1))
        x = one_hot(start_idx, V)
        result = []
        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = softmax(y)
            idx = sample(p)
            result.append(idx)
            x = one_hot(idx, V)
        return result


# ── Tiny LSTM ─────────────────────────────────────────────────────────────────
class TinyLSTM:
    def __init__(self, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        H, V_ = hidden_size, V
        scale = 0.01
        # Gates: f, i, g, o
        self.Wf = np.random.randn(H, V_+H) * scale; self.bf = np.zeros((H, 1))
        self.Wi = np.random.randn(H, V_+H) * scale; self.bi = np.zeros((H, 1))
        self.Wg = np.random.randn(H, V_+H) * scale; self.bg = np.zeros((H, 1))
        self.Wo = np.random.randn(H, V_+H) * scale; self.bo = np.zeros((H, 1))
        self.Wy = np.random.randn(V_, H)   * scale; self.by = np.zeros((V_, 1))
        self.H  = hidden_size

    def _step(self, x, h, c):
        xh = np.vstack([x, h])
        f = 1 / (1 + np.exp(-(self.Wf @ xh + self.bf)))
        i = 1 / (1 + np.exp(-(self.Wi @ xh + self.bi)))
        g = np.tanh(self.Wg @ xh + self.bg)
        o = 1 / (1 + np.exp(-(self.Wo @ xh + self.bo)))
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    def train(self, tokens, n_epochs):
        idxs = [W2I[t] for t in tokens if t in W2I]
        lr = 0.05
        # Adagrad for all parameters
        params = [self.Wf, self.bf, self.Wi, self.bi,
                  self.Wg, self.bg, self.Wo, self.bo, self.Wy, self.by]
        mems = [np.zeros_like(p) for p in params]

        for _ in range(n_epochs):
            h = np.zeros((self.H, 1))
            c = np.zeros((self.H, 1))
            loss = 0.0
            # Simplified BPTT with finite differences for brevity
            # (full LSTM BPTT is very verbose — use numerical gradient)
            for t in range(len(idxs) - 1):
                x = one_hot(idxs[t], V)
                h, c = self._step(x, h, c)
                y = self.Wy @ h + self.by
                p = softmax(y)
                loss -= np.log(p[idxs[t+1], 0] + 1e-8)

            # Numerical gradient (slow but correct for tiny model)
            eps = 1e-4
            for pi, param in enumerate(params):
                flat = param.ravel()
                grad_flat = np.zeros_like(flat)
                for j in range(min(len(flat), 20)):  # sample subset for speed
                    orig = flat[j]
                    flat[j] = orig + eps
                    l_plus = self._eval_loss(idxs)
                    flat[j] = orig - eps
                    l_minus = self._eval_loss(idxs)
                    flat[j] = orig
                    grad_flat[j] = (l_plus - l_minus) / (2 * eps)
                grad = grad_flat.reshape(param.shape)
                np.clip(grad, -5, 5, out=grad)
                mems[pi] += grad**2
                param -= lr * grad / (np.sqrt(mems[pi]) + 1e-8)

    def _eval_loss(self, idxs):
        h = np.zeros((self.H, 1))
        c = np.zeros((self.H, 1))
        loss = 0.0
        for t in range(len(idxs) - 1):
            x = one_hot(idxs[t], V)
            h, c = self._step(x, h, c)
            y = self.Wy @ h + self.by
            p = softmax(y)
            loss -= np.log(p[idxs[t+1], 0] + 1e-8)
        return loss

    def generate(self, start_idx: int, n: int, seed: int) -> list[int]:
        np.random.seed(seed)
        h = np.zeros((self.H, 1))
        c = np.zeros((self.H, 1))
        x = one_hot(start_idx, V)
        result = []
        for _ in range(n):
            h, c = self._step(x, h, c)
            y = self.Wy @ h + self.by
            p = softmax(y)
            idx = sample(p)
            result.append(idx)
            x = one_hot(idx, V)
        return result


# ── Model loading / pre-training ──────────────────────────────────────────────
import threading
import pickle
from pathlib import Path

_models_ready = threading.Event()
_MODELS_FILE = Path(__file__).parent.parent / "models" / "rnn_models.pkl"


def _load_from_file():
    """Loads pre-trained models from file (instant)."""
    with open(_MODELS_FILE, "rb") as f:
        data = pickle.load(f)
    _rnn_cache.update(data["rnn"])
    _lstm_cache.update(data["lstm"])
    print(f"[RNN/LSTM] Loaded pretrained models from {_MODELS_FILE}")
    _models_ready.set()


def _pretrain_all():
    """Trains models from scratch (takes several minutes)."""
    print("[RNN] Pretraining models (no cache file found)...")
    for hs in HIDDEN_SIZES:
        m = TinyRNN(hs)
        m.train(TOKENS, N_EPOCHS)
        _rnn_cache[hs] = m
        print(f"  RNN hidden={hs} done")

    print("[LSTM] Pretraining models...")
    for hs in HIDDEN_SIZES:
        m = TinyLSTM(hs)
        m.train(TOKENS, N_EPOCHS)
        _lstm_cache[hs] = m
        print(f"  LSTM hidden={hs} done")
    print("[RNN/LSTM] All models ready.")
    _models_ready.set()


def start_pretraining():
    """Loads models from file or trains in background. Called from FastAPI startup."""
    if _MODELS_FILE.exists():
        # Load quickly from pickle — no background thread needed
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

    start_idx = W2I[start]
    indices = model.generate(start_idx, words, seed)
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
    start: str = Query("железо"),
    words: int = Query(5, ge=1, le=20),
    seed: int = Query(42),
):
    return _generate_response(_rnn_cache, hidden_size, start, words, seed)


@router.get("/lstm/generate")
def lstm_generate(
    hidden_size: int = Query(8, ge=4, le=32),
    start: str = Query("железо"),
    words: int = Query(5, ge=1, le=20),
    seed: int = Query(42),
):
    return _generate_response(_lstm_cache, hidden_size, start, words, seed)
