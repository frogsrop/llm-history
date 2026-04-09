"""
routers/llm_era.py — LLM-era module.

1. TinySeq2Seq: LSTM encoder-decoder trained on the corpus (CPU, fast).
   Shows the bottleneck problem: all info compressed to one vector.
2. Qwen2.5-3B: real attention heatmap + text generation.

Endpoints:
  GET  /api/llm-era/status              → model loading status
  GET  /api/llm-era/seq2seq/generate    → Seq2Seq generation (encoder→bottleneck→decoder)
  GET  /api/llm-era/attention?sentence=&layer=&head= → attention weights (per-head or averaged, from Qwen2.5-3B)
  POST /api/llm-era/generate            → text generation with temperature control
"""

import re
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import APIRouter
from pydantic import BaseModel

from corpus import SENTENCES

router = APIRouter()

# Few-shot prefix to prime Qwen2.5-3B into sentence-completion mode.
# Examples use longer continuations (finish sentence + one more) and avoid
# domain-specific words (пушка/пистолет) so the model isn't primed on them.
_FEW_SHOT_PREFIX = (
    "Продолжи текст — заверши предложение и добавь ещё одно.\n\n"
    "Начало: кот сидел у окна и смотрел\n"
    "Продолжение: на дождь за стеклом. Капли стучали по карнизу, и он щурился от каждого удара.\n\n"
    "Начало: девочка открыла старую книгу и начала\n"
    "Продолжение: читать вслух первую страницу. Буквы были крупные, а картинки яркие.\n\n"
    "Начало: поезд остановился на маленькой станции и\n"
    "Продолжение: двери открылись с тихим шипением. На перрон вышли всего два пассажира.\n\n"
    "Начало: {prompt}\n"
    "Продолжение:"
)

_MODELS_DIR = Path(__file__).parent.parent / "models"

# ═══════════════════════════════════════════════════════════════════════════════
# TinySeq2Seq — LSTM encoder-decoder with bottleneck
# ═══════════════════════════════════════════════════════════════════════════════

# Build vocabulary from corpus
_s2s_all_words: list[str] = []
for _s in SENTENCES:
    _s2s_all_words.extend(_s.split())
_S2S_VOCAB = sorted(set(_s2s_all_words))
_S2S_W2I = {w: i for i, w in enumerate(_S2S_VOCAB)}
_S2S_I2W = {i: w for w, i in _S2S_W2I.items()}
_S2S_V = len(_S2S_VOCAB)
_S2S_EOS_IDX = _S2S_V  # extra token for end-of-sequence
_S2S_TOTAL_V = _S2S_V + 1


class TinySeq2Seq(nn.Module):
    """Minimal LSTM encoder-decoder. The encoder compresses input to a single
    hidden state vector (the bottleneck). The decoder must reconstruct the
    target from that vector alone — no attention, no peeking."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.H = hidden_size
        self.encoder = nn.LSTM(input_size=_S2S_TOTAL_V, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=_S2S_TOTAL_V, hidden_size=hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, _S2S_TOTAL_V)

    def encode(self, src_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source words → (h, c) bottleneck."""
        x = F.one_hot(torch.tensor(src_indices), _S2S_TOTAL_V).float().unsqueeze(0)
        _, (h, c) = self.encoder(x)
        return h, c

    def decode_greedy(self, h: torch.Tensor, c: torch.Tensor, max_len: int) -> list[int]:
        """Greedy decode from bottleneck state."""
        result = []
        # First decoder input: zero vector (start token)
        inp = torch.zeros(1, 1, _S2S_TOTAL_V)
        for _ in range(max_len):
            out, (h, c) = self.decoder(inp, (h, c))
            logits = self.output_proj(out.squeeze(0))
            idx = logits.argmax(dim=-1).item()
            if idx == _S2S_EOS_IDX:
                break
            result.append(idx)
            inp = F.one_hot(torch.tensor([[idx]]), _S2S_TOTAL_V).float()
        return result


def _build_seq2seq_pairs() -> list[tuple[list[int], list[int]]]:
    """For each sentence, create pairs at every split point:
    prefix(1..n-1) → continuation. This teaches the model to continue
    from any starting word(s), not just midpoint."""
    pairs = []
    for sent in SENTENCES:
        words = sent.split()
        if len(words) < 3:
            continue
        for split in range(1, len(words)):
            src = [_S2S_W2I[w] for w in words[:split]]
            tgt = [_S2S_W2I[w] for w in words[split:]]
            pairs.append((src, tgt))
    return pairs


_seq2seq_model: TinySeq2Seq | None = None
_seq2seq_ready = threading.Event()
_S2S_CACHE_PATH = _MODELS_DIR / "seq2seq_model.pt"


def _train_seq2seq():
    global _seq2seq_model
    print("[Seq2Seq] Training TinySeq2Seq on corpus...", flush=True)
    torch.manual_seed(42)

    model = TinySeq2Seq(hidden_size=64)

    # Check for cached model
    if _S2S_CACHE_PATH.exists():
        try:
            model.load_state_dict(torch.load(_S2S_CACHE_PATH, weights_only=True))
            model.eval()
            _seq2seq_model = model
            _seq2seq_ready.set()
            print("[Seq2Seq] Loaded from cache.", flush=True)
            return
        except Exception:
            print("[Seq2Seq] Cache invalid, retraining...", flush=True)

    pairs = _build_seq2seq_pairs()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2000):
        total_loss = 0.0
        for src, tgt in pairs:
            optimizer.zero_grad()
            h, c = model.encode(src)

            # Teacher forcing: feed ground-truth target tokens
            tgt_with_eos = tgt + [_S2S_EOS_IDX]
            loss = torch.tensor(0.0)
            inp = torch.zeros(1, 1, _S2S_TOTAL_V)
            for t_idx in range(len(tgt_with_eos)):
                out, (h, c) = model.decoder(inp, (h, c))
                logits = model.output_proj(out.squeeze(0))
                target = torch.tensor([tgt_with_eos[t_idx]])
                loss = loss + criterion(logits, target)
                if t_idx < len(tgt):
                    inp = F.one_hot(torch.tensor([[tgt[t_idx]]]), _S2S_TOTAL_V).float()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 500 == 0:
            print(f"[Seq2Seq] epoch {epoch+1}/2000, loss={total_loss/len(pairs):.3f}", flush=True)

    model.eval()
    torch.save(model.state_dict(), _S2S_CACHE_PATH)
    _seq2seq_model = model
    _seq2seq_ready.set()
    print("[Seq2Seq] Training complete.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen2.5-7B-AWQ — text generation + attention (4-bit quantized)
# ═══════════════════════════════════════════════════════════════════════════════

_tokenizer = None
_model = None
_device = None
_n_heads = 0
_n_layers = 0
_model_ready = threading.Event()

_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
_CACHE_DIR = str(_MODELS_DIR / "qwen2.5-7b-awq")


def _load_model():
    global _tokenizer, _model, _device, _n_heads, _n_layers
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[LLM-era] Loading {_MODEL_ID} on {_device}...", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, cache_dir=_CACHE_DIR)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            cache_dir=_CACHE_DIR,
            torch_dtype=torch.float16,
        )
        _model.to(_device)
        _model.eval()
        _n_heads = _model.config.num_attention_heads
        _n_layers = _model.config.num_hidden_layers
        _model_ready.set()
        print(f"[LLM-era] Model ready on {_device} ({_n_layers} layers, {_n_heads} heads).", flush=True)
    except Exception as e:
        print(f"[LLM-era] Failed to load model: {e}", flush=True)


def start_loading():
    """Start model loading in background threads. Called from FastAPI startup."""
    threading.Thread(target=_train_seq2seq, daemon=True).start()
    threading.Thread(target=_load_model, daemon=True).start()


# ── Request models ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 40
    seed: int = 42


# ── Endpoints ───────────────────────────────────────────────────────────────
@router.get("/status")
def llm_status():
    return {
        "ready": _model_ready.is_set(),
        "seq2seq_ready": _seq2seq_ready.is_set(),
    }


@router.get("/seq2seq/generate")
def seq2seq_generate(start: str = "", words: int = 5):
    """Generate N words using Seq2Seq encoder→bottleneck→decoder.
    Same interface as N-gram and RNN: start word + number of words."""
    if not _seq2seq_ready.is_set():
        return {"error": "Seq2Seq is still training, please wait",
                "input": "", "words": []}

    if start.strip():
        input_words = start.strip().split()
    else:
        input_words = [SENTENCES[0].split()[0]]

    words = max(1, min(15, words))

    # Encode only known words
    src_indices = [_S2S_W2I[w] for w in input_words if w in _S2S_W2I]
    if not src_indices:
        return {"error": f"Unknown words: {', '.join(input_words)}",
                "input": start, "words": []}

    with torch.no_grad():
        h, c = _seq2seq_model.encode(src_indices)
        result_indices = _seq2seq_model.decode_greedy(h, c, max_len=words)
    output_words = [_S2S_I2W.get(i, "?") for i in result_indices]

    return {
        "input": " ".join(input_words),
        "words": output_words,
        "bottleneck_size": _seq2seq_model.H,
    }


@router.get("/seq2seq/vocab")
def seq2seq_vocab():
    """Return Seq2Seq vocabulary for the UI dropdown."""
    return {"vocab": _S2S_VOCAB}


@router.get("/attention")
def llm_attention(sentence: str = "", layer: int = -1, head: int = 0):
    """Attention heatmap from Qwen2.5-3B (causal, autoregressive).
    - layer: 1-based (default: last layer). 0 = average across all layers.
    - head:  1-based specific head. 0 = average across all heads in the layer.
    Qwen2.5 uses SentencePiece tokenizer with native Unicode support."""
    if not _model_ready.is_set():
        return {"error": "Model is still loading, please wait", "tokens": [], "weights": []}

    if not sentence.strip():
        sentence = SENTENCES[0]

    if layer < 0:
        layer = _n_layers  # default: last layer
    layer = max(0, min(_n_layers, layer))
    head = max(0, min(_n_heads, head))

    inputs = _tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True).to(_device)
    with torch.no_grad():
        outputs = _model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (batch, n_heads, seq_len, seq_len) per layer
    if layer == 0:
        # Average across all layers
        stacked = torch.stack(outputs.attentions)  # (n_layers, batch, n_heads, seq, seq)
        attn = stacked.mean(dim=0)[0]  # (n_heads, seq, seq)
    else:
        attn = outputs.attentions[layer - 1][0]  # (n_heads, seq, seq)

    if head == 0:
        # Average across all heads
        weights_tensor = attn.mean(dim=0)  # (seq, seq)
    else:
        weights_tensor = attn[head - 1]  # (seq, seq)

    # Decode each token individually for clean display.
    token_ids = inputs["input_ids"][0].tolist()
    display_tokens = [_tokenizer.decode([tid]).strip() for tid in token_ids]

    weights = weights_tensor.cpu().tolist()

    return {
        "tokens": display_tokens,
        "weights": weights,
        "model": "Qwen2.5-3B",
        "causal": True,
        "layer": layer,
        "total_layers": _n_layers,
        "head": head,
        "total_heads": _n_heads,
    }


@router.post("/generate")
def llm_generate(req: GenerateRequest):
    if not _model_ready.is_set():
        return {"error": "Model is still loading, please wait", "text": ""}

    prompt = req.prompt.strip()
    if not prompt:
        prompt = SENTENCES[0]

    temperature = max(0.1, min(2.0, req.temperature))
    max_tokens = max(1, min(100, req.max_tokens))

    seed = req.seed
    if seed == -1:
        seed = None
    torch.manual_seed(seed if seed is not None else torch.seed())

    full_prompt = _FEW_SHOT_PREFIX.format(prompt=prompt)
    inputs = _tokenizer(full_prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens (skip the few-shot prefix)
    input_len = inputs["input_ids"].shape[1]
    new_token_ids = output[0][input_len:]
    continuation = _tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    # Truncate at newline to prevent generating more few-shot examples
    if "\n" in continuation:
        continuation = continuation[: continuation.index("\n")].strip()

    # Keep up to 2 sentences (finish current + one more), drop the rest
    sentences = re.split(r'(?<=[.!?])\s+', continuation, maxsplit=2)
    if len(sentences) > 2:
        continuation = " ".join(sentences[:2])

    if not continuation:
        continuation = "(модель не смогла продолжить)"

    return {
        "text": prompt + " " + continuation,
        "prompt": prompt,
    }
