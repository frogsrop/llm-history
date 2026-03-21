"""
routers/embeddings.py — fastText word vectors, analogy API.

Loads cc.ru.300.bin (fastText Russian model), projects a curated VOCAB
to 2D via UMAP, and provides analogy arithmetic.

Endpoints:
  GET  /api/embeddings/map     → 2D word coordinates for canvas
  POST /api/embeddings/analogy → vector arithmetic result
  GET  /api/embeddings/status  → model loading status
"""

import re
import threading
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
import umap

router = APIRouter()

# ── Curated vocabulary (56 words in 6 semantic groups) ────────────────────────
VOCAB_GROUPS: dict[str, list[str]] = {
    "animals": [
        "кот", "собака", "лошадь", "рыба", "птица",
        "волк", "медведь", "кошка", "мышь", "лев",
    ],
    "people": [
        "мужчина", "женщина", "король", "королева",
        "ребёнок", "врач", "учитель", "студент",
    ],
    "actions": [
        "бежать", "прыгать", "спать", "есть",
        "говорить", "плыть", "летать", "идти",
    ],
    "nature": [
        "река", "гора", "лес", "море", "небо",
        "озеро", "поле", "дождь", "солнце", "ветер",
    ],
    "adjectives": [
        "большой", "маленький", "быстрый", "старый",
        "молодой", "сильный", "тёплый", "холодный",
    ],
    "places": [
        "Москва", "Париж", "Лондон", "Россия",
        "Франция", "Англия", "хлеб", "вода",
        "дом", "машина", "книга", "окно",
    ],
}

VOCAB: list[str] = []
WORD_TO_GROUP: dict[str, str] = {}
for group, words in VOCAB_GROUPS.items():
    for w in words:
        VOCAB.append(w)
        WORD_TO_GROUP[w] = group

# ── Model state ───────────────────────────────────────────────────────────────
_model = None
_coords_2d: dict[str, tuple[float, float]] = {}
_all_words: list[str] = []          # full word list (curated + frequent)
_all_word_groups: dict[str, str] = {}
_reducer = None                     # fitted UMAP (only set when computed, not from cache)
_model_ready = threading.Event()
_MODELS_DIR = Path(__file__).parent.parent / "models"

# Regex: word is pure Cyrillic (no digits, punctuation, mixed scripts)
_WORD_RE = re.compile(r'^[а-яёА-ЯЁ]+$')
MAP_SIZE = 50_000


def _pick_frequent_words(model, n: int) -> list[str]:
    """Pick top-n frequent Cyrillic words from fastText vocab, excluding curated VOCAB."""
    curated = set(VOCAB)
    result = []
    for word in model.index_to_key:
        if len(result) >= n:
            break
        if word in curated:
            continue
        if len(word) < 2 or len(word) > 20:
            continue
        if not _WORD_RE.match(word):
            continue
        result.append(word)
    return result


_UMAP_CACHE = _MODELS_DIR / "umap_coords.npz"


def compute_umap(model):
    """Build word list and run UMAP from scratch. Returns (words, groups, coords_2d, reducer)."""
    extra = _pick_frequent_words(model, MAP_SIZE - len(VOCAB))
    words = list(VOCAB) + extra
    groups = dict(WORD_TO_GROUP)
    for w in extra:
        groups[w] = "frequent"

    print(f"[Embeddings] {len(VOCAB)} curated + {len(extra)} frequent = {len(words)} words", flush=True)

    vectors = np.array([model[w] for w in words])
    print("[Embeddings] Running UMAP (this may take ~1 min)...", flush=True)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(vectors)
    return words, groups, coords, reducer


def save_umap_cache(words, coords):
    """Save precomputed UMAP to disk."""
    np.savez_compressed(
        _UMAP_CACHE,
        words=np.array(words, dtype=object),
        coords=coords,
    )
    print(f"[Embeddings] UMAP cache saved to {_UMAP_CACHE}", flush=True)


def _load_model():
    global _model, _coords_2d, _all_words, _all_word_groups
    global _reducer
    model_path = _MODELS_DIR / "cc.ru.300.bin"
    if not model_path.exists():
        print(f"[Embeddings] Model not found at {model_path}")
        return

    from gensim.models.fasttext import load_facebook_vectors

    print("[Embeddings] Loading fastText model...", flush=True)
    _model = load_facebook_vectors(str(model_path))
    print("[Embeddings] Model loaded.", flush=True)

    # Try loading precomputed UMAP cache
    if _UMAP_CACHE.exists():
        print(f"[Embeddings] Loading UMAP cache from {_UMAP_CACHE}...", flush=True)
        data = np.load(_UMAP_CACHE, allow_pickle=True)
        words = list(data["words"])
        coords = data["coords"]

        _all_words.clear()
        _all_words.extend(words)
        _all_word_groups.update(WORD_TO_GROUP)
        for w in words:
            if w not in WORD_TO_GROUP:
                _all_word_groups[w] = "frequent"
    else:
        print(f"[Embeddings] No cache at {_UMAP_CACHE}, computing UMAP...", flush=True)
        print("[Embeddings] Run 'python pretrain_embeddings.py' to pre-compute.", flush=True)
        words, groups, coords, _reducer = compute_umap(_model)
        _all_words.clear()
        _all_words.extend(words)
        _all_word_groups.update(groups)
        save_umap_cache(words, coords)

    # Normalize to [0, 1] range
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1
    normed = (coords - mins) / span

    _coords_2d.clear()
    for i, w in enumerate(_all_words):
        _coords_2d[w] = (float(normed[i, 0]), float(normed[i, 1]))

    _model_ready.set()
    print(f"[Embeddings] Ready: {len(_all_words)} words on map.", flush=True)


def start_loading():
    """Start model loading in a background thread. Called from FastAPI startup."""
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()


# ── Request / Response models ─────────────────────────────────────────────────
class AnalogyRequest(BaseModel):
    expression: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/status")
def embeddings_status():
    return {"ready": _model_ready.is_set()}


@router.get("/map")
def embeddings_map():
    if not _model_ready.is_set():
        return {"error": "Model is still loading, please wait", "words": []}

    words = []
    for w in _all_words:
        x, y = _coords_2d[w]
        words.append({
            "word": w,
            "x": x,
            "y": y,
            "group": _all_word_groups.get(w, "frequent"),
        })
    return {"words": words}


@router.post("/analogy")
def embeddings_analogy(req: AnalogyRequest):
    if not _model_ready.is_set():
        return {"error": "Model is still loading, please wait"}

    # Parse expression: "король - мужчина + женщина"
    expr = req.expression.strip()
    if not expr:
        return {"error": "Empty expression"}

    # Tokenize: split by + and - while keeping the operator
    tokens = re.split(r'\s*([+\-])\s*', expr)
    tokens = [t.strip() for t in tokens if t.strip()]

    positive = []
    negative = []
    sign = "+"

    for token in tokens:
        if token in ("+", "-"):
            sign = token
            continue
        word = token
        if word not in _model:
            return {"error": f"Слово «{word}» не найдено в модели"}
        if sign == "+":
            positive.append(word)
        else:
            negative.append(word)

    if not positive and not negative:
        return {"error": "No words in expression"}

    try:
        # Exclude input words from results
        exclude = {w.lower() for w in positive + negative}
        results = _model.most_similar(
            positive=positive or None,
            negative=negative or None,
            topn=5,
        )

        # Find first result not in input words
        best = None
        for word, score in results:
            if word.lower() not in exclude:
                best = (word, score)
                break
        if best is None and results:
            best = results[0]

        if best is None:
            return {"error": "No result found"}

        result_word, score = best

        return {
            "result": {
                "word": result_word,
                "score": round(score, 4),
            },
            "top5": [{"word": w, "score": round(s, 4)} for w, s in results[:5]],
        }
    except Exception as e:
        return {"error": str(e)}
