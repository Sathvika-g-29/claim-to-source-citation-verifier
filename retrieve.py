"""Step 3: pick the passages of a source most relevant to a claim.

Uses a small open embedding model; falls back to keyword overlap if it cannot load (offline, etc.).
"""
import math
import re

import numpy as np

_model = None
_model_failed = False
_emb_cache = {}
_STOP = set("the a an of to in and or for on with by is are was were be as at that this it from which these those".split())


def warmup():
    """Load the embedding model in the background. Until it is ready, keyword retrieval is used."""
    import threading

    def run():
        global _model, _model_failed
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[retrieve] embedding model ready", flush=True)
        except Exception as e:
            _model_failed = True
            print(f"[retrieve] embeddings unavailable ({e}); using keyword retrieval", flush=True)
    threading.Thread(target=run, daemon=True).start()


def _load():
    return _model  # None until warmup() finishes


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 1]


def _keyword_scores(claim, chunks):
    q = set(_tokens(claim))
    n = len(chunks)
    df = {}
    toks = [set(_tokens(c)) for c in chunks]
    for t in toks:
        for w in t:
            df[w] = df.get(w, 0) + 1
    scores = []
    for t in toks:
        s = sum(math.log(1 + n / df[w]) for w in q & t)
        scores.append(s / (1 + 0.1 * len(t)))
    return np.array(scores)


def top_k(claim, chunks, k=4):
    """Return [(chunk_index, score)] best first."""
    if not chunks:
        return []
    model = _load()
    if model is not None:
        ck = hash(tuple(chunks))
        if ck not in _emb_cache:
            _emb_cache[ck] = model.encode(chunks, normalize_embeddings=True, batch_size=32)
        q = model.encode([claim], normalize_embeddings=True)[0]
        scores = _emb_cache[ck] @ q
    else:
        scores = _keyword_scores(claim, chunks)
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]
