"""Step 4: judge each claim against its retrieved source. Four verdicts, evidence-grounded."""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from claims import extract_claims
from llm import chat_json
from retrieve import top_k
from sources import chunk_text, get_source

VERDICTS = ["SUPPORTED", "WEAK", "MISMATCHED", "UNVERIFIABLE"]
RANK = {"SUPPORTED": 3, "WEAK": 2, "MISMATCHED": 1, "UNVERIFIABLE": 0}
SCORE = {"SUPPORTED": 1.0, "WEAK": 0.5, "MISMATCHED": 0.0}

PROMPT = """You are a careful peer reviewer checking whether a cited source supports a claim.

CLAIM (as written in the document): {claim}
CLAIM (atomic): {atomic}
CITED SOURCE: {entry}
EVIDENCE AVAILABLE: {level}

PASSAGES retrieved from the cited source:
{passages}

Decide ONE verdict:
- SUPPORTED: the passages clearly state or directly imply the claim, including its specifics (numbers, scope, strength).
- WEAK: on topic, but the passages only partly back the claim: narrower scope, hedged wording, indirect, a different number or scale, or the claim is stronger than the evidence.
- MISMATCHED: the source is irrelevant to the claim, or the passages contradict it.
- UNVERIFIABLE: the passages are not enough to decide either way (e.g. only an abstract, or the relevant part is not shown).

Rules: judge only from the passages. The passages are excerpts of the full source, so if EVIDENCE AVAILABLE is fulltext and a
passage states the claim's key facts (same result, same numbers), choose SUPPORTED. Use UNVERIFIABLE only when no passage is on topic. Overstated claims ("always", "eliminates", "every") backed by hedged
evidence are WEAK or MISMATCHED. Copy "evidence_quote" EXACTLY from one passage (one sentence, max 40 words),
or use "" if none is relevant.

Return JSON: {{"verdict": "SUPPORTED|WEAK|MISMATCHED|UNVERIFIABLE", "confidence": 0.0-1.0,
"evidence_quote": "...", "reason": "one short sentence explaining the verdict"}}"""


def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def _quote_in(quote, text):
    q = _norm(quote)
    return bool(q) and q in _norm(text)


def verify_one(claim, cit, user_sources=None):
    src = get_source(cit, user_sources)
    res = {"key": cit["key"], "entry": cit.get("entry", ""), "origin": src["origin"],
           "evidence_level": src["evidence_level"], "verdict": "UNVERIFIABLE", "confidence": 1.0,
           "evidence_quote": "", "quote_verified": False, "reason": "", "passages": []}
    if src["evidence_level"] == "none":
        res["reason"] = "The cited source could not be retrieved (no accessible full text, DOI, arXiv id or URL)."
        return res
    chunks = chunk_text(src["text"])
    hits = top_k(claim["claim_atomic"] or claim["claim"], chunks, k=6)
    passages = [chunks[i] for i, _ in hits]
    res["passages"] = passages
    if not passages:
        res["reason"] = "The retrieved source had no usable text."
        return res
    block = "\n".join(f"[P{n + 1}] {p}" for n, p in enumerate(passages))
    try:
        out = chat_json(PROMPT.format(claim=claim["claim"], atomic=claim["claim_atomic"], entry=cit.get("entry", ""),
                                      level=src["evidence_level"], passages=block), key="verdict")
    except Exception as e:
        res["reason"] = f"Model error: {e}"
        res["confidence"] = 0.0
        return res
    v = str(out.get("verdict", "")).upper().strip()
    res["verdict"] = v if v in VERDICTS else "UNVERIFIABLE"
    try:
        res["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        res["confidence"] = 0.5
    res["evidence_quote"] = str(out.get("evidence_quote", "")).strip()
    res["reason"] = str(out.get("reason", "")).strip()
    # Anti-hallucination guard: the quoted evidence must really exist in the source.
    res["quote_verified"] = _quote_in(res["evidence_quote"], src["text"])
    if res["evidence_quote"] and not res["quote_verified"]:
        res["evidence_quote"] = ""
        res["confidence"] = min(res["confidence"], 0.4)
        res["reason"] += " (Model's quote could not be found in the source, so it was discarded.)"
    print(f"[verify] ref {cit['key']} {src['evidence_level']} -> {res['verdict']} ({res['confidence']:.2f}) quote_found={res['quote_verified']}", flush=True)
    # SUPPORTED must be backed by a quote that really exists in the source.
    if res["verdict"] == "SUPPORTED" and not res["quote_verified"]:
        res["verdict"] = "WEAK"
        res["reason"] += " (Downgraded: no verifiable supporting quote.)"
    # Abstract-only evidence can never be high-confidence.
    if src["evidence_level"] == "abstract_only":
        res["confidence"] = min(res["confidence"], 0.7)
        if res["verdict"] == "SUPPORTED" and res["confidence"] < 0.5:
            res["verdict"] = "WEAK"
    return res


def verify_claim(claim, user_sources=None):
    results = [verify_one(claim, c, user_sources) for c in claim["citations"]]
    if not results:
        results = [{"key": "-", "entry": "", "origin": "", "evidence_level": "none", "verdict": "UNVERIFIABLE",
                    "confidence": 1.0, "evidence_quote": "", "quote_verified": False, "passages": [],
                    "reason": "The sentence cites a reference that is not in the bibliography."}]
    best = max(results, key=lambda r: (RANK[r["verdict"]], r["confidence"]))
    return {**claim, "verdict": best["verdict"], "confidence": best["confidence"], "results": results}


def trust_summary(items):
    counts = {v: 0 for v in VERDICTS}
    for it in items:
        counts[it["verdict"]] += 1
    verifiable = [it for it in items if it["verdict"] != "UNVERIFIABLE"]
    score = (sum(SCORE[it["verdict"]] for it in verifiable) / len(verifiable)) if verifiable else None
    return {"counts": counts, "total": len(items),
            "trust_score": None if score is None else round(100 * score),
            "coverage": round(100 * len(verifiable) / len(items)) if items else 0}


def _cache_path(text, user_sources, limit):
    import hashlib
    import json
    import os

    import llm
    h = hashlib.sha256(json.dumps([text, sorted((user_sources or {}).items()), limit, llm.GEMMA_MODEL, llm.PROVIDER]).encode()).hexdigest()[:24]
    os.makedirs(".cache", exist_ok=True)
    return os.path.join(".cache", h + ".json")


def _safe_verify(claim, user_sources):
    try:
        return verify_claim(claim, user_sources)
    except Exception as e:  # never let one claim kill the whole run
        r = {"key": "-", "entry": "", "origin": "", "evidence_level": "none", "verdict": "UNVERIFIABLE", "confidence": 0.0,
             "evidence_quote": "", "quote_verified": False, "passages": [], "reason": f"Model error: {str(e)[:300]}"}
        return {**claim, "verdict": "UNVERIFIABLE", "confidence": 0.0, "results": [r]}


def iter_verify(text, user_sources=None, limit=10, workers=5, fresh=False):
    """Yield events: cached | claims | result (one per claim as it finishes) | done."""
    import json
    import os
    import time
    path = _cache_path(text, user_sources, limit)
    if not fresh and os.path.exists(path):
        print("[verify] served from cache", flush=True)
        yield {"type": "cached", "data": {**json.load(open(path, encoding="utf-8")), "cached": True}}
        return
    t0 = time.time()
    claims, refs = extract_claims(text, limit=limit)
    print(f"[verify] extracted {len(claims)} claims ({time.time() - t0:.1f}s)", flush=True)
    yield {"type": "claims", "claims": claims, "references": refs}
    items = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_safe_verify, c, user_sources) for c in claims]
        for f in as_completed(futs):
            it = f.result()
            items.append(it)
            print(f"[verify] {len(items)}/{len(claims)} done: {it['verdict']} ({time.time() - t0:.1f}s)", flush=True)
            yield {"type": "result", "item": it}
    items.sort(key=lambda x: x["id"])
    summary = trust_summary(items)
    out = {"claims": items, "references": refs, "summary": summary, "cached": False}
    if all(r["reason"] and not r["reason"].startswith("Model error") for it in items for r in it["results"]):
        json.dump(out, open(path, "w", encoding="utf-8"))  # only cache clean runs
    print(f"[verify] finished in {time.time() - t0:.1f}s", flush=True)
    yield {"type": "done", "summary": summary}


def verify_document(text, user_sources=None, limit=12, workers=5, fresh=False):
    items, refs, summary = [], {}, None
    for ev in iter_verify(text, user_sources, limit, workers, fresh):
        if ev["type"] == "cached":
            return ev["data"]
        if ev["type"] == "claims":
            refs = ev["references"]
        elif ev["type"] == "result":
            items.append(ev["item"])
        elif ev["type"] == "done":
            summary = ev["summary"]
    items.sort(key=lambda x: x["id"])
    return {"claims": items, "references": refs, "summary": summary, "cached": False}


IMG_PROMPT = """You are a careful peer reviewer. Check whether the attached image (a screenshot, chart,
table, or figure from a cited source) supports the claim below.

CLAIM: {claim}

Verdicts: SUPPORTED (image clearly shows it, including numbers/scope), WEAK (related but only partly
backs it, or claim overstates), MISMATCHED (image is irrelevant or contradicts), UNVERIFIABLE (image
unreadable or too little visible to decide). Judge only from what is visible.

Return JSON: {{"verdict": "...", "confidence": 0.0-1.0,
"evidence_description": "what in the image is relevant, quoting any visible text/numbers",
"reason": "one short sentence"}}"""


def verify_image(claim, image_bytes, mime):
    out = chat_json(IMG_PROMPT.format(claim=claim), images=[(mime, image_bytes)], key="verdict")
    v = str(out.get("verdict", "")).upper().strip()
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"verdict": v if v in VERDICTS else "UNVERIFIABLE", "confidence": conf,
            "evidence_description": str(out.get("evidence_description", "")).strip(),
            "reason": str(out.get("reason", "")).strip()}


if __name__ == "__main__":
    import json
    import sys
    out = verify_document(open(sys.argv[1], encoding="utf-8").read())
    for it in out["claims"]:
        print(f"[{it['verdict']:<12}] {it['confidence']:.2f}  {it['claim'][:90]}")
        for r in it["results"]:
            print(f"      ref {r['key']} ({r['evidence_level']}): {r['reason']}")
    print(json.dumps(out["summary"]))
