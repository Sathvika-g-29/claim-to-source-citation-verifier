"""Step 2: retrieve the cited source and turn it into text chunks.

Retrieval order: user-supplied text -> arXiv (full PDF, then abstract) -> DOI (Unpaywall PDF,
then Semantic Scholar/Crossref abstract) -> plain URL (trafilatura / PDF) -> none.
evidence_level is one of: fulltext | abstract_only | none.
"""
import io
import os
import re

import threading
import time

import requests

_CACHE = {}
_LOCKS = {}
_GUARD = threading.Lock()

UA = {"User-Agent": "CiteCheck/0.1 (hackathon project; mailto:%s)" % os.getenv("CONTACT_EMAIL", "citecheck@example.com")}
TIMEOUT = 25
MAX_CHARS = 120_000


def _pdf_text(raw):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages[:30])


def _get(url, **kw):
    r = requests.get(url, headers=UA, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def _arxiv_abs(arxiv_id):
    """Return (abstract, title) from the arXiv API, or (None, None)."""
    try:
        r = _get(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
        m = re.search(r"<summary>(.*?)</summary>", r.text, re.S)
        t = re.search(r"<title>(.*?)</title>", r.text.split("<entry>", 1)[-1], re.S)
        if m:
            return " ".join(m.group(1).split()), (" ".join(t.group(1).split()) if t else "")
    except Exception:
        pass
    return None, None


def _arxiv(arxiv_id):
    abstract, title = _arxiv_abs(arxiv_id)
    try:
        r = _get(f"https://arxiv.org/pdf/{arxiv_id}")
        text = _pdf_text(r.content)
        if len(text) > 2000:
            if abstract:  # always keep the abstract searchable: it holds the paper's headline claims
                text = f"ABSTRACT: {abstract}\n\n{text}"
            return {"text": text, "evidence_level": "fulltext", "origin": f"arXiv:{arxiv_id} (PDF)", "title": title or ""}
    except Exception as e:
        print(f"[source] arXiv PDF failed for {arxiv_id}: {str(e)[:120]}", flush=True)
    if abstract:
        return {"text": abstract, "evidence_level": "abstract_only", "origin": f"arXiv:{arxiv_id} (abstract)", "title": title or ""}
    return None


def _doi(doi):
    try:  # open-access full text via Unpaywall
        r = _get(f"https://api.unpaywall.org/v2/{doi}?email={os.getenv('CONTACT_EMAIL', 'citecheck@example.com')}")
        loc = (r.json().get("best_oa_location") or {})
        pdf = loc.get("url_for_pdf")
        if pdf:
            text = _pdf_text(_get(pdf).content)
            if len(text) > 2000:
                return {"text": text, "evidence_level": "fulltext", "origin": f"DOI {doi} (open-access PDF)"}
    except Exception:
        pass
    try:  # abstract via Semantic Scholar
        r = _get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract")
        j = r.json()
        if j.get("abstract"):
            return {"text": j["abstract"], "evidence_level": "abstract_only",
                    "origin": f"DOI {doi} (abstract)", "title": j.get("title", "")}
    except Exception:
        pass
    try:  # abstract via Crossref
        r = _get(f"https://api.crossref.org/works/{doi}")
        ab = r.json()["message"].get("abstract")
        if ab:
            return {"text": re.sub(r"<[^>]+>", " ", ab), "evidence_level": "abstract_only",
                    "origin": f"DOI {doi} (Crossref abstract)"}
    except Exception:
        pass
    return None


def _url(url):
    try:
        r = _get(url)
        if "pdf" in r.headers.get("content-type", "").lower() or url.lower().endswith(".pdf"):
            text = _pdf_text(r.content)
        else:
            import trafilatura
            text = trafilatura.extract(r.text) or ""
        if len(text) > 500:
            return {"text": text, "evidence_level": "fulltext", "origin": url}
    except Exception:
        pass
    return None


def get_source(citation, user_sources=None):
    """citation: dict from claims.py (key, entry, doi, arxiv, url). user_sources: {key: text}."""
    key = citation.get("key")
    if user_sources and user_sources.get(key):
        return {"text": user_sources[key][:MAX_CHARS], "evidence_level": "fulltext", "origin": "user-supplied"}
    ck = (citation.get("arxiv"), citation.get("doi"), citation.get("url"))
    with _GUARD:
        lock = _LOCKS.setdefault(ck, threading.Lock())
    with lock:
        if ck in _CACHE:
            return _CACHE[ck]
        t0, result = time.time(), None
        for fn, arg in ((_arxiv, citation.get("arxiv")), (_doi, citation.get("doi")), (_url, citation.get("url"))):
            if arg:
                res = fn(arg)
                if res:
                    res["text"] = res["text"][:MAX_CHARS]
                    result = res
                    break
        result = result or {"text": "", "evidence_level": "none", "origin": "could not retrieve"}
        print(f"[source] ref {key}: {result['evidence_level']} via {result['origin']} ({time.time() - t0:.1f}s)", flush=True)
        _CACHE[ck] = result
        return result


def chunk_text(text, size=700, overlap=1):
    """Sentence-aware chunks of ~size chars; consecutive chunks share `overlap` sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", text)
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x) for x in cur) >= size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks
