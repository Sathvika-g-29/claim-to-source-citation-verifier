"""Step 1: pull citation-bearing claims out of a document and resolve them to bibliography entries."""
import re

from llm import chat_json

# Split the document body from its reference list.
_REF_HEAD = re.compile(r"^\s*(#+\s*)?(references|bibliography|works cited|sources)\s*:?\s*$", re.I | re.M)


def split_document(text):
    m = None
    for m in _REF_HEAD.finditer(text):
        pass  # keep the LAST matching heading
    if not m:
        return text, ""
    return text[:m.start()], text[m.end():]


def parse_bibliography(ref_text):
    """Return {key: entry_text}. Handles '[1] ...', '1. ...', and one-entry-per-paragraph styles."""
    refs = {}
    numbered = re.split(r"\n\s*(?=(?:\[\d+\]|\d+\.)\s)", "\n" + ref_text.strip())
    for chunk in numbered:
        chunk = chunk.strip()
        m = re.match(r"^\[?(\d+)[\].]\s*(.+)$", chunk, re.S)
        if m:
            refs[m.group(1)] = " ".join(m.group(2).split())
    if not refs:  # author-year style: one entry per non-empty line/paragraph
        for i, para in enumerate([p.strip() for p in re.split(r"\n\s*\n|\n", ref_text) if p.strip()], 1):
            refs[str(i)] = " ".join(para.split())
    return refs


def extract_ids(entry):
    """Pull DOI / arXiv id / URL out of a bibliography entry."""
    doi = re.search(r"10\.\d{4,9}/[^\s\"<>]+", entry)
    arxiv = re.search(r"arxiv[:.\s/]*(\d{4}\.\d{4,5})(v\d+)?", entry, re.I)
    url = re.search(r"https?://[^\s\"<>)]+", entry)
    return {
        "doi": doi.group(0).rstrip(".,;)") if doi else None,
        "arxiv": arxiv.group(1) if arxiv else None,
        "url": url.group(0).rstrip(".,;)") if url else None,
    }


PROMPT = """You are helping audit citations in a research or technical document.

Below is the document body and its numbered reference list. Find sentences that make a
factual, quantitative, or methodological claim AND cite a source. For each, return:
- "claim": the exact sentence text, copied verbatim from the document
- "claim_atomic": the single checkable assertion, rewritten as one standalone statement
- "citation_keys": list of reference numbers it cites, as strings (e.g. ["3"]). For author-year
  citations, map them to the matching reference number from the list.

Skip sentences with no citation. Skip pure opinion or transitions. Return at most {limit} claims,
prioritising the most specific and checkable ones (numbers, comparisons, named results).

Return JSON: {{"claims": [{{"claim": "...", "claim_atomic": "...", "citation_keys": ["1"]}}]}}

REFERENCES:
{refs}

DOCUMENT:
{body}
"""


def _extract_claims_llm(text, limit=12):
    body, ref_text = split_document(text)
    refs = parse_bibliography(ref_text)
    ref_block = "\n".join(f"[{k}] {v}" for k, v in refs.items()) or "(none found)"
    data = chat_json(PROMPT.format(limit=limit, refs=ref_block, body=body[:14000]), key="claims")
    out = []
    for i, c in enumerate(data.get("claims", [])[:limit]):
        keys = [str(k) for k in c.get("citation_keys", [])]
        out.append({
            "id": i,
            "claim": c.get("claim", "").strip(),
            "claim_atomic": c.get("claim_atomic", "").strip(),
            "citations": [{"key": k, "entry": refs.get(k, ""), **extract_ids(refs.get(k, ""))} for k in keys],
        })
    return out, refs


if __name__ == "__main__":
    import json
    import sys

    claims, _ = extract_claims(open(sys.argv[1], encoding="utf-8").read())
    print(json.dumps(claims, indent=2))


_MARK = re.compile(r"\[(\d+(?:\s*[,;\u2013-]\s*\d+)*)\]")
_HOT = re.compile(r"\b(improv\w*|reduc\w*|outperform\w*|achiev\w*|shown?|state-of-the-art|eliminat\w*|always|every|all|never|"
                  r"linear|quadratic\w*|faster|better|accuracy|significant\w*|substantial\w*|prove\w*|demonstrat\w*)\b", re.I)


def _marker_keys(sentence):
    keys = []
    for m in _MARK.finditer(sentence):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            if re.search(r"[\u2013-]", part):
                a, b = re.split(r"\s*[\u2013-]\s*", part)
                keys += [str(n) for n in range(int(a), min(int(b), int(a) + 20) + 1)]
            else:
                keys.append(part.strip())
    return list(dict.fromkeys(keys))


def _extract_claims_rules(body, refs, limit):
    """Fast, no-model extraction for numbered citations like [1], [2,3], [4-6]."""
    prot = body
    for a, b in (("et al.", "et al\u00a7"), ("e.g.", "e\u00a7g\u00a7"), ("i.e.", "i\u00a7e\u00a7"),
                 ("Fig.", "Fig\u00a7"), ("vs.", "vs\u00a7"), ("approx.", "approx\u00a7")):
        prot = prot.replace(a, b)
    cands, pos = [], 0
    for para in re.split(r"\n\s*\n", prot):
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para):
            s = " ".join(s.replace("\u00a7", ".").split())
            keys = [k for k in _marker_keys(s) if k in refs]
            if keys and len(s) >= 40:
                score = 2 * len(re.findall(r"\d", s) and [1]) + len(_HOT.findall(s))
                cands.append((pos, score, s, keys))
                pos += 1
    best = sorted(cands, key=lambda c: -c[1])[:limit]
    out = []
    for i, (_, _, s, keys) in enumerate(sorted(best)):
        atomic = " ".join(_MARK.sub("", s).replace(" .", ".").split())
        out.append({"id": i, "claim": s, "claim_atomic": atomic,
                    "citations": [{"key": k, "entry": refs[k], **extract_ids(refs[k])} for k in keys]})
    return out


def extract_claims(text, limit=12):
    body, ref_text = split_document(text)
    refs = parse_bibliography(ref_text)
    out = _extract_claims_rules(body, refs, limit)
    if out:
        return out, refs
    return _extract_claims_llm(text, limit)  # author-year style: let the model do it
