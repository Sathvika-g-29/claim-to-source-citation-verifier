# HANDOFF: CiteCheck (Claim-to-Source Citation Verifier)

Paste this whole file into any AI assistant (Gemini, ChatGPT, DeepSeek, etc.) to continue the build.
Update the "STATUS" section every time you finish something.

## Context
- Event: INFINITY AI FORGE, Hacktoberfest Hack Day Amaravati (VIT-AP), 26 Oct 2026
- Hard deadline: submission closes 26 Oct, 6:00 PM IST. Then demos and judging.
- Team size: 1-4 members. Fill in: TEAM = ___
- Target prizes (stacked, one project):
  1. Problem statement: "Claim-to-Source Citation Verifier" (Intermediate)
  2. Best Use of Gemma 4 (Google DeepMind): multimodal open-weights model via Gemini API
  3. Best Open-Source AI Project (DigitalOcean): open-source/open-weight AI central to the project
- OFFICIAL PROBLEM TEXT: "A bibliography entry can exist while the cited source still fails to support the sentence that refers to it. Researchers and reviewers need to trace claims to evidence and spot citations that are irrelevant, weak, or mismatched. Given a research or technical document, inspect selected claims and their citations, retrieve the cited source where available, and assess whether the source appears to support the claim. Distinguish supported, weakly supported, mismatched, and unverifiable citations."
- REQUIRED VERDICTS (exactly these four): SUPPORTED, WEAK, MISMATCHED, UNVERIFIABLE
- Gemma 4 model ids on Gemini API: `gemma-4-26b-a4b-it` (default, faster) and `gemma-4-31b-it` (stronger)
- Gemma on Gemini API: do not rely on system_instruction or JSON mode; put instructions in the user prompt and parse JSON defensively.

## What we are building
Input: a text (essay, AI answer, report) with citations/links, plus the source documents (URLs, PDFs, pasted text, or screenshots).
Output: per-claim verdicts with evidence.

Pipeline:
1. Claim extraction: split the text into atomic, checkable claims and map each to its cited source.
2. Source ingestion: fetch URL / parse PDF / OCR-or-vision on screenshots (Gemma 4 multimodal) -> clean text chunks.
3. Evidence retrieval: embed chunks (open embedding model), pick top-k passages per claim.
4. Verification: model judges each claim vs passages -> verdict: SUPPORTED | WEAK | MISMATCHED | UNVERIFIABLE, with confidence, an exact quoted evidence span, and a one-line reason.
   - SUPPORTED: source clearly states or directly implies the claim.
   - WEAK: source is on topic but only partly backs it (narrower, hedged, indirect, or different scale).
   - MISMATCHED: source is retrievable but irrelevant to or contradicts the claim.
   - UNVERIFIABLE: source could not be retrieved or only a title/abstract was found and it is not enough to decide.
   - Always record `evidence_level`: fulltext | abstract_only | none. Abstract-only can never give SUPPORTED with high confidence.
   Source retrieval order: user-supplied source text/PDF -> arXiv id -> DOI via Crossref/Unpaywall -> Semantic Scholar API (abstract) -> plain URL via trafilatura -> else UNVERIFIABLE.
5. Report UI: original text with claims color-highlighted, click a claim to see the source passage highlighted. Overall "citation trust score". Export as JSON/Markdown.

Standout features (the "wow"):
- Highlight the exact supporting/contradicting sentence in the source.
- Detects fabricated or irrelevant citations (source exists but does not say that).
- Image input: verify a claim against a screenshot/chart (Gemma 4 vision).
- Runs fully open-source; swappable model backend.

## Architecture (keep it simple, 4-hour build)
- Backend: Python 3.11, FastAPI. Frontend: one static HTML page (vanilla JS), no build step.
- Model layer: `llm.py` with ONE function `chat(messages, images=None, json_schema=None)`.
  Providers selected by env var `LLM_PROVIDER`:
  - `gemini`: Gemma 4 via Gemini API (primary; needed for the Gemma prize)
  - `ollama`: local open-weight model (backup if API quota/network fails)
  - `openrouter` or `groq`: another hosted open model (backup)
- Retrieval: `sentence-transformers` (small open model) + cosine similarity in memory. No vector DB.
- Parsing: `trafilatura` for URLs, `pypdf` for PDFs.
- Always ask the model for strict JSON and validate with pydantic; retry once on parse failure.
- Repo: public GitHub, MIT or Apache-2.0 license, good README (needed for the open-source prize).

## File layout
```
citecheck/
  app.py            # FastAPI routes: POST /verify, GET /
  llm.py            # provider abstraction
  claims.py         # claim extraction
  sources.py        # URL/PDF/image -> text chunks
  retrieve.py       # embeddings + top-k
  verify.py         # verdict per claim
  static/index.html # UI
  samples/          # 2-3 demo inputs (one with a fake citation!)
  README.md  LICENSE  requirements.txt  .env.example
```

## Timeline (from ~2:15 PM)
- 2:15-2:45  repo + llm.py + claim extraction working in terminal
- 2:45-3:30  sources + retrieval + verification, end to end on a sample
- 3:30-4:30  FastAPI + highlighted UI
- 4:30-5:15  image input, trust score, polish, samples
- 5:15-5:40  README, license, push, record 90s demo video
- 5:40-5:55  SUBMIT (do not wait until 6:00)

## Rules for any assistant continuing this
- Do not redesign. Finish the pipeline above in the file layout above.
- Output complete files, not snippets. Keep each file short.
- Prefer working and demoable over clever.
- Never hardcode API keys; use `.env`.

## STATUS (update me)
- [ ] repo created and pushed (do this first: git init, commit, push)
- [x] llm.py (gemini + ollama fallback)
- [x] claims.py
- [x] sources.py (live fetch UNTESTED: sandbox blocked arxiv.org; test on your machine)
- [x] retrieve.py
- [x] verify.py (logic tested with stub model; test with real Gemma key)
- [x] app.py + UI (tested with stub model; run with real key)
- [x] image input (verify_image + Image tab; untested with real model)
- [x] README + license (add team names)
- [x] speed work: rule-based claim extraction, parallel checks, streaming UI, result cache
- [ ] demo video
- [ ] submitted on MLH page
