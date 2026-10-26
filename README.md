# CiteCheck

**Does the cited source really say that?**

A bibliography entry can exist while the source still fails to support the sentence that cites it.
CiteCheck takes a research or technical document, pulls out its citation-bearing claims, retrieves
each cited source, and tells you whether the source supports the claim. Every claim gets one of four verdicts:

| Verdict | Meaning |
|---|---|
| **Supported** | The source clearly states or directly implies the claim, backed by a quote that is verified to exist in the source |
| **Weakly supported** | On topic, but narrower, hedged, indirect, or weaker than what the claim asserts |
| **Mismatched** | The source is irrelevant to the claim or contradicts it |
| **Unverifiable** | The source could not be retrieved, or the evidence available (e.g. only an abstract) is not enough |

## What makes it different

- **Evidence you can check.** The model must quote the source. CiteCheck verifies the quote exists in the retrieved text; invented quotes are discarded and the verdict is downgraded. A "Supported" always has real evidence behind it.
- **Honest about what it saw.** Each result records whether it judged the full text or only an abstract, and abstract-only evidence is capped at moderate confidence.
- **Catches overstated claims.** "Eliminates hallucinations" citing a paper that says "reduces" is flagged, not waved through.
- **Multimodal check.** Verify a claim against a screenshot, chart, or table image using Gemma 4's vision.
- **Fully open-source.** Built on the open-weight Gemma 4 model and an open embedding model. The model layer is swappable (Gemini API, local Ollama, OpenRouter) and falls back automatically if one fails.

## How it works

1. **Extract**: the model finds claims that carry citations and maps them to the reference list.
2. **Retrieve**: fetch the source (your pasted text, then arXiv, then DOI via Unpaywall / Semantic Scholar / Crossref, then the URL).
3. **Rank**: split into passages and pick the most relevant ones with open embeddings (keyword fallback if offline).
4. **Judge**: Gemma 4 returns a verdict, confidence, and an exact evidence quote; guards check it.
5. **Report**: highlighted document, per-claim evidence, a trust score, and JSON / Markdown export.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then put your GEMINI_API_KEY in .env
python app.py                                         # open http://127.0.0.1:8000
```

Get a free key at https://aistudio.google.com. To run fully local, install [Ollama](https://ollama.com),
pull a model, and set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=<model>` in `.env`.

CLI: `python verify.py samples/demo_paper.txt`

## Limits

- Paywalled sources can't be fetched; paste their text in the UI, otherwise they are marked Unverifiable.
- Verdicts are a reviewer's aid, not a final ruling. Always read the quoted evidence.
- Claim extraction covers numbered citations `[1]` and common author-year styles, up to about 14k characters of body text.

## License

MIT
