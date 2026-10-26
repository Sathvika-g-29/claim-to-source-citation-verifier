# CiteCheck: submission kit (copy and paste from here)

## Title
CiteCheck: does the cited source really say that?

## One-line tagline
Paste a paper, and CiteCheck fetches every cited source and tells you whether each citation truly supports its claim, with the evidence quoted.

## Description (paste into the MLH submission form)

**The problem.** A bibliography entry can exist while the cited source still fails to support the sentence that cites it. Reviewers and researchers rarely have time to open every reference, so weak, mismatched, and even fabricated citations slip through.

**What CiteCheck does.** You paste a research or technical document with its reference list. CiteCheck finds the claims that carry citations, retrieves each cited source (arXiv full text, DOI, or URL, or text you paste), and judges every claim with one of four verdicts: Supported, Weakly supported, Mismatched, or Unverifiable. Each verdict comes with a confidence level, a one-line reason, and the exact quote from the source.

**What makes it trustworthy.**
- *Quotes are verified.* The model must quote the source, and CiteCheck checks that the quote really exists in the retrieved text. Invented quotes are thrown away and the verdict is downgraded, so "Supported" always has real evidence behind it.
- *It is honest about what it saw.* Each result says whether it judged the full text or only an abstract, and abstract-only evidence is capped at moderate confidence.
- *It catches overstated claims,* such as "eliminates hallucinations" citing a paper that only says "reduces".
- *Multimodal check.* Using Gemma 4's vision, it also verifies a claim against a screenshot, chart, or table image.

**How it is built.** Python and FastAPI backend, a single-page web UI with live streaming results, Gemma 4 (open weights) via the Gemini API for judging and image understanding, open embedding models (sentence-transformers) for finding the relevant passages, and pypdf and trafilatura for source retrieval. The model layer is swappable (Gemini API, local Ollama, OpenRouter) with automatic fallback between Gemma 4 models.

**Challenges.** Getting the model to quote real evidence instead of plausible-sounding text, handling overloaded model servers during the hackathon, and making a multi-source pipeline feel fast (claims are checked in parallel and stream in live).

**What's next.** Click-through evidence highlighting inside the source PDF, suggested rewordings for weak claims, and support for author-year citation styles at scale.

## Prizes to select
1. Claim-to-Source Citation Verifier (problem statement)
2. Best Use of Gemma 4
3. Best Open-Source AI Project

## Built with
Gemma 4, Gemini API, Python, FastAPI, sentence-transformers, pypdf, trafilatura, JavaScript

## Demo script (about 90 seconds, screen recording)
1. (0:00) "Citations can exist while the source says something else. CiteCheck checks every citation." Show the app.
2. (0:10) Click **Load demo (4 verdicts)** and press **Check citations**. Say that claims are checked in parallel and appear live.
3. (0:25) Results arrive. Walk through them:
   - Green, Supported: the BLEU score claim, with the quote found in the source.
   - Yellow, Weakly supported: "no architecture changes for any task", which overstates the paper.
   - Red, Mismatched: the GPT-3 claim cites the wrong paper, which says nothing about it.
   - Grey, Unverifiable: a reference that does not exist, so no source could be retrieved.
4. (1:00) Click one claim to show the evidence quote and the "found in source" check. Say that invented quotes are rejected.
5. (1:15) Open **Image check**, upload the test chart, and check a claim. Mention Gemma 4 vision.
6. (1:30) "Open-source, built on Gemma 4. Thank you."

Tip: run the demo document once before recording, so the second run is served instantly from the cache.

## GitHub steps (Windows terminal, inside the project folder)
1. Create an empty PUBLIC repo on github.com (no README), named `citecheck`. Add the topic `hacktoberfest`.
2. In the terminal:
```
git init
git add .
git status
```
   Check that `.env` is NOT in the list. If it is, stop and tell Claude.
```
git commit -m "CiteCheck: claim-to-source citation verifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/citecheck.git
git push -u origin main
```
3. Add your team's names to the README before pushing.
