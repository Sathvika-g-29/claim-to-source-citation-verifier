"""Provider-agnostic LLM layer. One function: chat(). Swap backends with LLM_PROVIDER.

LLM_PROVIDER = gemini (Gemma 4 via Gemini API, default) | ollama | openrouter
"""
import base64
import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")  # or gemma-4-31b-it
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")  # any local model you have pulled
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")

# If the primary provider fails (quota, network), try these in order.
FALLBACKS = [p.strip() for p in os.getenv("LLM_FALLBACKS", "").split(",") if p.strip()]
# If the main Gemma model is overloaded (503) or erroring, try these other Gemma 4 models next.
GEMMA_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMMA_FALLBACK_MODELS", "gemma-4-31b-it,gemma-4-26b-a4b-it").split(",") if m.strip()]


def _gemini(prompt, images, model=None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or GEMMA_MODEL}:generateContent"
    parts = [{"text": prompt}]
    for img in images or []:  # img = (mime, raw_bytes)
        parts.append({"inline_data": {"mime_type": img[0], "data": base64.b64encode(img[1]).decode()}})
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}}
    r = requests.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json=body, timeout=45)
    if r.status_code >= 400:
        raise requests.HTTPError(f"Gemini API {r.status_code}: {r.text[:300]}", response=r)
    j = r.json()
    cands = j.get("candidates") or []
    if not cands or not cands[0].get("content", {}).get("parts"):
        raise RuntimeError(f"Gemini returned no content: {str(j)[:300]}")
    parts = [p for p in cands[0]["content"]["parts"] if not p.get("thought")]  # drop thinking parts
    return "".join(p.get("text", "") for p in parts)


def _ollama(prompt, images, model=None):
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = [base64.b64encode(i[1]).decode() for i in images]
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=180,
                      json={"model": OLLAMA_MODEL, "messages": [msg], "stream": False,
                            "options": {"temperature": 0.1}})
    r.raise_for_status()
    return r.json()["message"]["content"]


def _openrouter(prompt, images, model=None):
    content = [{"type": "text", "text": prompt}]
    for img in images or []:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{img[0]};base64,{base64.b64encode(img[1]).decode()}"}})
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=90,
                      headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                      json={"model": OPENROUTER_MODEL, "temperature": 0.1,
                            "messages": [{"role": "user", "content": content}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_IMPL = {"gemini": _gemini, "ollama": _ollama, "openrouter": _openrouter}


def _plan():
    """Ordered list of (provider, model) to try."""
    plan = []
    if PROVIDER == "gemini":
        plan += [("gemini", m) for m in dict.fromkeys([GEMMA_MODEL] + GEMMA_FALLBACK_MODELS)]
    else:
        plan.append((PROVIDER, None))
    plan += [(f, None) for f in FALLBACKS if f != PROVIDER and f in _IMPL]
    return plan


def chat(prompt, images=None, retries=1):
    """Return raw text. Retries transient errors with backoff, then moves to the next model/provider."""
    errors = []
    for prov, model in _plan():
        for attempt in range(retries + 1):
            try:
                return _IMPL[prov](prompt, images, model)
            except Exception as e:  # quota, overload, timeout, bad key, provider down
                label = f"{prov}:{model}" if model else prov
                msg = f"{label}: {str(e)[:300]}"
                print(f"[llm] attempt {attempt + 1} failed -> {msg}", flush=True)
                if msg not in errors:
                    errors.append(msg)
                code = getattr(getattr(e, "response", None), "status_code", 0) or 0
                if prov == "ollama" and "refused" in str(e):
                    break  # not installed/running: do not keep retrying
                if 400 <= code < 500 and code != 429:
                    break  # bad key / bad model name: retrying will not help
                if attempt < retries:
                    time.sleep(3 * (attempt + 1))  # 3s, 6s backoff for 429/5xx/timeouts
        print(f"[llm] giving up on {prov}{':' + model if model else ''}, trying next option", flush=True)
    raise RuntimeError("All LLM providers failed. " + " || ".join(errors))


def chat_json(prompt, images=None, key=None):
    """Ask for JSON and parse defensively. `key`: a field the wanted object must contain."""
    text = chat(prompt + "\n\nReturn ONLY valid JSON. No markdown fences, no commentary.", images)
    for attempt in range(2):
        try:
            return _extract_json(text, key)
        except ValueError:
            if attempt == 1:
                break
            text = chat("Rewrite the following as ONLY one valid JSON object, nothing else:\n\n" + text[-6000:])
    _log_bad(text)
    raise ValueError("Model did not return valid JSON. Start of its reply: " + text[:300].replace("\n", " "))


def _log_bad(text):
    try:
        with open("llm_debug.log", "a", encoding="utf-8") as f:
            f.write("---- bad reply ----\n" + text + "\n")
    except OSError:
        pass


def _extract_json(text, key=None):
    """Find top-level JSON objects anywhere in `text`; return the last one (that has `key`, if given)."""
    text = re.sub(r"```(?:json)?", "", text)
    dec, i, found = json.JSONDecoder(), 0, []
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                found.append(obj)
            i += end
        except json.JSONDecodeError:
            i += 1
    if key:
        found = [o for o in found if key in o]
    if not found:
        raise ValueError("no JSON found")
    return found[-1]
