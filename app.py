"""FastAPI server: serves the UI and the verification API."""
import json
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llm  # noqa: E402
import retrieve  # noqa: E402
from verify import iter_verify, verify_document, verify_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="CiteCheck")
retrieve.warmup()


class VerifyReq(BaseModel):
    text: str
    sources: dict[str, str] = {}   # optional: reference number -> pasted source text
    limit: int = 10
    fresh: bool = False


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/info")
def info():
    return {"provider": llm.PROVIDER, "model": llm.GEMMA_MODEL if llm.PROVIDER == "gemini" else
            (llm.OLLAMA_MODEL if llm.PROVIDER == "ollama" else llm.OPENROUTER_MODEL),
            "has_key": bool(llm.GEMINI_KEY) or llm.PROVIDER != "gemini"}


@app.get("/api/sample")
def sample(n: int = 1):
    name = "demo_paper_2.txt" if n == 2 else "demo_paper.txt"
    return {"text": open(os.path.join(HERE, "samples", name), encoding="utf-8").read()}


@app.post("/api/verify")
def verify(req: VerifyReq):
    if len(req.text.strip()) < 40:
        raise HTTPException(400, "Paste a document with citations and a reference list.")
    try:
        return verify_document(req.text, user_sources=req.sources or None, limit=max(1, min(req.limit, 20)), fresh=req.fresh)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/verify_stream")
def verify_stream(req: VerifyReq):
    if len(req.text.strip()) < 40:
        raise HTTPException(400, "Paste a document with citations and a reference list.")

    def gen():
        try:
            for ev in iter_verify(req.text, req.sources or None, max(1, min(req.limit, 20)), fresh=req.fresh):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/image")
async def image(claim: str = Form(...), file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Upload an image (PNG/JPG).")
    data = await file.read()
    if len(data) > 8_000_000:
        raise HTTPException(400, "Image too large (max 8 MB).")
    try:
        return verify_image(claim, data, file.content_type)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
