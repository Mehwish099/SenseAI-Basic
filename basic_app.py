"""Small single-page SenseAI server for a beginner-friendly WSD experience."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from backend.main import AnalyzeRequest, analyze

app = FastAPI(title="SenseAI Basic", version="1.0.0")
PAGE = Path(__file__).with_name("basic.html")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PAGE)

@app.get("/api/health")
def health():
    from backend.main import _wordnet
    return {"status": "ok", "wordnet": "ready" if _wordnet() else "missing"}

@app.post("/api/wsd/analyze")
def analyze_word(payload: AnalyzeRequest):
    if len(payload.text) > 5000:
        raise HTTPException(413, "Keep the text under 5,000 characters.")
    return analyze(payload)
