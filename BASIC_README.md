# SenseAI Basic

A simple, single-page WSD website. It has one sentence box, one target-word field, and one result panel. There is no account or frontend build step. The page calls the same WordNet/contextual-transformer analysis pipeline as the full app.

## Run it locally on Windows

From the project folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader wordnet omw-1.4
.\.venv\Scripts\python.exe -m uvicorn basic_app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Enter your own sentence and target word. The first analysis downloads the sentence-transformer model; allow a little time and internet access for that download.

## Publish it with Render

Connect this GitHub repository as one **Web Service** in Render:

- Build command: `pip install -r requirements.txt && python -m nltk.downloader wordnet omw-1.4`
- Start command: `uvicorn basic_app:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- Python version: `3.11.11`

When the service deploys, open its `onrender.com` URL. The page and API are served from the same address, so no frontend API URL or CORS configuration is needed.

Render free web services have 512 MB RAM, which may not be enough for PyTorch and the transformer. If model loading crashes, use a service with more memory. The service uses the existing SQLite database and does not offer login or history in this simple mode; without persistent storage, saved local database data can be lost when the host restarts or redeploys.
