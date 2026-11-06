# Put SenseAI on the web with Render

GitHub stores the source code. SenseAI needs two running services: a static host for the Vite frontend and a Python web service for FastAPI.

## 1. Push the production CORS setting to GitHub

The API now reads `FRONTEND_ORIGIN` so it can accept the deployed frontend URL. Commit and push the current project files, including `backend/main.py`, to the GitHub repository before creating the Render services.

## 2. Create the backend web service

In Render, choose **New → Web Service**, connect `Mehwish099/SenseAI`, and set:

| Setting | Value |
|---|---|
| Root Directory | leave blank |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python -m nltk.downloader wordnet omw-1.4` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/health` |

Set `PYTHON_VERSION` to `3.11.11`. Deploy and copy the service URL, for example `https://senseai-api.onrender.com`.

For a durable SQLite account database, choose a paid web-service plan and attach a persistent disk mounted at `/var/data`; set `DATABASE_URL` to `sqlite:////var/data/senseai.db`. Render's free web services have an ephemeral filesystem, so account and history records can be lost on restarts or redeploys. The free backend plan has 512 MB RAM; PyTorch and the contextual transformer may exceed that, so use a larger memory plan if the service crashes while loading the model. Free web services also spin down when idle.

## 3. Create the frontend static site

Choose **New → Static Site** and connect the same repository. Set:

| Setting | Value |
|---|---|
| Root Directory | leave blank |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |

Add the build environment variable `VITE_API_URL` with the backend URL from step 2, including `https://` and without a trailing slash. Deploy and copy the frontend URL, for example `https://senseai-web.onrender.com`.

## 4. Allow the deployed frontend to call the API

In the backend service's **Environment** settings, add `FRONTEND_ORIGIN` with the exact frontend origin from step 3 (no trailing slash). Save and redeploy the backend.

Open the frontend URL, create an account, and analyze a word. If login says “Failed to fetch,” confirm the backend's `/api/health` URL opens and that `FRONTEND_ORIGIN` exactly matches the frontend URL.

## Notes

- GitHub Pages alone can host the built frontend but cannot run this FastAPI backend. Keep the API hosted separately and set `VITE_API_URL` to it.
- The first contextual analysis downloads `sentence-transformers/all-MiniLM-L6-v2`. Allow network access during that request. The first model load may take time and needs more RAM than the 512 MB free backend plan reliably provides.
- Render's current free web-service filesystem is ephemeral, and free services spin down after 15 minutes without traffic. Use persistent storage for accounts you need to keep.
