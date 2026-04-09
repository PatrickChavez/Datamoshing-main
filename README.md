# Datamoshing

## Deploy To Render (Web Service)

This repository is now set up for Render using a `Dockerfile` and `render.yaml`.

### One-time setup

1. Push this repo to GitHub.
2. In Render, create a new Blueprint service and select this repository.
3. Render will detect `render.yaml` and create a web service automatically.
4. In service environment variables, set:
   - `OPENAI_API_KEY` = your API key

### Runtime behavior

- App starts via `gunicorn app:app` and binds to Render's `PORT`.
- `ffmpeg` is installed in the container image, so video processing works in production.
