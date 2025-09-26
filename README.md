Deployed on Hugging Face at https://huggingface.co/spaces/Aditya-Darshan-G/Virtual_TA

---

# Virtual\_TA — Gemini-powered Semantic Search API

A lightweight FastAPI service that performs **semantic search over your documents using Google Gemini embeddings**, with a simple REST API and container-first deployment.

> **Endpoints (current app)**
>
> * `POST /query` — Ask a question; returns top semantic matches (Gemini + vector search).
> * `GET /health` — Liveness/readiness check. ([GitHub][1])

---

## Table of Contents

* [Why this project](#why-this-project)
* [Features](#features)
* [Architecture](#architecture)
* [Quickstart](#quickstart)

  * [Run locally (Python)](#run-locally-python)
  * [Run with Docker](#run-with-docker)
  * [Deploy on Hugging Face Spaces](#deploy-on-hugging-face-spaces)
* [Configuration](#configuration)
* [API Reference](#api-reference)
* [Project Structure](#project-structure)
* [Development](#development)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Why this project

Virtual\_TA gives you a **minimal, reliable backend** for question-answering over your own corpus. It exposes a tiny HTTP surface, is trivial to deploy as a Docker image, and is compatible with HF Spaces. The repository already includes a `Dockerfile` and a concise FastAPI app entry (`app/main.py`). ([GitHub][1])

---

## Features

* 🔎 **Semantic search API** — Query your indexed documents through `POST /query`. ([GitHub][1])
* ⚙️ **FastAPI** server with health check via `GET /health`. ([GitHub][1])
* 📦 **Container-first** — Ready-to-build `Dockerfile`. ([GitHub][1])
* 🚀 **HF Spaces friendly** — App entrypoint at `app/main.py` and a minimal README already tailored for Spaces. ([GitHub][1])

---

## Architecture

```
Client ──HTTP──> FastAPI (app/main.py)
                     ├─ Loads embeddings model (Gemini)
                     ├─ Builds/loads vector index over your docs
                     └─ Returns top-K matches & metadata
```

**Key components in this repo**

* `app/main.py` — FastAPI app with `/query` and `/health`.
* `requirements.txt` — Python dependencies to run the service.
* `Dockerfile` — Container build.
* `data/` — Place your source files here (txt/markdown, etc.) for indexing.
* `app/…` — Embedding + search utilities (load model, embed, search).
  (See the tree below.) ([GitHub][1])

> Note: The repo’s language breakdown is mostly Python with a small HTML portion and the Dockerfile, matching a typical FastAPI microservice. ([GitHub][1])

---

## Quickstart

### Prerequisites

* Python 3.10+
* A Google Gemini API key (set `GEMINI_API_KEY`)

### Run locally (Python)

```bash
# 1) Clone
git clone https://github.com/Aditya-Darshan-G/Virtual_TA
cd Virtual_TA

# 2) Create & activate a venv (Windows PowerShell shown)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Set your Gemini API key
$env:GEMINI_API_KEY="YOUR_KEY_HERE"      # PowerShell
# export GEMINI_API_KEY="YOUR_KEY_HERE"  # macOS/Linux

# 5) Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

### Run with Docker

```bash
# Build
docker build -t virtual-ta .

# Run (pass your API key)
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=YOUR_KEY virtual-ta
```

The service will be available at `http://localhost:8000`.

### Deploy on Hugging Face Spaces

1. Create a new **Space** (SDK: **Docker**).
2. Push this repo (or mirror it) to the Space.
3. Ensure `Dockerfile` is present (it is).
4. Add your `GEMINI_API_KEY` as a **secret** in the Space.
5. Deploy — the container will expose the FastAPI app from `app/main.py`. ([GitHub][1])

---

## Configuration

Set via environment variables (recommended):

| Variable          | Required | Description                                   |
| ----------------- | -------- | --------------------------------------------- |
| `GEMINI_API_KEY`  | Yes      | Google Gemini key for embeddings/Q\&A.        |
| `EMBEDDING_MODEL` | No       | Override the default Gemini embedding model.  |
| `TOP_K`           | No       | Number of results to return (default: 5).     |
| `DATA_DIR`        | No       | Path to documents to index (default: `data/`) |

> Place your documents (e.g., `.txt`, `.md`, etc.) into `data/`. On startup the app can load or build an index so that `POST /query` can retrieve semantically similar chunks.

---

## API Reference

### `GET /health`

Simple liveness/readiness probe. Returns JSON like:

```json
{"status": "ok"}
```

(Useful for Docker, Spaces, or k8s probes.) ([GitHub][1])

### `POST /query`

Submit a semantic query.

**Request**

```json
{
  "query": "What are the key steps to run the agent locally?",
  "top_k": 5
}
```

**Response (example)**

```json
{
  "query": "What are the key steps to run the agent locally?",
  "results": [
    {
      "text": "Create a virtual environment and install -r requirements.txt ...",
      "score": 0.83,
      "source": "data/getting_started.md"
    }
  ]
}
```

* `query` (string) — your natural-language question.
* `top_k` (int) — optional; number of results to return.
* `results` — ranked passages with a similarity score and source.

> The exact shape of responses depends on the code in `app/main.py` and helpers (embedding/indexing). Endpoint names are current and stable in this repo. ([GitHub][1])

---

## Project Structure

```text
Virtual_TA/
├─ app/
│  ├─ main.py                # FastAPI app (endpoints: /query, /health)
│  ├─ ...                    # Embedding + semantic search utilities
├─ data/                     # Put your documents here
├─ requirements.txt          # Python deps
├─ Dockerfile                # Container build
└─ README.md                 # This file
```

(Structure based on the repository listing and files present.) ([GitHub][1])

---

## Development

**Formatting & linting**

```bash
pip install ruff black
ruff check app
black app
```

**Run tests (if/when you add them)**

```bash
pytest -q
```

**Hot reload**

```bash
uvicorn app.main:app --reload
```

---

## Troubleshooting

* **`401`/auth errors:** Ensure `GEMINI_API_KEY` is set in your shell or Docker environment.
* **Empty results:** Confirm documents exist in `data/` and that the index builds on startup.
* **Port already in use:** Change `--port` in the `uvicorn` command or stop the conflicting process.
* **HF Spaces build fails:** Double-check that the Space is set to **Docker** and secrets are configured.

---

## License

MIT License
