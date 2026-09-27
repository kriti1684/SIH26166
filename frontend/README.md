# ChandaShakti frontend

This is the local React, TypeScript, and Vite workspace for the registration API.
It uploads a source product, a reference product, and optional label/payload
sidecars; follows the persisted job and stage events; and provides previews,
metrics, and downloads from the backend artifact manifest.

## Run locally

Install the Python dependencies using the repository root's README section
**12.1 Environment Setup** first, then activate that Python environment in the
API terminal.

Run the API from the repository root:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API starts each registration pipeline in-process after creating the job.

In another terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`). Vite
proxies `/api/*` and `/health` to `http://127.0.0.1:8000`. Set
`VITE_BACKEND_URL` in a local `.env` file if the API is listening elsewhere.
`VITE_API_BASE_URL` can be set when the browser should call an API directly
instead of going through the Vite proxy.

The run history and active run id are restored from the backend and browser
local storage. Uploaded products and outputs are retained by the backend's
local storage configuration. Only generated PNG/JPEG quicklooks are embedded;
GeoTIFFs, CSVs, labels, metadata, and other outputs are served as downloads.
