# DIX VISION — Dashboard 2026 (wave-02 React/Vite port)

Wave-02 of the Dashboard-2026 plan. Ports the operator console from
the vanilla HTML pages in `ui/static/` to a React + TypeScript +
Vite + Tailwind app served under `/dash2/` by the FastAPI harness.

The vanilla pages remain canonical until parity is proven; the React
app is additive and can be disabled by deleting `dashboard2026/dist/`.

## Quickstart

```bash
cd dashboard2026
npm install        # one-time
npm run dev        # http://localhost:5173 (proxies /api → :8000)
npm run build      # emits dist/ for the FastAPI mount
npm run lint
npm run typecheck
```

Run the FastAPI harness in another terminal:

```bash
uvicorn ui.server:app --reload --port 8000
```

Then open `http://localhost:5173/` for the dev server, or build and
visit `http://localhost:8000/dash2/` to load the production bundle
served by FastAPI.

## Codegen

TypeScript types in `src/types/generated/` are produced from Pydantic
v2 response models in `core/contracts/api/` by
`tools/codegen/pydantic_to_ts.py`. Edit the Python model, then:

```bash
python -m tools.codegen.pydantic_to_ts \
    core.contracts.api.credentials.CredentialsStatusResponse \
    core.contracts.api.operator.OperatorSummaryResponse \
    core.contracts.api.operator.OperatorActionResponse \
    core.contracts.api.cognitive_chat.ChatStatusResponse \
    core.contracts.api.cognitive_chat.ChatTurnRequest \
    core.contracts.api.cognitive_chat.ChatTurnResponse \
    --out dashboard2026/src/types/generated/api.ts
```

A pytest test reruns the generator with `--check` and fails CI on
drift, so regenerate before committing.

## Stack

* React 19 + TypeScript (strict).
* Vite 5.
* Tailwind CSS 3 with a small palette matching the operator dashboard.
* TanStack Query for server state (refetch on focus, 5s stale window).
* Hash-based router (no `react-router` dep) — `#/credentials`
  (default) and `#/operator` are the two ported pages today.

## Why a separate directory

Keeping the React app under `dashboard2026/` rather than replacing
`ui/static/` means:

* The vanilla pages stay live at their canonical URLs (`/credentials`,
  `/operator`, `/indira-chat`, `/dyon-chat`, `/forms-grid`) for
  fallback and reference.
* The React build artefact is gitignored (`dist/`), so the Python
  package stays small.
* The React surface is opt-in — operators on machines without Node
  installed can still run the console.
