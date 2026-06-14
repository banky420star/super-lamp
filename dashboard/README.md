Phase 2 SPA skeleton
- This folder contains a minimal React/TypeScript SPA scaffold (Vite) that will be wired to the Python backend REST API endpoints.
- The MVP wiring from Phase 1 (patterns, LR state) is already in place and exposed via /api/patterns and /api/perf and augmented /api/status.
- This scaffold is intentionally lightweight to allow fast iteration and a real SPA later.

How to run locally (when dependencies are installed):
- cd frontend
- npm install
- npm run dev
- Open http://localhost:5173/

APIs to consume:
- GET /api/status
- GET /api/patterns
- GET /api/perf
