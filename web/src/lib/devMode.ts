/**
 * DEV ONLY: `?mock=1` under `npm run dev` swaps the API for canned responses
 * (src/dev/). Every use is written as `import.meta.env.DEV && MOCK_MODE`, so
 * in a production build the guard is the literal `false` at each call site
 * and the dev modules are never bundled — `npm run build` output is checked
 * for their absence.
 */
export const MOCK_MODE = import.meta.env.DEV && new URLSearchParams(window.location.search).has("mock");
