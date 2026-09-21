import { useEffect, useMemo, useState } from "react";
import { realBatchApi, type BatchApi } from "./api/batch";
import { getHealth, getRuleset, initialToken, rememberToken } from "./api/client";
import type { HealthResponse, RulesetMetadata } from "./api/types";
import { BatchMode } from "./components/batch/BatchMode";
import { Icon } from "./components/Icon";
import { SingleLabel } from "./components/SingleLabel";
import { MOCK_MODE } from "./lib/devMode";
import { getParam } from "./lib/url";

type Mode = "one" | "many";

/**
 * The shell: title, the One label | Many labels switch, and the server-health
 * banner. Still one screen — the switch is the only navigation there is.
 *
 * Both modes stay mounted and the inactive one is hidden, so switching to
 * check a single label mid-batch neither loses the chosen files nor stops
 * the batch's progress polling.
 */
export function App() {
  const [ruleset, setRuleset] = useState<RulesetMetadata | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [token, setTokenState] = useState(() => initialToken());
  const setToken = (next: string) => {
    setTokenState(next);
    rememberToken(next);
  };
  // A batch id in the URL means a reload mid-batch: go straight back to it.
  const [mode, setMode] = useState<Mode>(() => (getParam("batch") ? "many" : "one"));
  const [batchApi, setBatchApi] = useState<BatchApi>(realBatchApi);
  const [devSample, setDevSample] = useState<(() => Promise<File[]>) | undefined>(undefined);
  // In mock mode, wait for the mock before mounting batch mode, so a reload
  // with ?batch= never polls the real server first.
  const [batchReady, setBatchReady] = useState(!(import.meta.env.DEV && MOCK_MODE));

  useEffect(() => {
    getRuleset().then(setRuleset, () => setRuleset(null));
    getHealth().then(setHealth, () => setHealth(null));
    if (import.meta.env.DEV && MOCK_MODE) {
      import("./dev/batchMock").then((dev) => {
        setBatchApi(dev.mockBatchApi);
        setDevSample(() => dev.sampleBatchFiles);
        setBatchReady(true);
      });
    }
  }, []);

  const implementedById = useMemo(() => {
    const map: Record<string, boolean> = {};
    for (const f of ruleset?.fields ?? []) map[f.id] = f.implemented;
    return map;
  }, [ruleset]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Label Verification</h1>
        <p>
          Compare labels with their applications. <strong>The tool recommends — you decide.</strong>
        </p>
        {import.meta.env.DEV && MOCK_MODE && (
          <p className="dev-flag">Development mock mode — results are canned, not real.</p>
        )}
        <div className="mode-switch" role="group" aria-label="How many labels">
          <button type="button" aria-pressed={mode === "one"} onClick={() => setMode("one")}>
            {mode === "one" && <Icon name="check" />}
            One label
          </button>
          <button type="button" aria-pressed={mode === "many"} onClick={() => setMode("many")}>
            {mode === "many" && <Icon name="check" />}
            Many labels
          </button>
        </div>
      </header>

      {health && health.status !== "ok" && (
        <div className="notice notice-review banner" role="status">
          <p className="notice-title">
            <Icon name="alert" /> Automatic label reading is not available on this server right now.
          </p>
          <p>You can still fill in the form, but checks will not run. Please tell your administrator.</p>
        </div>
      )}

      <div hidden={mode !== "one"}>
        <SingleLabel ruleset={ruleset} implementedById={implementedById} token={token} setToken={setToken} />
      </div>
      <div hidden={mode !== "many"}>
        {batchReady && (
          <BatchMode
            api={batchApi}
            token={token}
            setToken={setToken}
            implementedById={implementedById}
            devSample={devSample}
            active={mode === "many"}
          />
        )}
      </div>
    </div>
  );
}
