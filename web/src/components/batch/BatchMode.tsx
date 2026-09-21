import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { pairingFromDetail, type BatchApi, type UploadHandle } from "../../api/batch";
import { ApiError } from "../../api/client";
import type { BatchStatus } from "../../api/types";
import { displayItem, needsFetch, prefetchTarget, ResultCache } from "../../lib/itemCache";
import { findLocalFile, hasAcceptedExtension, sortSelection } from "../../lib/batchFiles";
import { browserCodec, prepareImage } from "../../lib/downscale";
import { friendlyError, isCapacityError, type FriendlyError } from "../../lib/errors";
import { formatBytes, formatDuration, formatRemaining, percent, pollDelay } from "../../lib/progress";
import { getParam, setParam } from "../../lib/url";
import {
  countByCategory,
  filterItems,
  findByKey,
  positionOf,
  shouldKeepPolling,
  type WorklistFilter,
} from "../../lib/worklist";
import { DownloadButton } from "../DownloadButton";
import { Icon } from "../Icon";
import { TokenField } from "../TokenField";
import { BatchDetail } from "./BatchDetail";
import { BatchPicker } from "./BatchPicker";
import { PairingPanel, type PairingState } from "./PairingPanel";
import { cssId, Worklist } from "./Worklist";

type Phase = "choose" | "preparing" | "uploading" | "busy" | "tracking";

const FILTER_LABEL: Record<WorklistFilter, string> = {
  attention: "Needs attention",
  clear: "All clear",
  pending: "Waiting or reading",
  unchecked: "Not checked",
};

interface Props {
  api: BatchApi;
  token: string;
  setToken: (t: string) => void;
  implementedById: Record<string, boolean>;
  /** DEV ONLY: offers a canned sample selection in mock mode. */
  devSample?: () => Promise<File[]>;
  /** False while the other mode is showing: keyboard shortcuts must not fire. */
  active: boolean;
}

/**
 * "Many labels": choose files → pairing check → upload → worst-first worklist.
 * Everything happens on this one screen, top to bottom, in that order.
 */
export function BatchMode({ api, token, setToken, implementedById, devSample, active }: Props) {
  const [manifest, setManifest] = useState<File | null>(null);
  const [images, setImages] = useState<File[]>([]);
  const [pairing, setPairing] = useState<PairingState>({ kind: "waiting", missing: "both" });
  const [pickerNote, setPickerNote] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>(() => (getParam("batch") ? "tracking" : "choose"));
  const [prep, setPrep] = useState({ done: 0, total: 0, resized: 0, saved: 0 });
  const [upload, setUpload] = useState({ sent: 0, total: 0 });
  const uploadHandle = useRef<UploadHandle | null>(null);
  const [startError, setStartError] = useState<FriendlyError | null>(null);
  const [showToken, setShowToken] = useState(false);
  // "Server busy with other batches": the prepared files wait here and the
  // upload is retried after the server's Retry-After.
  const prepared = useRef<File[]>([]);
  const [retryIn, setRetryIn] = useState(0);

  const [batchId, setBatchId] = useState<string | null>(() => getParam("batch"));
  const [status, setStatus] = useState<BatchStatus | null>(null);
  const [pollError, setPollError] = useState<FriendlyError | null>(null);
  const [confirmStop, setConfirmStop] = useState(false);
  const [stopping, setStopping] = useState(false);

  const [filter, setFilter] = useState<WorklistFilter>("attention");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [focusToken, setFocusToken] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const reattachInput = useRef<HTMLInputElement>(null);
  // Full results by manifest row, for the life of this batch view.
  const cache = useRef(new ResultCache());
  const [, setCacheVersion] = useState(0);
  const bump = useCallback(() => setCacheVersion((v) => v + 1), []);
  const [loadErrors, setLoadErrors] = useState<Record<number, string>>({});
  const keepGoing = useRef<HTMLButtonElement>(null);

  // ------------------------------------------------------------ choosing

  const onFiles = (files: File[]) => {
    const { csvs, images: picked } = sortSelection(files);
    setPickerNote(null);
    if (csvs.length > 0) {
      setManifest(csvs[0] ?? null);
      if (csvs.length > 1) {
        setPickerNote(`More than one CSV was chosen; using “${csvs[0]?.name}” as the manifest.`);
      }
    }
    if (picked.length > 0) {
      // Adding more files extends the selection; the same file twice is ignored.
      setImages((prev) => sortSelection([...prev, ...picked]).images);
    }
  };

  // Pairing check: as soon as both halves are present, and again whenever
  // either changes. Filenames only — no image bytes leave the computer here.
  useEffect(() => {
    if (phase !== "choose") return;
    if (!manifest || images.length === 0) {
      setPairing({
        kind: "waiting",
        missing: !manifest && images.length === 0 ? "both" : !manifest ? "manifest" : "images",
      });
      return;
    }
    let stale = false;
    setPairing({ kind: "checking" });
    const timer = window.setTimeout(() => {
      api
        .check(
          manifest,
          images.map((f) => f.name),
          token,
        )
        .then(
          (report) => !stale && setPairing({ kind: "ready", report }),
          (err: unknown) => {
            if (stale) return;
            const e = err instanceof ApiError ? friendlyError(err.status, err.detail, "batch") : friendlyError(null);
            if (e.needsToken) setShowToken(true);
            setPairing({ kind: "error", error: e });
          },
        );
    }, 250);
    return () => {
      stale = true;
      window.clearTimeout(timer);
    };
  }, [manifest, images, phase, token, api]);

  // ------------------------------------------------------------ starting

  async function start() {
    if (!manifest) return;
    setStartError(null);
    // Unsupported files were already reported by the pairing check; sending
    // them would only cost bytes the server throws away.
    const candidates = images.filter((f) => hasAcceptedExtension(f.name));

    setPhase("preparing");
    setPrep({ done: 0, total: candidates.length, resized: 0, saved: 0 });
    const ready: File[] = [];
    let resized = 0;
    let saved = 0;
    for (const [i, file] of candidates.entries()) {
      const outcome = await prepareImage(file, browserCodec);
      ready.push(outcome.file);
      if (outcome.kind === "resized") {
        resized += 1;
        saved += file.size - outcome.file.size;
      }
      setPrep({ done: i + 1, total: candidates.length, resized, saved });
    }
    prepared.current = ready;
    await uploadPrepared();
  }

  async function uploadPrepared() {
    if (!manifest) return;
    const files = prepared.current;
    setPhase("uploading");
    setUpload({ sent: 0, total: files.reduce((s, f) => s + f.size, 0) });
    const handle = api.submit(manifest, files, token, (sent, total) => setUpload({ sent, total }));
    uploadHandle.current = handle;
    try {
      const next = await handle.promise;
      setStatus(next);
      setBatchId(next.batch_id);
      setParam("batch", next.batch_id);
      setPollError(null);
      prepared.current = [];
      setPhase("tracking");
    } catch (err) {
      if (err instanceof ApiError && isCapacityError(err.status, err.detail)) {
        // Honour the server's Retry-After (30–600 s); 30 s if it sent none.
        setRetryIn(err.retryAfter ?? 30);
        setPhase("busy");
        return;
      }
      setPhase("choose");
      if (err instanceof DOMException && err.name === "AbortError") {
        setStartError({
          title: "Upload cancelled",
          message: "Nothing was checked. Your files are still chosen; start again when ready.",
          needsToken: false,
        });
        return;
      }
      const e = err instanceof ApiError ? friendlyError(err.status, err.detail, "batch") : friendlyError(null);
      if (e.needsToken) setShowToken(true);
      if (err instanceof ApiError) {
        const report = pairingFromDetail(err.detail);
        if (report) setPairing({ kind: "ready", report });
      }
      setStartError(e);
    } finally {
      uploadHandle.current = null;
    }
  }

  // Count down, then retry the upload with the already-prepared files.
  useEffect(() => {
    if (phase !== "busy") return;
    if (retryIn <= 0) {
      void uploadPrepared();
      return;
    }
    const timer = window.setTimeout(() => setRetryIn((n) => n - 1), 1000);
    return () => window.clearTimeout(timer);
    // uploadPrepared reads current state; re-running on its identity would loop.
  }, [phase, retryIn]);

  // ------------------------------------------------------------ tracking

  const terminal = status ? !shouldKeepPolling(status) : false;

  // Poll ~every 2 s while visible, slower while the tab is hidden, and at
  // once when the agent comes back to the tab. Polling ends when the server
  // says nothing more will change: done, or stopped with no label RUNNING.
  useEffect(() => {
    if (phase !== "tracking" || !batchId) return;
    let alive = true;
    let timer = 0;
    let stopped = false;
    const tick = async () => {
      window.clearTimeout(timer);
      if (stopped) return;
      try {
        const next = await api.get(batchId, token);
        if (!alive) return;
        setStatus(next);
        setPollError(null);
        if (!shouldKeepPolling(next)) {
          stopped = true; // also ignores a later visibilitychange
          return;
        }
      } catch (err) {
        if (!alive) return;
        const code = err instanceof ApiError ? err.status : null;
        const e = friendlyError(code, err instanceof ApiError ? err.detail : null, "batch");
        setPollError(e);
        if (code === 404) {
          setParam("batch", null);
          return; // gone for good; stop polling
        }
        if (e.needsToken) {
          setShowToken(true);
          return;
        }
      }
      if (alive) timer = window.setTimeout(tick, pollDelay(document.hidden));
    };
    const onVisible = () => {
      if (!document.hidden) void tick();
    };
    void tick();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [phase, batchId, token, api]);

  // A new batch view starts with an empty result cache.
  useEffect(() => {
    cache.current = new ResultCache();
    setLoadErrors({});
    bump();
  }, [batchId, bump]);

  const fetchItem = useCallback(
    (row: number) => {
      if (!batchId || cache.current.isLoading(row)) return;
      const owner = cache.current;
      owner.begin(row);
      setLoadErrors(({ [row]: _gone, ...rest }) => rest);
      bump();
      api
        .getItem(batchId, row, token)
        .then(
          (item) => owner.put(item),
          (err: unknown) => {
            const e = friendlyError(err instanceof ApiError ? err.status : null, null, "batch");
            setLoadErrors((prev) => ({ ...prev, [row]: e.message }));
          },
        )
        .finally(() => {
          owner.end(row);
          bump();
        });
    },
    [api, batchId, token, bump],
  );

  async function stop() {
    if (!batchId) return;
    setStopping(true);
    try {
      const final = await api.cancel(batchId, token);
      setStatus(final);
    } catch (err) {
      setPollError(friendlyError(err instanceof ApiError ? err.status : null, null, "batch"));
    } finally {
      // The batch id stays in the URL: a stopped batch keeps its finished
      // results and CSV until it expires, so a reload should still find it.
      setStopping(false);
      setConfirmStop(false);
    }
  }

  function startOver() {
    setParam("batch", null);
    setBatchId(null);
    setStatus(null);
    setPollError(null);
    setSelectedKey(null);
    setFilter("attention");
    setManifest(null);
    setImages([]);
    setStartError(null);
    setPhase("choose");
  }

  // ------------------------------------------------------------ worklist

  const items = useMemo(() => status?.items ?? [], [status]);
  const counts = useMemo(() => countByCategory(items), [items]);
  const visible = useMemo(() => filterItems(items, filter), [items, filter]);
  const queue = visible;
  // Selection is by key, looked up afresh in every poll's list: rows arriving
  // above it renumber it but never move the agent off the label they are on.
  const selected = findByKey(items, selectedKey);
  const position = positionOf(queue, selectedKey);
  const shown = selected ? displayItem(selected, cache.current) : null;

  // A filter that has emptied out (e.g. "Waiting" once a stop turns those
  // labels into "Not checked") falls back to the default view.
  useEffect(() => {
    if ((filter === "pending" || filter === "unchecked") && counts[filter] === 0) setFilter("attention");
  }, [filter, counts]);

  // Fetch the open label's full result when it is finished and not cached —
  // including a pending label that a later poll shows as done.
  useEffect(() => {
    if (selected && needsFetch(selected, cache.current)) fetchItem(selected.row_number);
  });

  // Then prefetch one ahead, so → feels instant. Only once the open label is
  // settled, so the prefetch never competes with what the agent is waiting on.
  useEffect(() => {
    if (!selected || needsFetch(selected, cache.current) || cache.current.isLoading(selected.row_number)) return;
    const next = prefetchTarget(queue, selectedKey, cache.current);
    if (next) fetchItem(next.row_number);
  });

  const open = useCallback((key: string) => {
    setSelectedKey(key);
    setFocusToken((n) => n + 1);
  }, []);
  const go = useCallback((key: string) => setSelectedKey(key), []);

  const backToList = () => {
    const key = selectedKey;
    if (!key) return;
    const row = document.getElementById(`wl-${cssId(key)}`);
    row?.scrollIntoView({ block: "center" });
    (row?.querySelector("button") as HTMLButtonElement | null)?.focus({ preventScroll: true });
  };

  // One object URL at a time — for the label being looked at — from the file
  // the agent chose. The server never returns images.
  const localFile = selected ? findLocalFile(images, selected.filename) : null;
  useEffect(() => {
    if (!localFile) {
      setImageUrl(null);
      return;
    }
    const url = URL.createObjectURL(localFile);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [localFile]);

  // A quiet milestone announcement for screen readers — every 10%, not every
  // poll, which would talk over everything else.
  const milestone =
    status && status.total > 0 ? Math.floor((status.completed / status.total) * 10) * 10 : null;

  // ------------------------------------------------------------ render

  const busy = phase === "preparing" || phase === "uploading" || phase === "busy";

  return (
    <div className="batch">
      {phase !== "tracking" && (
        <section className="panel" aria-labelledby="batch-choose-heading">
          <h2 id="batch-choose-heading">1. Choose the manifest and label images</h2>
          <BatchPicker
            manifest={manifest}
            images={images}
            onDownloadTemplate={() => api.downloadTemplate(token)}
            onFiles={onFiles}
            onClearImages={() => setImages([])}
            onClearManifest={() => setManifest(null)}
            disabled={busy}
          />
          {import.meta.env.DEV && devSample && phase === "choose" && (
            <button type="button" className="button button-quiet dev-only" onClick={async () => onFiles(await devSample())}>
              DEV: load a sample batch
            </button>
          )}
          {pickerNote && <p className="notice notice-info">{pickerNote}</p>}
        </section>
      )}

      {phase !== "tracking" && (
        <section className="panel" aria-labelledby="batch-check-heading">
          <h2 id="batch-check-heading">2. Check they line up, then start</h2>
          {showToken && <TokenField id="batch-token" token={token} onChange={setToken} />}
          {startError && (
            <div className="notice notice-error" role="alert">
              <p className="notice-title">
                <Icon name="cross" /> {startError.title}
              </p>
              <p>{startError.message}</p>
            </div>
          )}
          {phase === "choose" && <PairingPanel state={pairing} onStart={start} busy={busy} />}

          {phase === "preparing" && (
            <div className="progress-block" role="status">
              <p className="progress-line">
                Preparing images: {prep.done} of {prep.total}
              </p>
              <progress max={prep.total || 1} value={prep.done} aria-label="Preparing images" />
              <p className="muted small">
                Very large photos are made smaller here first, so the upload is faster. The originals are not
                changed.
              </p>
            </div>
          )}

          {phase === "busy" && (
            <div className="notice notice-info" role="status">
              <p className="notice-title">
                <Icon name="clock" /> The server is busy with other batches — trying again in {retryIn} s
              </p>
              <p>Your files are ready and still chosen. You do not need to do anything.</p>
              <div className="actions">
                <button type="button" className="button button-secondary" onClick={() => setRetryIn(0)}>
                  Try now
                </button>
                <button type="button" className="button button-quiet" onClick={() => setPhase("choose")}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          {phase === "uploading" && (
            <div className="progress-block">
              <p className="progress-line" role="status">
                Uploading: {formatBytes(upload.sent)} of {formatBytes(upload.total)} ({percent(upload.sent, upload.total)}%)
              </p>
              <progress max={upload.total || 1} value={upload.sent} aria-label="Upload progress" />
              {prep.resized > 0 && (
                <p className="muted small">
                  {prep.resized} large {prep.resized === 1 ? "photo was" : "photos were"} made smaller first, saving{" "}
                  {formatBytes(prep.saved)}.
                </p>
              )}
              <button type="button" className="button button-quiet" onClick={() => uploadHandle.current?.abort()}>
                Cancel upload
              </button>
            </div>
          )}
        </section>
      )}

      {phase === "tracking" && (
        <section className="panel" aria-labelledby="batch-results-heading">
          <h2 id="batch-results-heading">Results</h2>
          {showToken && <TokenField id="batch-token-2" token={token} onChange={setToken} />}

          {pollError && (
            <div className="notice notice-error" role="alert">
              <p className="notice-title">
                <Icon name="cross" /> {pollError.title}
              </p>
              <p>{pollError.message}</p>
              {!status && (
                <button type="button" className="button button-secondary" onClick={startOver}>
                  Start a new batch
                </button>
              )}
            </div>
          )}

          {!status && !pollError && <p role="status">Finding your batch…</p>}

          {status && (
            <>
              <BatchHeader status={status} onDownloadCsv={() => api.downloadResults(status.batch_id, token)} />
              <p className="visually-hidden" aria-live="polite">
                {status.state === "done"
                  ? `All ${status.total} labels checked.`
                  : milestone != null
                    ? `${milestone} percent checked.`
                    : ""}
              </p>

              {/* Stop is offered only while the batch itself is running — not
                  while a stopped batch finishes its last in-flight label. */}
              {(status.state === "queued" || status.state === "running") && (
                <div className="stop-row">
                  {!confirmStop ? (
                    <button
                      type="button"
                      className="button button-quiet"
                      onClick={() => {
                        setConfirmStop(true);
                        requestAnimationFrame(() => keepGoing.current?.focus());
                      }}
                    >
                      Stop checking
                    </button>
                  ) : (
                    <div className="notice notice-review confirm" role="group" aria-labelledby="confirm-stop">
                      <p id="confirm-stop" className="notice-title">
                        <Icon name="alert" /> Stop checking? Labels already checked stay in the list and in the
                        download. The rest will not be checked.
                      </p>
                      <div className="actions">
                        <button type="button" className="button button-danger" onClick={stop} disabled={stopping}>
                          {stopping ? "Stopping…" : "Stop"}
                        </button>
                        <button
                          ref={keepGoing}
                          type="button"
                          className="button button-secondary"
                          onClick={() => setConfirmStop(false)}
                        >
                          Keep going
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {(status.state === "done" || status.state === "cancelled") && (
                <div className="actions">
                  <button type="button" className="button button-quiet" onClick={startOver}>
                    Start a new batch
                  </button>
                </div>
              )}

              <Worklist
                items={visible}
                counts={counts}
                filter={filter}
                onFilter={setFilter}
                selectedKey={selectedKey}
                onOpen={open}
                running={!terminal}
              />
            </>
          )}
        </section>
      )}

      {phase === "tracking" && shown && (
        <section className="panel">
          <BatchDetail
            item={shown}
            loading={cache.current.isLoading(shown.row_number)}
            loadError={loadErrors[shown.row_number] ?? null}
            onRetry={() => fetchItem(shown.row_number)}
            position={position}
            filterLabel={FILTER_LABEL[filter]}
            imageUrl={imageUrl}
            implementedById={implementedById}
            onGo={go}
            onBackToList={backToList}
            onReattach={() => reattachInput.current?.click()}
            focusToken={focusToken}
            keysEnabled={active}
          />
          {/* Re-select images after a reload, for the highlights only. */}
          <input
            ref={reattachInput}
            type="file"
            multiple
            accept="image/*"
            className="visually-hidden"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              e.target.value = "";
              if (files.length) setImages((prev) => sortSelection([...prev, ...files]).images);
            }}
          />
        </section>
      )}
    </div>
  );
}

/** Progress while running; the total time and the CSV download when finished or stopped. */
function BatchHeader({ status, onDownloadCsv }: { status: BatchStatus; onDownloadCsv: () => Promise<void> }) {
  const expires = new Date(status.expires_at);
  const expiresText = Number.isNaN(expires.getTime())
    ? null
    : expires.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const download = (
    <p>
      <DownloadButton label="Download results (CSV)" download={onDownloadCsv} />
      {expiresText && <span className="muted"> — the server keeps this batch until {expiresText}.</span>}
    </p>
  );
  const failedNote =
    status.failed > 0 ? (
      <p>
        {status.failed} {status.failed === 1 ? "image" : "images"} could not be read — they are in the list under
        “Needs attention”.
      </p>
    ) : null;

  if (status.state === "done") {
    return (
      <div className="batch-header">
        <p className="elapsed">
          Checked {status.total} {status.total === 1 ? "label" : "labels"} in{" "}
          <strong>{formatDuration(status.elapsed_ms)}</strong>
        </p>
        {failedNote}
        {download}
      </div>
    );
  }
  if (status.state === "cancelled") {
    const notChecked = status.items.filter((i) => i.state === "skipped").length;
    const finishing = status.items.filter((i) => i.state === "running").length;
    return (
      <div className="batch-header">
        <p className="elapsed">
          <Icon name="dash" /> Stopped after checking {status.completed} of {status.total} labels
        </p>
        {finishing > 0 && (
          <p>
            {finishing} {finishing === 1 ? "label was" : "labels were"} already being read when you stopped;{" "}
            {finishing === 1 ? "it will finish" : "they will finish"} and appear in the list.
          </p>
        )}
        {notChecked > 0 && (
          <p>
            {notChecked} {notChecked === 1 ? "label was" : "labels were"} <strong>not checked</strong> — they are
            listed under “Not checked”. Check them by hand or in a new batch.
          </p>
        )}
        {failedNote}
        {download}
      </div>
    );
  }
  return (
    <div className="batch-header">
      <p className="progress-line">
        {status.state === "queued" ? (
          "Waiting to start…"
        ) : (
          <>
            {status.completed} of {status.total} checked — {formatRemaining(status.estimated_remaining_ms)}
          </>
        )}
      </p>
      <progress max={status.total || 1} value={status.completed} aria-label="Labels checked" />
      <p className="muted small">
        You can start working through the list below now; it fills in as labels are checked, most urgent first.
      </p>
    </div>
  );
}
