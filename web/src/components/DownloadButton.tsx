import { useState } from "react";
import { ApiError } from "../api/client";
import { friendlyError } from "../lib/errors";
import { Icon } from "./Icon";

/**
 * A download that needs the access token. It looks and reads like a link,
 * but it is a button that fetches the file with the `X-Access-Token` header
 * and saves it from script — a plain link would have to carry the token in
 * its URL, which ends up in proxy logs.
 */
export function DownloadButton({ label, download }: { label: string; download: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <>
      <button
        type="button"
        className="linklike download-link"
        aria-busy={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await download();
          } catch (err) {
            const e = err instanceof ApiError ? friendlyError(err.status, err.detail, "batch") : friendlyError(null);
            setError(`${e.title}. ${e.message}`);
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? "Downloading…" : label}
      </button>
      {error && (
        <span className="field-error inline-error" role="alert">
          <Icon name="cross" /> {error}
        </span>
      )}
    </>
  );
}
