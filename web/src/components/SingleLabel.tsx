import { useEffect, useRef, useState, type FormEvent } from "react";
import { ApiError, verifyLabel } from "../api/client";
import type { ApplicationForm, RulesetMetadata, VerificationResult } from "../api/types";
import { Icon } from "./Icon";
import { ImagePicker } from "./ImagePicker";
import { LabelImage } from "./LabelImage";
import { RecordForm, SAMPLE_VALUES, validateForm, type FormErrors } from "./RecordForm";
import { ResultRow } from "./ResultRow";
import { QualityNotice, ResultsSummary } from "./ResultsSummary";
import { friendlyError, type FriendlyError } from "../lib/errors";
import { checkFile, formatSeconds } from "../lib/format";
import { countTones, verdictPresentation } from "../lib/presentation";
import { MOCK_MODE } from "../lib/devMode";
import { TokenField } from "./TokenField";

const EMPTY_FORM: ApplicationForm = {
  brand_name: "",
  class_type: "",
  alcohol_content: "",
  net_contents: "",
  beverage_class: "distilled_spirits",
  label_width_mm: "",
};


type Status = "idle" | "checking" | "done";

interface Props {
  ruleset: RulesetMetadata | null;
  implementedById: Record<string, boolean>;
  token: string;
  setToken: (t: string) => void;
}

/**
 * "One label": label image on the left, application record and results on
 * the right. No navigation, no modals, no menus — the benchmark user is a
 * 73-year-old who has just learned video calls.
 */
export function SingleLabel({ ruleset, implementedById, token, setToken }: Props) {

  const [form, setForm] = useState<ApplicationForm>(EMPTY_FORM);
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [apiError, setApiError] = useState<FriendlyError | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [waitedMs, setWaitedMs] = useState(0);

  const [showToken, setShowToken] = useState(false);

  const resultsHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (import.meta.env.DEV && MOCK_MODE) {
      import("../dev/mock").then(async (dev) => {
        setForm({ ...EMPTY_FORM, ...SAMPLE_VALUES });
        replaceImage(await dev.loadSampleLabel());
      });
    }
  }, []);

  // The object URL is the only copy of the image the page holds; release it
  // whenever it is replaced so repeated checks do not leak memory.
  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  // Visible count-up while waiting. The previous tool's 30–40 s waits are
  // what these users remember; a moving number shows the tool is working.
  useEffect(() => {
    if (status !== "checking") return;
    const started = performance.now();
    setWaitedMs(0);
    const timer = window.setInterval(() => setWaitedMs(performance.now() - started), 100);
    return () => window.clearInterval(timer);
  }, [status]);

  function replaceImage(next: File) {
    setFile(next);
    setImageUrl(URL.createObjectURL(next));
    setFileError(null);
    // Results belong to the image they were read from; showing old
    // highlights over a new picture would be actively misleading.
    setResult(null);
    setSelectedId(null);
    setApiError(null);
    setStatus("idle");
  }

  function onFile(next: File) {
    const problem = checkFile(next);
    if (problem) {
      setFileError(problem);
      return;
    }
    replaceImage(next);
  }

  function onChange(key: keyof ApplicationForm, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }));
    if (formErrors[key]) setFormErrors((prev) => ({ ...prev, [key]: undefined }));
  }

  function startOver() {
    setForm(EMPTY_FORM);
    setFormErrors({});
    setFile(null);
    setImageUrl(null);
    setFileError(null);
    setResult(null);
    setSelectedId(null);
    setApiError(null);
    setStatus("idle");
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const errors = validateForm(form);
    setFormErrors(errors);
    const firstInvalid = Object.keys(errors)[0];
    if (!file) setFileError("Add a label image before checking.");
    if (firstInvalid) {
      document.getElementById(`f-${firstInvalid}`)?.focus();
      return;
    }
    if (!file) return;

    setApiError(null);
    setResult(null);
    setSelectedId(null);
    setStatus("checking");
    try {
      const next = import.meta.env.DEV && MOCK_MODE ? await (await import("../dev/mock")).mockVerify() : await verifyLabel(form, file, token);
      setResult(next);
      // Start with the first flagged item that can be shown on the label, so
      // the agent's eye lands where the tool wants a second look.
      const firstFlagged = next.fields.find((f) => {
        const tone = verdictPresentation(f, implementedById).tone;
        return f.bbox && (tone === "mismatch" || tone === "elevated" || tone === "review");
      });
      setSelectedId(firstFlagged?.field_id ?? null);
      setStatus("done");
      setShowToken(false);
      // Move keyboard/screen-reader focus to the results, which are otherwise
      // announced nowhere.
      requestAnimationFrame(() => resultsHeading.current?.focus());
    } catch (err) {
      const friendly =
        err instanceof ApiError ? friendlyError(err.status, err.detail) : friendlyError(null, null);
      setApiError(friendly);
      if (friendly.needsToken) setShowToken(true);
      setStatus("idle");
    }
  }

  const counts = result ? countTones(result.fields, implementedById) : null;
  const toneById: Record<string, string> = {};
  for (const f of result?.fields ?? []) toneById[f.field_id] = verdictPresentation(f, implementedById).tone;

  return (
      <main className="layout">
        <section className="col-image" aria-labelledby="image-heading">
          <h2 id="image-heading">Label image</h2>
          <ImagePicker hasImage={!!imageUrl} onFile={onFile} error={fileError}>
            {imageUrl && file && (
              <LabelImage
                imageUrl={imageUrl}
                fileName={file.name}
                result={result}
                selectedId={selectedId}
                onSelect={setSelectedId}
                toneById={toneById}
              />
            )}
          </ImagePicker>
        </section>

        <div className="col-work">
          <form onSubmit={onSubmit} noValidate aria-labelledby="form-heading">
            <h2 id="form-heading">Application record</h2>
            <RecordForm
              form={form}
              errors={formErrors}
              beverageClasses={ruleset?.beverage_classes ?? {}}
              onChange={onChange}
            />

            {showToken && <TokenField id="f-token" token={token} onChange={setToken} />}

            <div className="actions">
              <button type="submit" className="button button-primary" disabled={status === "checking"}>
                {status === "checking" ? "Checking…" : "Check label"}
              </button>
              <button
                type="button"
                className="button button-secondary"
                onClick={() => setForm((prev) => ({ ...prev, ...SAMPLE_VALUES }))}
              >
                Use sample values
              </button>
              <button type="button" className="button button-quiet" onClick={startOver}>
                Start a new label
              </button>
            </div>

            {apiError && (
              <div className="notice notice-error" role="alert">
                <p className="notice-title">
                  <Icon name="cross" /> {apiError.title}
                </p>
                <p>{apiError.message}</p>
              </div>
            )}
          </form>

          <section className="results" aria-labelledby="results-heading" aria-busy={status === "checking"}>
            <h2 id="results-heading" ref={resultsHeading} tabIndex={-1}>
              Results
            </h2>

            {status === "checking" && (
              <p className="checking">
                <span role="status">Checking the label…</span>{" "}
                <span className="muted" aria-hidden="true">
                  {formatSeconds(waitedMs)}
                </span>
              </p>
            )}

            {status === "idle" && !result && <WhatWeCheck ruleset={ruleset} />}

            {result && counts && imageUrl && (
              <>
                <ResultsSummary result={result} counts={counts} />
                <QualityNotice quality={result.quality} />
                {/* Server order is the agents' printed checklist order.
                    Deliberately not re-sorted by severity. */}
                <ol className="result-list">
                  {result.fields.map((field) => (
                    <ResultRow
                      key={field.field_id}
                      field={field}
                      presentation={verdictPresentation(field, implementedById)}
                      selected={field.field_id === selectedId}
                      onSelect={() => setSelectedId((cur) => (cur === field.field_id ? null : field.field_id))}
                      imageUrl={imageUrl}
                      imageWidth={result.image_width}
                      imageHeight={result.image_height}
                    />
                  ))}
                </ol>
              </>
            )}
          </section>
        </div>
      </main>
  );
}

/**
 * Before the first check: say plainly what is and is not automated, so a
 * partial prototype is never taken for a complete one.
 */
function WhatWeCheck({ ruleset }: { ruleset: RulesetMetadata | null }) {
  if (!ruleset) return <p className="muted">Add the label image and application details, then press Check label.</p>;
  return (
    <div>
      <p>Add the label image and application details, then press Check label. The tool looks at:</p>
      <ul className="scope-list">
        {ruleset.fields.map((f) => (
          // No tick for implemented items: before a check, a tick would read
          // as "passed". The unimplemented ones carry the dash-and-text style
          // used for them in the results.
          <li key={f.id} className={f.implemented ? "" : "tone-unchecked"}>
            {!f.implemented && <Icon name="dash" />}
            <span>
              {f.label}
              {f.implemented ? (
                <span className="muted"> — checked automatically</span>
              ) : (
                <strong> — not checked automatically; verify manually</strong>
              )}
            </span>
          </li>
        ))}
      </ul>
      <p className="muted small">
        Rule set {ruleset.version}, effective {ruleset.effective_date}.
      </p>
    </div>
  );
}
