import { useEffect, useState } from "react";
import type { VerificationResult } from "../api/types";
import { countTones, verdictPresentation } from "../lib/presentation";
import { LabelImage } from "./LabelImage";
import { ResultRow } from "./ResultRow";
import { QualityNotice, ResultsSummary } from "./ResultsSummary";

interface Props {
  result: VerificationResult;
  /** Object URL of the locally selected file, or null when it is not available. */
  imageUrl: string | null;
  fileName: string;
  implementedById: Record<string, boolean>;
  /**
   * Identity of the label shown. The result object is replaced on every poll
   * during a batch run; resetting the highlighted field on each would yank
   * the agent's selection away every two seconds, so reset on identity only.
   */
  labelKey: string;
  /** Offered when imageUrl is null: re-select the images to get highlights back. */
  onReattach?: () => void;
}

/**
 * The single-label result view, packaged for reuse in the batch worklist:
 * the same summary, quality notice, evidence overlay and checklist rows the
 * agent already knows from checking one label. One way of reading a result,
 * wherever it came from.
 */
export function ResultDetail({ result, imageUrl, fileName, implementedById, labelKey, onReattach }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // A new label: start on its first flagged, locatable field (as single mode
  // does) so the eye lands where the tool wants a second look.
  useEffect(() => {
    const first = result.fields.find((f) => {
      const tone = verdictPresentation(f, implementedById).tone;
      return f.bbox && tone !== "match" && tone !== "unchecked";
    });
    setSelectedId(first?.field_id ?? null);
    // Keyed on the label's identity only, deliberately (see `labelKey`).
  }, [labelKey]);

  const toneById: Record<string, string> = {};
  for (const f of result.fields) toneById[f.field_id] = verdictPresentation(f, implementedById).tone;

  return (
    <div className="detail-grid">
      <div className="detail-image">
        {imageUrl ? (
          <LabelImage
            imageUrl={imageUrl}
            fileName={fileName}
            result={result}
            selectedId={selectedId}
            onSelect={setSelectedId}
            toneById={toneById}
          />
        ) : (
          <div className="notice notice-info">
            <p>
              The highlights need the label images, which this page no longer has (it was reloaded). Choose the same
              images again to see them. The results below are complete either way.
            </p>
            {onReattach && (
              <button type="button" className="button button-secondary" onClick={onReattach}>
                Choose the images again
              </button>
            )}
          </div>
        )}
      </div>
      <div className="detail-results">
        <ResultsSummary result={result} counts={countTones(result.fields, implementedById)} />
        <QualityNotice quality={result.quality} />
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
      </div>
    </div>
  );
}
