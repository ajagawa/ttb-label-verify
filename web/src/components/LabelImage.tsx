import { useState } from "react";
import type { FieldResult, VerificationResult } from "../api/types";
import { aspectMismatch, bboxToPercent, percentStyle } from "../lib/overlay";

interface Props {
  imageUrl: string;
  fileName: string;
  result: VerificationResult | null;
  selectedId: string | null;
  onSelect: (fieldId: string) => void;
  /** Tone per field id, so outlines echo the row's status style. */
  toneById: Record<string, string>;
}

/**
 * The uploaded label with the evidence highlight.
 *
 * Requirement (Dave Morrison, 28-year veteran): the tool has to show its
 * evidence. Selecting a result row outlines the exact region the value was
 * read from and dims everything else, so the agent can check the tool's
 * reading with their own eyes in one glance.
 *
 * Boxes are positioned in percentages of the server's image size — see
 * lib/overlay.ts — so they stay aligned at any zoom level or window width.
 */
export function LabelImage({ imageUrl, fileName, result, selectedId, onSelect, toneById }: Props) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [broken, setBroken] = useState(false);

  const located: Array<{ field: FieldResult; style: ReturnType<typeof percentStyle>; nearTop: boolean }> = [];
  if (result) {
    for (const field of result.fields) {
      if (!field.bbox) continue;
      const pct = bboxToPercent(field.bbox, result.image_width, result.image_height);
      if (pct) located.push({ field, style: percentStyle(pct), nearTop: pct.top < 6 });
    }
  }
  const selected = located.find((l) => l.field.field_id === selectedId) ?? null;
  const selectedField = result?.fields.find((f) => f.field_id === selectedId) ?? null;
  const misaligned =
    result && natural ? aspectMismatch(natural.w, natural.h, result.image_width, result.image_height) : false;

  if (broken) {
    // Most likely a TIFF: only Safari displays those. The results are still
    // valid; only the picture is missing, so say exactly that.
    return (
      <p className="notice notice-info">
        Your browser cannot display this image file ({fileName}), so the highlight cannot be drawn. The check
        itself still works — the results are unaffected. Converting the image to JPEG or PNG will show it here.
      </p>
    );
  }

  return (
    <figure className="label-figure">
      <div className={`label-stage${selected ? " has-selection" : ""}`}>
        <img
          src={imageUrl}
          alt={`Label image: ${fileName}`}
          onLoad={(e) => {
            setBroken(false);
            setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight });
          }}
          onError={() => setBroken(true)}
        />
        {/* Mouse convenience only: the result rows are the keyboard and
            screen-reader path to the same action, so these are hidden from
            assistive technology rather than duplicated as tab stops. */}
        {located.map(({ field, style, nearTop }) => {
          const isSelected = field.field_id === selectedId;
          return (
            <div
              key={field.field_id}
              aria-hidden="true"
              className={`bbox tone-${toneById[field.field_id] ?? "review"}${isSelected ? " is-selected" : ""}${nearTop ? " tag-below" : ""}`}
              style={style}
              onClick={() => onSelect(field.field_id)}
              title={field.label}
            >
              {isSelected && <span className="bbox-tag">{field.label}</span>}
            </div>
          );
        })}
      </div>
      <figcaption className="label-caption" aria-live="polite">
        {!result && "Your label image. Results will be highlighted here after checking."}
        {result && !selectedField && "Select a result in the list to see where it was read on the label."}
        {result && selectedField && selected && (
          <>
            Showing where <strong>{selectedField.label}</strong> was read on the label.
          </>
        )}
        {result && selectedField && !selected && (
          <>
            <strong>{selectedField.label}</strong> was not located on the label, so there is nothing to
            highlight.
          </>
        )}
      </figcaption>
      {misaligned && (
        <p className="notice notice-review">
          This image is displayed with a different shape than the one the tool read (often a rotated phone photo).
          The highlights may be in the wrong place — rely on the text in the results instead.
        </p>
      )}
    </figure>
  );
}
