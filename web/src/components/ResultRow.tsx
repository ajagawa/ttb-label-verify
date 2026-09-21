import type { FieldResult } from "../api/types";
import { cropView } from "../lib/overlay";
import { formatPercent } from "../lib/format";
import { type Presentation } from "../lib/presentation";
import { Icon } from "./Icon";
import { CheckList, DifferenceList } from "./WarningDetails";

interface Props {
  field: FieldResult;
  presentation: Presentation;
  selected: boolean;
  onSelect: () => void;
  /** null when the image is not available locally (batch after a reload). */
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
}

/**
 * One line of the agent's checklist.
 *
 * The header is a single large button (the whole row width, >= 44 px tall):
 * "no hunting for buttons". Pressing it highlights the region on the label
 * image and, inside the row, shows that region enlarged next to the reading —
 * so the evidence is visible even at 200% zoom when the full image has
 * scrolled away.
 */
export function ResultRow({ field, presentation: p, selected, onSelect, imageUrl, imageWidth, imageHeight }: Props) {
  const unchecked = p.tone === "unchecked";
  const crop = field.bbox && imageUrl ? cropView(field.bbox, imageWidth, imageHeight) : null;
  const headingId = `row-${field.field_id}`;

  return (
    <li className={`result tone-${p.tone}${selected ? " is-selected" : ""}`}>
      <h3 className="result-heading" id={headingId}>
        <button type="button" className="result-button" aria-pressed={selected} onClick={onSelect}>
          <span className="result-name">{field.label}</span>
          <span className={`badge tone-${p.tone}`}>
            <Icon name={p.icon} />
            <span>{p.label}</span>
          </span>
          <span className="result-action">
            {!imageUrl || !field.bbox ? "Details" : selected ? "Shown on label" : "Show on label"}
          </span>
        </button>
      </h3>

      <div className="result-body">
        <p className="result-hint">{p.hint}</p>

        <dl className="compare">
          <div>
            <dt>Application says</dt>
            <dd>{field.expected ? <q>{field.expected}</q> : <span className="muted">Fixed by regulation</span>}</dd>
          </div>
          {!unchecked && (
            <div>
              <dt>Label reads</dt>
              <dd>
                {field.found ? <q>{field.found}</q> : <span className="muted">Not found on the label</span>}
              </dd>
            </div>
          )}
        </dl>

        {selected && crop && imageUrl && (
          <div className="evidence">
            <p className="evidence-caption">Read from this part of the label:</p>
            <div
              className="evidence-window"
              style={{
                aspectRatio: String(crop.aspect),
                // Cap height (~180 px) and magnification (3x) without
                // distorting: a small word should be legible, not a poster.
                width: `min(100%, ${Math.round(Math.min(180 * crop.aspect, crop.sourceWidth * 3))}px)`,
              }}
            >
              <img src={imageUrl} alt="" style={crop.img} />
            </div>
          </div>
        )}

        {!unchecked && <p className="reason">{field.reason}</p>}

        <CheckList checks={field.checks} />
        <DifferenceList differences={field.differences} />

        <p className="meta small">
          {field.citation_url ? (
            <a href={field.citation_url} target="_blank" rel="noreferrer">
              {field.citation}
              <span className="visually-hidden"> (opens eCFR in a new tab)</span>
            </a>
          ) : (
            field.citation
          )}
          {!unchecked && field.found && <> · Text-reading confidence {formatPercent(field.ocr_confidence)}</>}
        </p>
      </div>
    </li>
  );
}
