import type { CheckResult, TextDifference } from "../api/types";
import { formatNumber, formatPercent, humanizeKey } from "../lib/format";
import { checkPresentation, differenceLabel } from "../lib/presentation";
import { Icon } from "./Icon";

/**
 * Formatting checks for the government warning (27 CFR 16.21 / 16.22).
 *
 * Requirement (Jenny Park): "GOVERNMENT WARNING" in caps and bold, text exact.
 * Some of these are measured from pixels and cannot be adjudicated, so
 * advisory checks show their numbers as a measurement for the agent's
 * judgement, and "not evaluable" says plainly that it could not be checked —
 * neither is ever dressed up as pass or fail.
 */
export function CheckList({ checks }: { checks: CheckResult[] }) {
  if (checks.length === 0) return null;
  return (
    <div className="subsection">
      <h4>Formatting checks</h4>
      <ul className="check-list">
        {checks.map((check) => {
          const p = checkPresentation(check.outcome);
          const measurements = Object.entries(check.measurement);
          return (
            <li key={check.id} className={`check tone-${p.tone}`}>
              <span className="check-status">
                <Icon name={p.icon} />
                <span>{p.label}</span>
              </span>
              <div className="check-body">
                <p className="check-summary">{check.summary}</p>
                {check.detail && <p className="muted">{check.detail}</p>}
                {measurements.length > 0 && (
                  <dl className="measurements">
                    {measurements.map(([key, value]) => (
                      <div key={key}>
                        <dt>{humanizeKey(key)}</dt>
                        <dd>{formatNumber(value)}</dd>
                      </div>
                    ))}
                  </dl>
                )}
                <p className="muted small">
                  {check.citation}
                  {check.confidence != null && <> · Tool confidence {formatPercent(check.confidence)}</>}
                </p>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * Word-level differences between the required statement and the label.
 *
 * "The statement differs" sends an agent hunting; "'may' appears as 'can'" is
 * something they can act on. Each row says in words what kind of difference
 * it is, so the meaning never depends on colour.
 */
export function DifferenceList({ differences }: { differences: TextDifference[] }) {
  if (differences.length === 0) return null;
  // Server order is by position already; sorting defensively keeps the list
  // readable left-to-right if that ever changes.
  const ordered = [...differences].sort((a, b) => a.position - b.position);
  return (
    <div className="subsection">
      <h4>Wording differences from the required statement</h4>
      <table className="diff-table">
        <thead>
          <tr>
            <th scope="col">Word</th>
            <th scope="col">Required text</th>
            <th scope="col">On the label</th>
            <th scope="col">Type</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((diff, i) => (
            <tr key={`${diff.position}-${i}`} className={`diff-${diff.kind}`}>
              {/* position is a 0-based token index (assumed); agents count from 1. */}
              <td data-label="Word">{diff.position + 1}</td>
              <td data-label="Required text">{diff.expected ? <q className="diff-word">{diff.expected}</q> : <span className="muted">(nothing — extra word)</span>}</td>
              <td data-label="On the label">{diff.found ? <q className="diff-word">{diff.found}</q> : <span className="muted">(not on label)</span>}</td>
              <td data-label="Type">{differenceLabel(diff)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
