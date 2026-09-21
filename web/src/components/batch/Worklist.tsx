import type { BatchItem } from "../../api/types";
import { tierPresentation } from "../../lib/presentation";
import { describeCounts, emptyMessage, itemKey, type WorklistFilter } from "../../lib/worklist";
import { Icon } from "../Icon";

interface Props {
  items: BatchItem[];
  counts: Record<WorklistFilter, number>;
  filter: WorklistFilter;
  onFilter: (f: WorklistFilter) => void;
  selectedKey: string | null;
  onOpen: (key: string) => void;
  running: boolean;
}

const CHIPS: Array<{ id: WorklistFilter; label: string; icon: "alert" | "check" | "clock" | "dash" }> = [
  { id: "attention", label: "Needs attention", icon: "alert" },
  { id: "clear", label: "All clear", icon: "check" },
  { id: "pending", label: "Waiting or reading", icon: "clock" },
  { id: "unchecked", label: "Not checked", icon: "dash" },
];

/**
 * The worst-first worklist.
 *
 * Rows are rendered exactly in the server's order (rules/triage.py is the
 * single definition of "worst first"); filtering only hides rows. Each row's
 * file name is a real button — Tab reaches it, Enter or Space opens it —
 * and a click anywhere on the row does the same for mouse users.
 */
export function Worklist({ items, counts, filter, onFilter, selectedKey, onOpen, running }: Props) {
  // "Waiting" and "Not checked" appear only when they have something in them.
  const chips = CHIPS.filter((c) => (c.id !== "pending" && c.id !== "unchecked") || counts[c.id] > 0);
  return (
    <div className="worklist">
      <div className="chips" role="group" aria-label="Show">
        {chips.map((chip) => (
          <button
            key={chip.id}
            type="button"
            className={`chip chip-${chip.id}`}
            aria-pressed={filter === chip.id}
            onClick={() => onFilter(chip.id)}
          >
            <Icon name={chip.icon} />
            <span>
              {chip.label} ({counts[chip.id]})
            </span>
          </button>
        ))}
      </div>

      {items.length === 0 ? (
        <EmptyList filter={filter} counts={counts} running={running} onFilter={onFilter} />
      ) : (
        <table className="worklist-table">
          <caption className="visually-hidden">
            Labels, most urgent first. Choose a file name to open its result below.
          </caption>
          <thead>
            <tr>
              <th scope="col">Label</th>
              <th scope="col">Status</th>
              <th scope="col">First thing to look at</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => {
              const key = itemKey(item);
              const p = tierPresentation(item);
              const selected = key === selectedKey;
              const counts = describeCounts(item);
              return (
                <tr
                  key={key}
                  id={`wl-${cssId(key)}`}
                  className={`is-openable${selected ? " is-selected" : ""}`}
                  aria-current={selected ? "true" : undefined}
                  onClick={() => onOpen(key)}
                >
                  <td data-label="Label">
                    {/* Every row opens — a pending one shows its state now and
                        its result as soon as it has been checked. */}
                    <button
                      type="button"
                      className="linklike"
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpen(key);
                      }}
                    >
                      {item.filename}
                    </button>
                    <span className="brand">{item.brand_name}</span>
                    {selected && <span className="open-marker">Open below</span>}
                  </td>
                  <td data-label="Status" className="status-cell">
                    <span className={`badge tone-${p.tone}`}>
                      <Icon name={p.icon} />
                      <span>{p.label}</span>
                    </span>
                    {counts && <span className="counts">{counts}</span>}
                  </td>
                  <td data-label="First thing to look at">
                    {item.state === "error" ? (
                      <span className="item-error">{item.error ?? "This image could not be read."}</span>
                    ) : item.state === "skipped" ? (
                      <span className="unchecked-text">Not checked — check this label by hand or on its own</span>
                    ) : item.state === "running" ? (
                      <span className="muted">Being read now</span>
                    ) : item.state === "pending" ? (
                      <span className="muted">Not checked yet — waiting its turn</span>
                    ) : (
                      (item.headline ?? <span className="muted">Nothing flagged</span>)
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** Keys contain a NUL separator; make them safe for an element id. */
export function cssId(key: string): string {
  return key.replace(/[^A-Za-z0-9_-]/g, (c) => `_${c.charCodeAt(0).toString(16)}`);
}

function EmptyList({
  filter,
  counts,
  running,
  onFilter,
}: Pick<Props, "filter" | "counts" | "running" | "onFilter">) {
  const message = emptyMessage(filter, counts, running);
  return (
    <p className="empty">
      {message.text}
      {message.action && (
        <>
          {" "}
          <button
            type="button"
            className="button button-quiet button-small"
            onClick={() => onFilter(message.action!.filter)}
          >
            {message.action.label}
          </button>
        </>
      )}
    </p>
  );
}
