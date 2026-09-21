import { useEffect, useRef } from "react";
import type { BatchItem } from "../../api/types";
import { tierPresentation } from "../../lib/presentation";
import { itemKey, type Position } from "../../lib/worklist";
import { Icon } from "../Icon";
import { ResultDetail } from "../ResultDetail";

interface Props {
  /** The cached full item when fetched, otherwise the latest summary. */
  item: BatchItem;
  /** True while this item's full result is being fetched. */
  loading: boolean;
  loadError: string | null;
  onRetry: () => void;
  position: Position;
  filterLabel: string;
  imageUrl: string | null;
  implementedById: Record<string, boolean>;
  onGo: (key: string) => void;
  onBackToList: () => void;
  onReattach: () => void;
  /** Bumped by the parent when focus should move to this view's heading. */
  focusToken: number;
  /** ← / → only while batch mode is the one on screen. */
  keysEnabled: boolean;
}

/**
 * One label from the worklist, opened inline beneath the table — no modal,
 * no new page. Previous / Next walk the queue in the server's order, and the
 * ← / → keys do the same when focus is not in a text field.
 */
export function BatchDetail({
  item,
  loading,
  loadError,
  onRetry,
  position,
  filterLabel,
  imageUrl,
  implementedById,
  onGo,
  onBackToList,
  onReattach,
  focusToken,
  keysEnabled,
}: Props) {
  const heading = useRef<HTMLHeadingElement>(null);
  const p = tierPresentation(item);

  useEffect(() => {
    if (focusToken > 0) {
      heading.current?.focus();
      heading.current?.scrollIntoView({ block: "start" });
    }
  }, [focusToken]);

  // Pressing Next onto the last label disables Next while it has focus, which
  // drops focus to the page body — keyboard users would be lost. Catch that.
  const key = itemKey(item);
  useEffect(() => {
    const active = document.activeElement as HTMLButtonElement | null;
    if (!active || active === document.body || active.disabled) heading.current?.focus({ preventScroll: true });
  }, [key]);

  // Arrow keys walk the queue. Ignored while typing, when a modifier is held
  // (Alt+← is the browser's Back), and inside controls that use arrows.
  useEffect(() => {
    if (!keysEnabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.defaultPrevented) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (e.key === "ArrowLeft" && position.prev) {
        e.preventDefault();
        onGo(position.prev);
      } else if (e.key === "ArrowRight" && position.next) {
        e.preventDefault();
        onGo(position.next);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [position.prev, position.next, onGo, keysEnabled]);

  const where =
    position.index >= 0
      ? `${position.index + 1} of ${position.total} in “${filterLabel}”`
      : `not in “${filterLabel}”`;

  return (
    <section className="batch-detail" aria-labelledby="detail-heading">
      <div className="detail-nav">
        <button
          type="button"
          className="button button-secondary"
          disabled={!position.prev}
          onClick={() => position.prev && onGo(position.prev)}
        >
          ← Previous
        </button>
        <button
          type="button"
          className="button button-secondary"
          disabled={!position.next}
          onClick={() => position.next && onGo(position.next)}
        >
          Next →
        </button>
        <button type="button" className="button button-quiet" onClick={onBackToList}>
          Back to the list
        </button>
        <span className="muted small nav-hint">You can also use the ← and → keys.</span>
      </div>

      <h3 id="detail-heading" ref={heading} tabIndex={-1} className="detail-heading">
        {item.filename}
        <span className="detail-sub">
          {item.brand_name} · manifest row {item.row_number} · label {where}
        </span>
      </h3>
      <p>
        <span className={`badge tone-${p.tone}`}>
          <Icon name={p.icon} />
          <span>{p.label}</span>
        </span>
      </p>
      {!position.next && position.index >= 0 && (
        <p className="muted">This is the last label in “{filterLabel}”.</p>
      )}

      {item.state === "skipped" ? (
        <div className="notice notice-unchecked" role="status">
          <p className="notice-title">
            <Icon name="dash" /> Not checked
          </p>
          <p>
            The batch was stopped before this label was reached, so nothing on it was checked. Check it by hand, or
            check it on its own with “One label”.
          </p>
        </div>
      ) : item.state === "pending" || item.state === "running" ? (
        <div className="notice notice-info" role="status">
          <p className="notice-title">
            <Icon name={item.state === "running" ? "reading" : "clock"} />
            {item.state === "running" ? "Being read now" : "Waiting to be checked"}
          </p>
          <p>
            This label has not been checked yet. Its result will appear here as soon as it is ready — you can stay
            on this page.
          </p>
        </div>
      ) : item.state === "done" && !item.result ? (
        // Fixed-height placeholder: the page does not jump when the result
        // arrives, and Previous/Next stay exactly where the agent's eyes are.
        <div className="detail-loading" role="status">
          {loadError ? (
            <div className="notice notice-error">
              <p className="notice-title">
                <Icon name="cross" /> This label's result could not be loaded
              </p>
              <p>{loadError}</p>
              <button type="button" className="button button-secondary" onClick={onRetry}>
                Try again
              </button>
            </div>
          ) : (
            <p className={loading ? "loading-text" : "muted"}>Loading this label's result…</p>
          )}
        </div>
      ) : item.result ? (
        <ResultDetail
          result={item.result}
          imageUrl={imageUrl}
          fileName={item.filename}
          implementedById={implementedById}
          labelKey={itemKey(item)}
          onReattach={onReattach}
        />
      ) : (
        <div className="notice notice-review">
          <p className="notice-title">
            <Icon name="no-image" /> Nothing on this label was checked
          </p>
          <p>{item.error ?? "This image could not be read."}</p>
          <p>Check it by hand, or ask for a clearer photo and check it again on its own.</p>
        </div>
      )}
    </section>
  );
}
