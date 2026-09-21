import { useRef, useState, type DragEvent } from "react";
import { ACCEPTED_TYPES } from "../../lib/format";
import { formatBytes } from "../../lib/progress";
import { DownloadButton } from "../DownloadButton";
import { Icon } from "../Icon";

interface Props {
  manifest: File | null;
  images: File[];
  onDownloadTemplate: () => Promise<void>;
  /** Everything picked or dropped, CSVs and images mixed; the parent sorts it. */
  onFiles: (files: File[]) => void;
  onClearImages: () => void;
  onClearManifest: () => void;
  disabled: boolean;
}

/**
 * Step 1: choose the manifest and the label images.
 *
 * Three ordinary buttons do everything; the drop zone is a convenience for
 * those who use it (it is not keyboard accessible, and many of these users
 * will not think to drag). The template link and its one-sentence
 * explanation are always visible — the users are agents, not developers,
 * and "manifest" is not an everyday word.
 */
export function BatchPicker({ manifest, images, onDownloadTemplate, onFiles, onClearImages, onClearManifest, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const totalBytes = images.reduce((sum, f) => sum + f.size, 0);

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    onFiles(await filesFromDrop(e.dataTransfer));
  };

  return (
    <div className="batch-picker">
      <div className="manifest-explainer">
        <p>
          A <strong>manifest</strong> is a spreadsheet (saved as CSV) with one row per label: the image's file name
          and what the application says — brand name, class/type, alcohol content and net contents.
        </p>
        <p>
          <DownloadButton label="Download the manifest template" download={onDownloadTemplate} />{" "}
          <span className="muted">— fill it in with Excel or any spreadsheet program, then save as CSV.</span>
        </p>
      </div>

      <div
        className={`dropzone batch-drop${dragging ? " is-dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p className="dropzone-text">
          Drag the label images, a folder of them, and the manifest here — or use the buttons.
        </p>
        <div className="actions picker-buttons">
          <FileButton label="Choose images" accept={ACCEPTED_TYPES.join(",")} multiple onFiles={onFiles} disabled={disabled} />
          <FileButton label="Choose a folder" folder onFiles={onFiles} disabled={disabled} />
          <FileButton label="Choose manifest (.csv)" accept=".csv,text/csv" onFiles={onFiles} disabled={disabled} />
        </div>
      </div>

      <ul className="selection" aria-label="Your selection">
        <li>
          <Icon name={manifest ? "check" : "dash"} />
          <span>
            {manifest ? (
              <>
                Manifest: <strong>{manifest.name}</strong>
              </>
            ) : (
              "No manifest chosen yet"
            )}
          </span>
          {manifest && !disabled && (
            <button type="button" className="button button-quiet button-small" onClick={onClearManifest}>
              Remove<span className="visually-hidden"> manifest</span>
            </button>
          )}
        </li>
        <li>
          <Icon name={images.length ? "check" : "dash"} />
          <span>
            {images.length ? (
              <>
                <strong>{images.length}</strong> {images.length === 1 ? "file" : "files"} chosen (
                {formatBytes(totalBytes)})
              </>
            ) : (
              "No images chosen yet"
            )}
          </span>
          {images.length > 0 && !disabled && (
            <button type="button" className="button button-quiet button-small" onClick={onClearImages}>
              Remove<span className="visually-hidden"> all images</span>
            </button>
          )}
        </li>
      </ul>
    </div>
  );
}

/** A real button that opens a hidden file input — large, labelled, focusable. */
export function FileButton({
  label,
  accept,
  multiple,
  folder,
  onFiles,
  disabled,
  className = "button button-secondary",
}: {
  label: string;
  accept?: string;
  multiple?: boolean;
  folder?: boolean;
  onFiles: (files: File[]) => void;
  disabled?: boolean;
  className?: string;
}) {
  const input = useRef<HTMLInputElement | null>(null);
  return (
    <>
      <input
        ref={(el) => {
          input.current = el;
          // Not in React's typings; set as a plain attribute.
          if (el && folder) el.setAttribute("webkitdirectory", "");
        }}
        type="file"
        accept={accept}
        multiple={multiple || folder}
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        data-picker={label}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          if (files.length) onFiles(files);
        }}
      />
      <button type="button" className={className} disabled={disabled} onClick={() => input.current?.click()}>
        {label}
      </button>
    </>
  );
}

/** Dropped folders arrive as directory entries; walk them for their files. */
async function filesFromDrop(data: DataTransfer): Promise<File[]> {
  const entries = Array.from(data.items ?? [])
    .map((item) => (item.kind === "file" ? item.webkitGetAsEntry?.() : null))
    .filter((e): e is FileSystemEntry => !!e);
  if (entries.length === 0) return Array.from(data.files);
  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      out.push(await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej)));
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns at most ~100 at a time; call until empty.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (batch.length === 0) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  for (const entry of entries) await walk(entry);
  return out;
}
