import { useRef, useState, type DragEvent, type ReactNode } from "react";
import { ACCEPTED_TYPES } from "../lib/format";

interface Props {
  hasImage: boolean;
  onFile: (file: File) => void;
  /** Rendered inside the drop area once an image is chosen. */
  children?: ReactNode;
  error: string | null;
}

/**
 * Image input: drag-and-drop for those who use it, and an ordinary, clearly
 * labelled button for everyone else. Drag-and-drop is never the only way —
 * many of these users will not think to try it, and it is not keyboard
 * accessible.
 */
export function ImagePicker({ hasImage, onFile, children, error }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={`dropzone${dragging ? " is-dragging" : ""}${hasImage ? " has-image" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {children}
      {!hasImage && (
        <p className="dropzone-text">
          Drag a photo or scan of the label here, or use the button below.
          <br />
          <span className="muted small">JPEG, PNG, WebP, TIFF or BMP, up to 20 MB.</span>
        </p>
      )}
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      <input
        ref={input}
        id="label-file"
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          // Reset so choosing the same file again still fires onChange.
          e.target.value = "";
        }}
      />
      <button type="button" className="button button-secondary" onClick={() => input.current?.click()}>
        {hasImage ? "Choose a different image" : "Choose label image"}
      </button>
    </div>
  );
}
