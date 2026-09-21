import type { ApplicationForm } from "../api/types";

export type FormErrors = Partial<Record<keyof ApplicationForm, string>>;

interface FieldSpec {
  key: keyof ApplicationForm;
  label: string;
  help?: string;
  optional?: boolean;
  inputMode?: "text" | "decimal";
}

/**
 * The four fields an agent reads off the application, plus the optional label
 * width. Labels sit above inputs and there are no placeholders: grey
 * placeholder text is easily mistaken for an already-filled value, and it
 * vanishes while typing.
 */
const FIELDS: FieldSpec[] = [
  { key: "brand_name", label: "Brand name" },
  { key: "class_type", label: "Class/type designation" },
  { key: "alcohol_content", label: "Alcohol content" },
  { key: "net_contents", label: "Net contents" },
  {
    key: "label_width_mm",
    label: "Label width in millimetres",
    optional: true,
    inputMode: "decimal",
    help:
      "The printed width of the physical label. A photo has no sense of scale, so the " +
      "warning's minimum type-size check can only run when this is given; otherwise it is reported " +
      "as “couldn't check”.",
  },
];

export const SAMPLE_VALUES: Omit<ApplicationForm, "beverage_class" | "label_width_mm"> = {
  brand_name: "OLD TOM DISTILLERY",
  class_type: "Kentucky Straight Bourbon Whiskey",
  alcohol_content: "45% Alc./Vol. (90 Proof)",
  net_contents: "750 mL",
};

/** Returns an error message per missing/invalid field. Pure; used on submit. */
export function validateForm(form: ApplicationForm): FormErrors {
  const errors: FormErrors = {};
  for (const spec of FIELDS) {
    if (!spec.optional && !form[spec.key].trim()) errors[spec.key] = `Enter the ${spec.label.toLowerCase()}.`;
  }
  const width = form.label_width_mm.trim();
  if (width && !(Number(width) > 0)) errors.label_width_mm = "Enter a number greater than 0, or leave it blank.";
  return errors;
}

interface Props {
  form: ApplicationForm;
  errors: FormErrors;
  beverageClasses: Record<string, string>;
  onChange: (key: keyof ApplicationForm, value: string) => void;
}

export function RecordForm({ form, errors, beverageClasses, onChange }: Props) {
  const classes = Object.entries(beverageClasses);
  return (
    <fieldset className="record">
      <legend>What the application says</legend>
      {FIELDS.map((spec) => {
        const id = `f-${spec.key}`;
        const error = errors[spec.key];
        const describedBy = [spec.help ? `${id}-help` : "", error ? `${id}-error` : ""].filter(Boolean).join(" ");
        return (
          <div className="form-row" key={spec.key}>
            <label htmlFor={id}>
              {spec.label}
              {spec.optional && <span className="muted"> (optional)</span>}
            </label>
            {spec.help && (
              <p className="help" id={`${id}-help`}>
                {spec.help}
              </p>
            )}
            {error && (
              <p className="field-error" id={`${id}-error`}>
                {error}
              </p>
            )}
            <input
              id={id}
              name={spec.key}
              type="text"
              inputMode={spec.inputMode ?? "text"}
              autoComplete="off"
              spellCheck={false}
              value={form[spec.key]}
              aria-invalid={error ? true : undefined}
              aria-describedby={describedBy || undefined}
              onChange={(e) => onChange(spec.key, e.target.value)}
            />
          </div>
        );
      })}
      {/* Only one class exists today; a one-option dropdown is a pointless
          control for these users, so it is shown as plain text instead. */}
      {classes.length > 1 ? (
        <div className="form-row">
          <label htmlFor="f-beverage_class">Beverage type</label>
          <select
            id="f-beverage_class"
            value={form.beverage_class}
            onChange={(e) => onChange("beverage_class", e.target.value)}
          >
            {classes.map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <p className="muted">Beverage type: {classes[0]?.[1] ?? "Distilled Spirits"}</p>
      )}
    </fieldset>
  );
}
