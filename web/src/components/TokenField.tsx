/** Shown only after a 401: a small, plain field for the shared access code. */
export function TokenField({ id, token, onChange }: { id: string; token: string; onChange: (t: string) => void }) {
  return (
    <div className="form-row token-row">
      <label htmlFor={id}>Access code</label>
      <input id={id} type="password" autoComplete="off" value={token} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
