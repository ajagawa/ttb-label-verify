/**
 * Query-string state. The batch id lives in the URL (`?batch=<id>`) so a
 * reload reattaches to a batch the server still holds. replaceState, not
 * pushState: this is state, not navigation, and Back should leave the tool.
 */
export function getParam(name: string, search: string = window.location.search): string | null {
  return new URLSearchParams(search).get(name);
}

export function setParam(name: string, value: string | null): void {
  const url = new URL(window.location.href);
  if (value == null) url.searchParams.delete(name);
  else url.searchParams.set(name, value);
  window.history.replaceState(window.history.state, "", url);
}
