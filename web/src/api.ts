/**
 * The gateway's contract, as the browser sees it.
 *
 * The local model is called through /v1/sql/stream - the same streaming
 * endpoint any client would use - so the page is exercising the production
 * path rather than a demo-only shortcut.
 */

export interface DatabaseInfo {
  db_id: string;
  schema: string;
  sample_questions: string[];
}

export interface SchemasResponse {
  databases: DatabaseInfo[];
  comparison_available: boolean;
}

interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
}

/** Terminal event of a /v1/sql/stream response. */
export interface StreamDone {
  event: "done";
  sql: string | null;
  valid: boolean;
  reason: string | null;
  rejected_sql: string | null;
  usage: Usage;
  latency_ms: number;
}

export interface TablePreview {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  error: string | null;
}

export interface ExecuteResponse {
  ok: boolean;
  preview: TablePreview;
  gold_sql: string | null;
  matches_gold: boolean | null;
  rejected: string | null;
}

export interface CompareResponse {
  model: string;
  sql: string | null;
  valid: boolean;
  reason: string | null;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  usd_per_query: number | null;
  usd_per_1k_queries: number | null;
}

const TOKEN_STORAGE_KEY = "sqlforge.token";

/**
 * The demo token, from ?token=... on first visit or from storage after.
 *
 * A token in a browser is not a secret from the person holding it - it exists
 * so that a link can be shared deliberately rather than crawled, and so the
 * rate limits have something to attach to. The GPU-spending endpoints are
 * metered regardless.
 */
export function readToken(): string {
  const fromUrl = new URLSearchParams(window.location.search).get("token");
  if (fromUrl) {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, fromUrl);
    // Drop it from the address bar so a screenshot does not leak it.
    const url = new URL(window.location.href);
    url.searchParams.delete("token");
    window.history.replaceState({}, "", url);
    return fromUrl;
  }
  return window.localStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function headers(token: string): HeadersInit {
  return token
    ? { "Content-Type": "application/json", "X-Demo-Token": token }
    : { "Content-Type": "application/json" };
}

async function failure(response: Response): Promise<ApiError> {
  let detail = response.statusText;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // A proxy error page is not JSON; the status text will do.
  }
  return new ApiError(response.status, detail);
}

export async function fetchSchemas(token: string): Promise<SchemasResponse> {
  const response = await fetch("/v1/demo/schemas", { headers: headers(token) });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function executeSql(
  token: string,
  body: { db_id: string; sql: string; question?: string },
): Promise<ExecuteResponse> {
  const response = await fetch("/v1/demo/execute", {
    method: "POST",
    headers: headers(token),
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function compare(
  token: string,
  body: { db_id: string; question: string },
): Promise<CompareResponse> {
  const response = await fetch("/v1/demo/compare", {
    method: "POST",
    headers: headers(token),
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/**
 * Stream SQL from the local model, calling `onDelta` per token.
 *
 * Server-sent events over fetch rather than EventSource, because EventSource
 * cannot issue a POST or set the token header.
 */
export async function streamSql(
  token: string,
  body: { db_id: string; question: string },
  onDelta: (text: string) => void,
): Promise<StreamDone> {
  const response = await fetch("/v1/sql/stream", {
    method: "POST",
    headers: headers(token),
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  if (!response.body) throw new ApiError(500, "the model server sent no response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done: StreamDone | null = null;

  while (true) {
    const { value, done: finished } = await reader.read();
    if (finished) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a chunk can split one, so
    // keep the trailing partial frame in the buffer.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = JSON.parse(line.slice("data:".length).trim());
      if (payload.event === "error") throw new ApiError(502, payload.reason);
      if (payload.event === "done") done = payload as StreamDone;
      else if (typeof payload.delta === "string") onDelta(payload.delta);
    }
  }

  if (!done) throw new ApiError(502, "the stream ended without a verdict");
  return done;
}
