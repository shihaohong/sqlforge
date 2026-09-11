/**
 * SQLForge demo: one question, two engines, side by side.
 *
 * Both columns get the same schema and the same question. The left one is a
 * QLoRA fine-tuned Llama-3.2 3B, quantized to 4-bit and served from a single
 * L4 GPU in this cluster; the right one is Claude Haiku 4.5 over the API. Each
 * answer is executed against the real sqlite database and checked against the
 * gold query, so "correct" means the rows matched, not that the SQL looked
 * plausible.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  ApiError,
  compare,
  executeSql,
  fetchSchemas,
  readToken,
  streamSql,
  type DatabaseInfo,
} from "./api";
import { EnginePanel, emptyEngineState, type EngineState } from "./components/EnginePanel";

/** Measured at M3/M4: 74.5 QPS sustained on one L4 at $0.85/GPU-hour. */
const LOCAL_USD_PER_1K = 0.0032;
/** Spider's canonical example database, and a friendlier first impression. */
const PREFERRED_DB = "concert_singer";

export default function App() {
  const [token] = useState(readToken);
  const [databases, setDatabases] = useState<DatabaseInfo[]>([]);
  const [comparisonAvailable, setComparisonAvailable] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [dbId, setDbId] = useState("");
  const [question, setQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [showSchema, setShowSchema] = useState(false);

  const [local, setLocal] = useState<EngineState>(emptyEngineState);
  const [api, setApi] = useState<EngineState>(emptyEngineState);
  const [idle, setIdle] = useState(true);

  useEffect(() => {
    fetchSchemas(token)
      .then((response) => {
        setDatabases(response.databases);
        setComparisonAvailable(response.comparison_available);
        setDbId((current) => {
          if (current) return current;
          const preferred = response.databases.find((d) => d.db_id === PREFERRED_DB);
          return preferred?.db_id ?? response.databases[0]?.db_id ?? "";
        });
      })
      .catch((error: ApiError) =>
        setLoadError(
          error.status === 401
            ? "This demo needs a token. Open the link you were given, including its ?token=… part."
            : `Could not reach the gateway: ${error.message}`,
        ),
      );
  }, [token]);

  const database = useMemo(
    () => databases.find((d) => d.db_id === dbId),
    [databases, dbId],
  );

  const run = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || !dbId || running) return;

    setRunning(true);
    setIdle(false);
    setLocal({ ...emptyEngineState, streaming: true });
    setApi({ ...emptyEngineState, streaming: comparisonAvailable });

    // The two engines run concurrently: the comparison is of latency as much
    // as of correctness, and serializing them would make the slower one look
    // like it was waiting its turn.
    const localRun = (async () => {
      try {
        let streamed = "";
        const done = await streamSql(token, { db_id: dbId, question: trimmed }, (delta) => {
          streamed += delta;
          setLocal((s) => ({ ...s, sql: streamed }));
        });
        setLocal((s) => ({
          ...s,
          streaming: false,
          sql: done.sql ?? done.rejected_sql ?? streamed,
          valid: done.valid,
          reason: done.reason,
          latencyMs: done.latency_ms,
          usdPer1k: LOCAL_USD_PER_1K,
          // Same units as the API column so the two tiles compare directly;
          // the local figure is GPU-hours divided by measured throughput
          // rather than a per-token price.
          costNote: `${done.usage.prompt_tokens} in / ${done.usage.completion_tokens} out tokens`,
        }));
        if (!done.sql) return;

        const executed = await executeSql(token, {
          db_id: dbId,
          sql: done.sql,
          question: trimmed,
        });
        setLocal((s) => ({
          ...s,
          preview: executed.preview,
          matchesGold: executed.matches_gold,
        }));
      } catch (error) {
        const message = error instanceof ApiError ? error.message : String(error);
        setLocal((s) => ({ ...s, streaming: false, error: message }));
      }
    })();

    const apiRun = (async () => {
      if (!comparisonAvailable) return;
      try {
        const result = await compare(token, { db_id: dbId, question: trimmed });
        setApi((s) => ({
          ...s,
          streaming: false,
          sql: result.sql ?? "—",
          valid: result.valid,
          reason: result.reason,
          latencyMs: result.latency_ms,
          usdPer1k: result.usd_per_1k_queries,
          costNote: `${result.input_tokens} in / ${result.output_tokens} out tokens`,
        }));
        if (!result.sql) return;

        const executed = await executeSql(token, {
          db_id: dbId,
          sql: result.sql,
          question: trimmed,
        });
        setApi((s) => ({
          ...s,
          preview: executed.preview,
          matchesGold: executed.matches_gold,
        }));
      } catch (error) {
        const message = error instanceof ApiError ? error.message : String(error);
        setApi((s) => ({ ...s, streaming: false, error: message }));
      }
    })();

    await Promise.all([localRun, apiRun]);
    setRunning(false);
  }, [comparisonAvailable, dbId, question, running, token]);

  const savings =
    local.usdPer1k && api.usdPer1k ? Math.round(api.usdPer1k / local.usdPer1k) : null;

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-5 py-8">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">SQLForge</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-ink-secondary">
          A QLoRA fine-tuned <strong className="font-medium text-ink">Llama-3.2 3B</strong>,
          quantized to 4-bit and self-hosted on one NVIDIA L4, against{" "}
          <strong className="font-medium text-ink">Claude Haiku 4.5</strong> over the API. Ask a
          real database a question; both answers are executed and checked against the benchmark's
          gold query. On the Spider dev set the fine-tune scores{" "}
          <strong className="font-medium text-ink">71.2%</strong> to Haiku's{" "}
          <strong className="font-medium text-ink">74.0%</strong>, at roughly 1/200th the cost per
          query.
        </p>
      </header>

      {loadError ? (
        <p className="rounded-xl border border-surface-3 bg-surface-2 px-4 py-3 text-sm text-ink-secondary">
          {loadError}
        </p>
      ) : (
        <>
          <section className="flex flex-col gap-3 rounded-xl border border-surface-3 bg-surface-2 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <label className="flex flex-col gap-1.5">
                <span className="text-[11px] font-medium uppercase tracking-wider text-ink-muted">
                  database
                </span>
                <select
                  value={dbId}
                  onChange={(event) => {
                    setDbId(event.target.value);
                    setQuestion("");
                  }}
                  className="rounded-lg border border-surface-3 bg-surface-3/60 px-3 py-2 text-sm text-ink outline-none focus:border-ink-muted"
                >
                  {databases.map((d) => (
                    <option key={d.db_id} value={d.db_id}>
                      {d.db_id}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-1 flex-col gap-1.5">
                <span className="text-[11px] font-medium uppercase tracking-wider text-ink-muted">
                  question
                </span>
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void run();
                  }}
                  placeholder="How many singers are there?"
                  className="w-full rounded-lg border border-surface-3 bg-surface-3/60 px-3 py-2 text-sm text-ink placeholder:text-ink-muted outline-none focus:border-ink-muted"
                />
              </label>

              <button
                onClick={() => void run()}
                disabled={running || !question.trim()}
                className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-surface-1 transition-colors disabled:bg-surface-3 disabled:text-ink-muted"
              >
                {running ? "running…" : "Ask both"}
              </button>
            </div>

            {database && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-ink-muted">try:</span>
                {database.sample_questions.map((sample) => (
                  <button
                    key={sample}
                    onClick={() => setQuestion(sample)}
                    className="rounded-full border border-surface-3 px-2.5 py-1 text-xs text-ink-secondary transition-colors hover:border-ink-muted hover:text-ink"
                  >
                    {sample}
                  </button>
                ))}
                <button
                  onClick={() => setShowSchema((s) => !s)}
                  className="ml-auto text-xs text-ink-muted underline decoration-dotted underline-offset-4 hover:text-ink-secondary"
                >
                  {showSchema ? "hide schema" : "show schema"}
                </button>
              </div>
            )}

            {showSchema && database && (
              <pre className="max-h-56 overflow-auto rounded-lg bg-surface-3/60 px-3 py-2.5 text-[12px] leading-relaxed text-ink-secondary">
                {database.schema}
              </pre>
            )}
          </section>

          <div className="grid gap-4 lg:grid-cols-2">
            <EnginePanel
              title="Fine-tuned Llama-3.2 3B"
              subtitle="QLoRA, GPTQ 4-bit, vLLM on one L4 in this cluster"
              accent="var(--color-series-local)"
              state={local}
              idle={idle}
            />
            <EnginePanel
              title="Claude Haiku 4.5"
              subtitle="Anthropic API, same prompt and schema"
              accent="var(--color-series-api)"
              state={api}
              idle={idle}
              unavailable={
                comparisonAvailable
                  ? null
                  : "The comparison is switched off on this deployment: no Anthropic API key is configured."
              }
            />
          </div>

          {savings && (
            <p className="text-center text-sm text-ink-secondary">
              On this query the self-hosted model cost{" "}
              <strong className="font-medium text-ink">{savings}× less</strong> per thousand
              queries.
            </p>
          )}

          <footer className="mt-auto flex flex-wrap gap-x-4 gap-y-1 border-t border-surface-3 pt-4 text-xs text-ink-muted">
            <span>
              Every query is parsed with sqlglot and must be a single read-only SELECT before it
              runs.
            </span>
            <a
              href="https://github.com/shihaohong/sqlforge"
              className="underline decoration-dotted underline-offset-4 hover:text-ink-secondary"
            >
              source and full benchmarks
            </a>
          </footer>
        </>
      )}
    </div>
  );
}
