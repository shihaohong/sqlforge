/**
 * One engine's answer: its SQL, its rows, and what it cost.
 *
 * Both engines get the identical panel so the comparison is like-for-like;
 * only the accent color and the labels differ. Identity is carried by the
 * accent bar plus the heading text, never by color alone.
 */

import type { TablePreview } from "../api";
import { ResultTable } from "./ResultTable";
import { StatTile, Verdict } from "./StatTile";

export interface EngineState {
  sql: string;
  streaming: boolean;
  valid: boolean | null;
  reason: string | null;
  latencyMs: number | null;
  usdPer1k: number | null;
  costNote: string;
  preview: TablePreview | null;
  matchesGold: boolean | null | undefined;
  error: string | null;
}

export const emptyEngineState: EngineState = {
  sql: "",
  streaming: false,
  valid: null,
  reason: null,
  latencyMs: null,
  usdPer1k: null,
  costNote: "",
  preview: null,
  matchesGold: undefined,
  error: null,
};

interface EnginePanelProps {
  title: string;
  subtitle: string;
  accent: string;
  state: EngineState;
  /** Shown instead of the body when this engine cannot run at all. */
  unavailable?: string | null;
  idle: boolean;
}

export function EnginePanel({
  title,
  subtitle,
  accent,
  state,
  unavailable,
  idle,
}: EnginePanelProps) {
  const pending = idle || (state.streaming && !state.sql);

  return (
    <section className="flex flex-col overflow-hidden rounded-xl border border-surface-3 bg-surface-2">
      <div className="h-1 w-full" style={{ backgroundColor: accent }} aria-hidden="true" />
      <div className="flex flex-col gap-4 p-4">
        <header>
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <p className="mt-0.5 text-xs text-ink-secondary">{subtitle}</p>
        </header>

        {unavailable ? (
          <p className="rounded-lg bg-surface-3/60 px-3 py-2.5 text-sm text-ink-secondary">
            {unavailable}
          </p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-2">
              <StatTile
                label="latency"
                value={state.latencyMs !== null ? `${Math.round(state.latencyMs)} ms` : "—"}
                detail="end to end, this query"
                pending={state.latencyMs === null}
              />
              <StatTile
                label="cost / 1k queries"
                value={state.usdPer1k !== null ? `$${state.usdPer1k.toFixed(4)}` : "—"}
                detail={state.costNote}
                pending={state.usdPer1k === null}
              />
            </div>

            <div>
              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-muted">
                generated sql
              </div>
              {/* Fixed height, not min-height: the two panels are read side
                  by side, so the rows below have to start at the same y even
                  when one engine writes a longer query. */}
              <pre className="h-24 overflow-auto rounded-lg bg-surface-3/60 px-3 py-2.5 text-[13px] leading-relaxed whitespace-pre-wrap text-ink">
                {state.sql || (pending ? "" : "—")}
                {state.streaming && (
                  <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-ink align-middle" />
                )}
              </pre>
              {state.valid === false && state.reason && (
                <p className="mt-1.5 text-xs" style={{ color: "var(--color-status-critical)" }}>
                  blocked by the guardrail: {state.reason}
                </p>
              )}
            </div>

            {state.error ? (
              <p className="text-sm" style={{ color: "var(--color-status-critical)" }}>
                {state.error}
              </p>
            ) : (
              state.preview && (
                <div className="flex flex-col gap-2">
                  <Verdict matches={state.matchesGold} />
                  <ResultTable preview={state.preview} />
                </div>
              )
            )}
          </>
        )}
      </div>
    </section>
  );
}
