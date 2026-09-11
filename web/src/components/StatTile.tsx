/**
 * A single headline number.
 *
 * Latency, cost and correctness are one value each, so they are stat tiles
 * rather than charts - there is no distribution to show and nothing to hover.
 * Values wear text colors; the engine's identity is carried by the accent bar
 * on the panel, not by coloring the number.
 */

interface StatTileProps {
  label: string;
  value: string;
  /** Smaller print under the value: units, a comparison, a caveat. */
  detail?: string;
  pending?: boolean;
}

export function StatTile({ label, value, detail, pending }: StatTileProps) {
  return (
    <div className="rounded-lg bg-surface-3/60 px-3 py-2.5">
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-muted">
        {label}
      </div>
      <div
        className={`mt-1 font-mono text-xl leading-none tabular-nums ${
          pending ? "text-ink-muted" : "text-ink"
        }`}
      >
        {pending ? "—" : value}
      </div>
      {detail && !pending && (
        <div className="mt-1 text-[11px] leading-tight text-ink-secondary">{detail}</div>
      )}
    </div>
  );
}

/**
 * The correctness verdict.
 *
 * Status color never carries the meaning alone: every state ships with an
 * icon and a word, so it survives colorblindness, greyscale printing and
 * forced-colors mode.
 */
export function Verdict({
  matches,
  pending,
}: {
  matches: boolean | null | undefined;
  pending?: boolean;
}) {
  if (pending) {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-ink-muted">
        <span aria-hidden="true">○</span> checking
      </span>
    );
  }
  if (matches === null || matches === undefined) {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-ink-muted">
        <span aria-hidden="true">–</span> no gold query for this question
      </span>
    );
  }
  return matches ? (
    <span
      className="inline-flex items-center gap-1.5 text-sm font-medium"
      style={{ color: "var(--color-status-good)" }}
    >
      <span aria-hidden="true">✓</span> rows match the gold query
    </span>
  ) : (
    <span
      className="inline-flex items-center gap-1.5 text-sm font-medium"
      style={{ color: "var(--color-status-critical)" }}
    >
      <span aria-hidden="true">✗</span> rows differ from the gold query
    </span>
  );
}
