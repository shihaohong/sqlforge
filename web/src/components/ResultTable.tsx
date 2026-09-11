/**
 * The rows a query actually returned.
 *
 * This is the part that makes the demo more than a text generator: the SQL
 * ran against the real sqlite file and these are its rows.
 */

import type { TablePreview } from "../api";

function cell(value: unknown): string {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

export function ResultTable({ preview }: { preview: TablePreview }) {
  if (preview.error) {
    return (
      <div className="rounded-lg bg-surface-3/60 px-3 py-2.5">
        <div className="text-[11px] font-medium uppercase tracking-wider text-ink-muted">
          execution error
        </div>
        <code className="mt-1 block text-sm break-words text-ink-secondary">{preview.error}</code>
      </div>
    );
  }

  if (preview.row_count === 0) {
    return (
      <div className="rounded-lg bg-surface-3/60 px-3 py-2.5 text-sm text-ink-secondary">
        The query ran and returned no rows.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg bg-surface-3/60">
      <div className="max-h-64 overflow-auto">
        <table className="w-full border-collapse text-left text-sm">
          <thead className="sticky top-0 bg-surface-3">
            <tr>
              {preview.columns.map((column, i) => (
                <th
                  key={`${column}-${i}`}
                  className="whitespace-nowrap px-3 py-2 text-[11px] font-medium uppercase tracking-wider text-ink-muted"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, r) => (
              <tr key={r} className="border-t border-surface-1/60">
                {row.map((value, c) => (
                  <td
                    key={c}
                    className="whitespace-nowrap px-3 py-1.5 font-mono text-[13px] tabular-nums text-ink-secondary"
                  >
                    {cell(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-3 py-1.5 text-[11px] text-ink-muted">
        {preview.row_count} row{preview.row_count === 1 ? "" : "s"}
        {preview.truncated && " shown (truncated)"}
      </div>
    </div>
  );
}
