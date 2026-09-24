// Circular dates arrive as "2026-01-14". Parsed as UTC and printed in UTC, so a
// visitor west of Greenwich doesn't see the day before.
const dateFormat = new Intl.DateTimeFormat('en-IN', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
})

export function formatDate(iso: string | null): string {
  if (!iso) return 'Undated'
  const d = new Date(`${iso}T00:00:00Z`)
  return Number.isNaN(d.getTime()) ? iso : dateFormat.format(d)
}

export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`
}

// "A", "A and B", "A, B and C"
export function listJoin(items: string[]): string {
  if (items.length <= 1) return items.join('')
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}
