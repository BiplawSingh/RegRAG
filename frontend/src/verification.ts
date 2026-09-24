import type { Citation } from './types'

export type Tone = 'ok' | 'ambiguous' | 'warn' | 'bad'

export type Verdict = {
  tone: Tone
  icon: string
  short: string // a few words, for screen readers and labels
  long: string // one sentence a visitor can act on
}

// What a citation's verification result means, in words. Kept in one place so
// the marker, the panel and the retrieval table cannot describe the same result
// differently.
export function verdict(c: Citation): Verdict {
  if (c.status === 'ok') {
    const others = c.found_in.filter((n) => n !== c.source)
    if (others.length === 0) {
      return {
        tone: 'ok',
        icon: '✓',
        short: 'verified',
        long: 'The quote appears word for word in the passage it cites.',
      }
    }
    // The unfalsifiable case: near-identical circulars share this sentence, so
    // "verified" cannot confirm which one the model meant.
    return {
      tone: 'ambiguous',
      icon: '✓',
      short: 'verified, source ambiguous',
      long: `The quote appears in the cited passage, and also in ${plural(others)}. Near-identical circulars share this sentence, so which one the model meant can't be confirmed.`,
    }
  }
  if (c.status === 'wrong_source') {
    return {
      tone: 'warn',
      icon: '!',
      short: 'wrong source',
      long: `The quote is real, but it comes from ${plural(c.found_in)}, not the passage cited.`,
    }
  }
  return {
    tone: 'bad',
    icon: '✕',
    short: 'not found',
    long: 'The quote does not appear in any retrieved passage.',
  }
}

// Retrieved passages are referred to by plain number ("passage 3"), never "[3]":
// square brackets are reserved for citation markers, and citation [2] can cite
// passage 1, so reusing the notation would make the two read as the same thing.
function plural(ns: number[]): string {
  if (ns.length === 1) return `passage ${ns[0]}`
  return `passages ${ns.slice(0, -1).join(', ')} and ${ns[ns.length - 1]}`
}

// Tailwind classes per tone. Written out in full rather than assembled from
// pieces, because Tailwind only generates classes it can find verbatim in the code.
export const toneClasses: Record<Tone, string> = {
  ok: 'border-emerald-300 bg-emerald-50 text-emerald-800',
  ambiguous: 'border-sky-300 bg-sky-50 text-sky-800',
  warn: 'border-amber-300 bg-amber-50 text-amber-800',
  bad: 'border-rose-300 bg-rose-50 text-rose-800',
}
