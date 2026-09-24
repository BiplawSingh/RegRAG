import type { Citation } from '../types'
import { toneClasses, verdict } from '../verification'

type Props = {
  citations: Citation[]
  active: number | null
  onSelect: (id: number) => void
}

// One clickable [n] per citation, coloured by its verification result. They sit
// under the answer rather than inside it: the model returns citations as a list,
// not as positions in the text (DECISIONS.md #1).
export function CitationMarkers({ citations, active, onSelect }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-slate-500">Citations</span>
      {citations.map((c) => {
        const v = verdict(c)
        return (
          <button
            key={c.id}
            type="button"
            onClick={() => onSelect(c.id)}
            aria-pressed={active === c.id}
            aria-label={`Citation ${c.id}: ${v.short}`}
            title={v.long}
            className={`rounded-md border px-2 py-0.5 font-mono text-sm transition hover:brightness-95 aria-pressed:ring-2 aria-pressed:ring-indigo-400 ${toneClasses[v.tone]}`}
          >
            [{c.id}] <span aria-hidden="true">{v.icon}</span>
          </button>
        )
      })}
    </div>
  )
}
