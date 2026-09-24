import type { AskResponse } from '../types'
import { verdict } from '../verification'
import { CitationMarkers } from './CitationMarkers'

type Props = {
  result: AskResponse
  activeCitation: number | null
  onSelectCitation: (id: number) => void
}

export function AnswerCard({ result, activeCitation, onSelectCitation }: Props) {
  const active = result.citations.find((c) => c.id === activeCitation)
  const cited = active ? result.sources[active.source - 1] : undefined
  const verified = result.citations.filter((c) => c.verified).length

  return (
    <section aria-labelledby="answer-heading" className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="answer-heading" className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Answer
        </h2>
        {result.citations.length > 0 && (
          <span className="text-xs text-slate-500">
            {verified} of {result.citations.length} citations verified
          </span>
        )}
      </div>

      {/* Why there is no answer: quota used up, or the model unreachable. */}
      {result.notice && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">{result.notice}</p>
      )}

      {result.insufficient_context && (
        <p className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
          <strong>The retrieved circulars don't answer this.</strong> The model was told to say so rather than guess.
        </p>
      )}

      {/* whitespace-pre-line keeps the model's line breaks, e.g. in a numbered list. */}
      {result.answer && <p className="whitespace-pre-line leading-relaxed">{result.answer}</p>}

      {result.citations.length > 0 && (
        <CitationMarkers citations={result.citations} active={activeCitation} onSelect={onSelectCitation} />
      )}

      {active && (
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
          <p className="text-slate-600">
            <span className="font-mono">[{active.id}]</span>{' '}
            cites passage {active.source}
            {cited && ` · ${cited.circular.number} · ${cited.circular.issue_date ?? 'undated'}`}
          </p>
          <blockquote className="mt-2 border-l-2 border-slate-300 pl-3 italic text-slate-800">{active.quote}</blockquote>
          <p className="mt-2 text-slate-600">{verdict(active).long}</p>
        </div>
      )}
    </section>
  )
}
