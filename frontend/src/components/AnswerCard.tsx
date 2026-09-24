import type { AskResponse } from '../types'
import { CitationMarkers } from './CitationMarkers'

type Props = {
  result: AskResponse
  onSelectCitation: (id: number) => void
}

export function AnswerCard({ result, onSelectCitation }: Props) {
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
          <strong>The retrieved circulars don't answer this.</strong> The model is told to say so rather than guess. The
          passages it was given are under "How this answer was found".
        </p>
      )}

      {/* whitespace-pre-line keeps the model's line breaks, e.g. in a numbered list. */}
      {result.answer && <p className="whitespace-pre-line leading-relaxed">{result.answer}</p>}

      {result.citations.length > 0 && (
        <>
          <CitationMarkers citations={result.citations} onSelect={onSelectCitation} />
          <p className="text-xs text-slate-500">Open a citation to read the passage it quotes.</p>
        </>
      )}
    </section>
  )
}
