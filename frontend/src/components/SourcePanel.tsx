import { useEffect, useRef } from 'react'
import { formatDate, listJoin } from '../format'
import type { AskResponse } from '../types'
import { toneClasses, verdict } from '../verification'

// What the panel is showing: a citation (its passage, with the quote
// highlighted), or a passage opened from the retrieval details.
export type PanelTarget = { kind: 'citation'; id: number } | { kind: 'passage'; rank: number }

type Props = {
  result: AskResponse
  target: PanelTarget
  onClose: () => void
}

export function SourcePanel({ result, target, onClose }: Props) {
  // A ref points at a real element on the page, for the few things React
  // doesn't do for you: moving keyboard focus, scrolling to an element.
  const closeButton = useRef<HTMLButtonElement>(null)
  const mark = useRef<HTMLElement>(null)

  const citation = target.kind === 'citation' ? result.citations.find((c) => c.id === target.id) : undefined
  // Show the passage where the quote really is: for a misattributed quote
  // that's not the one the model cited.
  const rank = target.kind === 'passage' ? target.rank : (citation?.highlight?.source ?? citation?.source ?? 0)
  const passage = result.sources[rank - 1]
  const span = citation?.highlight?.source === rank ? citation.highlight : null

  // Runs once when the panel opens; the function it returns runs when it closes.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    closeButton.current?.focus({ preventScroll: true })
    mark.current?.scrollIntoView({ block: 'center' })
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden' // the page behind stays put
    return () => {
      document.body.style.overflow = overflow
      opener?.focus() // keyboard users land back on the marker they clicked
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const c = passage?.circular

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-slate-900/40" onClick={onClose} aria-hidden="true" />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="panel-title"
        className="relative flex h-full w-full flex-col bg-white shadow-2xl sm:max-w-lg"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 p-5">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              {citation ? `Citation [${citation.id}] · passage ${rank}` : `Passage ${rank}`}
            </p>
            <h2 id="panel-title" className="mt-1 font-semibold leading-snug">
              {c?.title ?? 'Passage not in the retrieved set'}
            </h2>
          </div>
          <button
            ref={closeButton}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-m-1 rounded-md p-1 text-xl leading-none text-slate-500 hover:bg-slate-100 hover:text-slate-800"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto p-5">
          {citation && (
            <div className={`rounded-lg border p-3 text-sm ${toneClasses[verdict(citation).tone]}`}>
              <p>
                <span aria-hidden="true">{verdict(citation).icon} </span>
                {verdict(citation).long}
              </p>
              {/* With no highlight, the quote can't be seen in the passage, so show it here. */}
              {!span && (
                <blockquote className="mt-2 border-l-2 border-current pl-3 italic opacity-90">{citation.quote}</blockquote>
              )}
            </div>
          )}

          {passage && c ? (
            <>
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
                <dt className="text-slate-500">Circular</dt>
                <dd className="font-mono text-slate-800">{c.number}</dd>
                {c.ref_code && (
                  <>
                    <dt className="text-slate-500">Reference</dt>
                    <dd className="break-all font-mono text-slate-800">{c.ref_code}</dd>
                  </>
                )}
                <dt className="text-slate-500">Issued</dt>
                <dd className="text-slate-800">{formatDate(c.issue_date)}</dd>
                <dt className="text-slate-500">Binds</dt>
                <dd className="text-slate-800">
                  {c.applies_to.length ? listJoin(c.applies_to) : 'No particular class of institution'}
                </dd>
                {passage.section && (
                  <>
                    <dt className="text-slate-500">Section</dt>
                    <dd className="text-slate-800">{passage.section}</dd>
                  </>
                )}
              </dl>

              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Passage text</h3>
                <p className="whitespace-pre-line text-sm leading-relaxed text-slate-800">
                  {span ? (
                    <>
                      {passage.text.slice(0, span.start)}
                      <mark ref={mark} className="rounded bg-yellow-200 px-0.5 text-slate-900">
                        {passage.text.slice(span.start, span.end)}
                      </mark>
                      {passage.text.slice(span.end)}
                    </>
                  ) : (
                    passage.text
                  )}
                </p>
              </div>

              {c.source_url && (
                <a
                  href={c.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-block text-sm font-medium text-indigo-700 hover:underline"
                >
                  Read the full circular on rbi.org.in ↗
                </a>
              )}
            </>
          ) : (
            <p className="text-sm text-slate-600">
              The model cited passage {citation?.source}, but only {result.sources.length} passages were retrieved.
            </p>
          )}
        </div>
      </aside>
    </div>
  )
}
