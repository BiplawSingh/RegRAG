import type { ReactNode } from 'react'
import { formatDate, formatSeconds, listJoin } from '../format'
import type { AskResponse, Scores } from '../types'
import { toneClasses, verdict } from '../verification'

type Props = {
  result: AskResponse
  defaultOpen: boolean
  onOpenPassage: (rank: number) => void
  onOpenCitation: (id: number) => void
}

// Which of the two searches surfaced a passage. A passage only one search
// found is the whole argument for running two.
function channel(s: Scores): { label: string; className: string } {
  if (s.dense_rank && s.bm25_rank) return { label: 'found by both searches', className: 'bg-slate-100 text-slate-600' }
  if (s.dense_rank) return { label: 'meaning search only', className: 'bg-violet-100 text-violet-800' }
  return { label: 'keyword search only', className: 'bg-teal-100 text-teal-800' }
}

const rank = (n: number | null) => (n ? `#${n}` : '—')

// With no filter, an unscoped question tends to pull in one copy of the same rule
// per class of institution. Say how many classes the passages span, a fact read
// off the data, rather than asserting the copies are identical.
function classNote(sources: AskResponse['sources']): string {
  const classes = new Set(sources.flatMap((s) => s.circular.applies_to))
  if (classes.size < 3) return ''
  return ` The ${sources.length} passages below come from circulars binding ${classes.size} different classes of institution: RBI reissues each rule once per class, so an unscoped question often retrieves several near-identical copies.`
}

// <details> is the browser's built-in collapsible section: it opens and closes
// itself, is keyboard-accessible, and needs no React state.
export function RetrievalDetails({ result, defaultOpen, onOpenPassage, onOpenCitation }: Props) {
  const { retrieval, sources, citations, timings_ms } = result
  const oneSearchOnly = sources.filter((s) => !s.scores.dense_rank || !s.scores.bm25_rank).length

  const counts = {
    ok: citations.filter((c) => verdict(c).tone === 'ok').length,
    ambiguous: citations.filter((c) => verdict(c).tone === 'ambiguous').length,
    warn: citations.filter((c) => c.status === 'wrong_source').length,
    bad: citations.filter((c) => c.status === 'not_found').length,
  }

  return (
    <details open={defaultOpen} className="group rounded-xl border border-slate-200 bg-white shadow-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 rounded-xl p-5 hover:bg-slate-50 [&::-webkit-details-marker]:hidden">
        <div>
          <h2 className="font-semibold">How this answer was found</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            {sources.length} passages chosen from {retrieval.candidates * 2} candidates
            {citations.length > 0 && ` · ${citations.filter((c) => c.verified).length} of ${citations.length} citations verified`}
          </p>
        </div>
        <span aria-hidden="true" className="text-slate-400 transition group-open:rotate-180">
          ▾
        </span>
      </summary>

      <div className="space-y-8 border-t border-slate-200 p-5">
        <ol className="space-y-3 text-sm text-slate-700">
          <Step n={1} title="Institution detection">
            {retrieval.entities.length
              ? `The question names ${listJoin(retrieval.entities)}, so the search only covers circulars that bind ${retrieval.entities.length === 1 ? 'that class' : 'those classes'}, plus circulars that bind no particular class. RBI reissues each rule once per class of institution, so without this filter a third or more of the results would be another institution's copy.`
              : `The question names no class of institution, so the whole corpus is searched.${classNote(sources)}`}
          </Step>
          <Step n={2} title="Two searches">
            A meaning-based search (dense embeddings) and a keyword search (BM25) each return their top{' '}
            {retrieval.candidates}. {oneSearchOnly} of the {sources.length} passages below were found by only one of
            them.
          </Step>
          <Step n={3} title="Fusion">
            Reciprocal-rank fusion merges the two lists by position alone, because the two searches score on unrelated
            scales.
          </Step>
          <Step n={4} title="Reranking">
            {retrieval.reranked
              ? `A cross-encoder reads the question and each candidate together and keeps the best ${retrieval.k}.`
              : `The fused list is cut to the top ${retrieval.k}.`}{' '}
            <Time ms={timings_ms.retrieval} label="retrieval" />
          </Step>
          <Step n={5} title="Answer and verification">
            {result.generation === 'ok' ? (
              <>
                {result.model} writes the answer from these passages, quoting its evidence, and each quote is checked
                against the passage it cites. {timings_ms.generation !== null && <Time ms={timings_ms.generation} label="generation" />}
              </>
            ) : (
              'Skipped this time; see the note above the answer.'
            )}
          </Step>
        </ol>

        {citations.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Citation checks</h3>
            <div className="flex flex-wrap gap-2 text-sm">
              <Count n={counts.ok} tone="ok" label="verified" />
              <Count n={counts.ambiguous} tone="ambiguous" label="verified, but shared with sibling passages" />
              <Count n={counts.warn} tone="warn" label="attributed to the wrong passage" />
              <Count n={counts.bad} tone="bad" label="not found in any passage" />
            </div>
          </div>
        )}

        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Retrieved passages</h3>
          <dl className="mb-4 grid gap-x-4 gap-y-1 text-xs text-slate-500 sm:grid-cols-2">
            <div><dt className="inline font-medium text-slate-700">Meaning</dt> <dd className="inline">rank in the dense search's top {retrieval.candidates}</dd></div>
            <div><dt className="inline font-medium text-slate-700">Keyword</dt> <dd className="inline">rank in the BM25 search's top {retrieval.candidates}</dd></div>
            <div><dt className="inline font-medium text-slate-700">Fused</dt> <dd className="inline">RRF score, which chose the candidates</dd></div>
            <div><dt className="inline font-medium text-slate-700">Rerank</dt> <dd className="inline">cross-encoder score, which set this order</dd></div>
          </dl>

          <ol className="space-y-3">
            {sources.map((s) => {
              const ch = channel(s.scores)
              const citedBy = citations.filter((c) => c.source === s.rank)
              return (
                <li key={s.rank} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                    <button
                      type="button"
                      onClick={() => onOpenPassage(s.rank)}
                      className="text-left text-sm font-medium text-indigo-700 hover:underline"
                    >
                      Passage {s.rank} · <span className="font-mono">{s.circular.number}</span>
                    </button>
                    <span className="text-xs text-slate-500">{formatDate(s.circular.issue_date)}</span>
                  </div>
                  <p className="mt-1 line-clamp-1 text-xs text-slate-500">{s.circular.title}</p>
                  <p className="mt-2 line-clamp-2 text-sm text-slate-700">{s.text}</p>

                  <dl className="mt-3 grid grid-cols-4 gap-2 text-center text-xs">
                    <Score label="Meaning" value={rank(s.scores.dense_rank)} />
                    <Score label="Keyword" value={rank(s.scores.bm25_rank)} />
                    <Score label="Fused" value={s.scores.rrf.toFixed(4)} />
                    <Score label="Rerank" value={s.scores.rerank?.toFixed(2) ?? '—'} />
                  </dl>

                  <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                    <span className={`rounded-full px-2 py-0.5 ${ch.className}`}>{ch.label}</span>
                    {citedBy.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => onOpenCitation(c.id)}
                        aria-label={`Open citation ${c.id}: ${verdict(c).short}`}
                        className={`rounded-md border px-1.5 font-mono ${toneClasses[verdict(c).tone]}`}
                      >
                        [{c.id}] {verdict(c).icon}
                      </button>
                    ))}
                  </div>
                </li>
              )
            })}
          </ol>
        </div>
      </div>
    </details>
  )
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-semibold text-indigo-700">
        {n}
      </span>
      <p>
        <strong className="font-medium text-slate-900">{title}.</strong> {children}
      </p>
    </li>
  )
}

function Time({ ms, label }: { ms: number; label: string }) {
  return (
    <span className="whitespace-nowrap text-slate-500">
      ({formatSeconds(ms)} {label})
    </span>
  )
}

function Count({ n, tone, label }: { n: number; tone: keyof typeof toneClasses; label: string }) {
  if (n === 0) return null
  return (
    <span className={`rounded-md border px-2 py-0.5 ${toneClasses[tone]}`}>
      {n} {label}
    </span>
  )
}

function Score({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-slate-50 py-1.5">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-mono text-slate-800">{value}</dd>
    </div>
  )
}
