import { useCallback, useState } from 'react'
import { ApiError, askQuestion } from './api'
import { AnswerCard } from './components/AnswerCard'
import { ErrorMessage } from './components/ErrorMessage'
import { ExampleQuestions } from './components/ExampleQuestions'
import { LoadingState } from './components/LoadingState'
import { QuestionForm } from './components/QuestionForm'
import { RetrievalDetails } from './components/RetrievalDetails'
import { SourcePanel, type PanelTarget } from './components/SourcePanel'
import type { AskResponse } from './types'

// Every state the page can be in, one at a time. Each variant carries only the
// data that exists in that state, so there is no way to render an answer while
// loading, or an error alongside a result.
type Status =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: AskResponse; id: number } // id: which request, for `key` below

export default function App() {
  const [question, setQuestion] = useState('')
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [panel, setPanel] = useState<PanelTarget | null>(null)

  // Takes the text as an argument rather than always reading `question`: an
  // example chip calls setQuestion and ask together, and a state update only
  // takes effect on the next render, so `question` would still hold the old text.
  async function ask(text: string = question) {
    setStatus({ kind: 'loading' })
    setPanel(null)
    try {
      const result = await askQuestion(text.trim())
      setStatus({ kind: 'done', result, id: Date.now() })
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.'
      setStatus({ kind: 'error', message })
    }
  }

  function askExample(text: string) {
    setQuestion(text)
    ask(text)
  }

  // The same function on every render, so the panel's Escape-key listener is
  // registered once rather than removed and re-added on each render.
  const closePanel = useCallback(() => setPanel(null), [])

  const busy = status.kind === 'loading'

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:py-16">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight">RegRAG</h1>
        <p className="mt-2 text-slate-600">
          Ask a question about Indian banking regulation. Answers are drawn from 506 RBI circulars, and every quote is
          checked against the source it cites.
        </p>
      </header>

      <main className="space-y-6">
        <QuestionForm value={question} onChange={setQuestion} onSubmit={() => ask()} busy={busy} />
        <ExampleQuestions onPick={askExample} disabled={busy} compact={status.kind !== 'idle'} />

        {status.kind === 'loading' && <LoadingState />}
        {status.kind === 'error' && <ErrorMessage message={status.message} onRetry={() => ask()} />}
        {status.kind === 'done' && (
          <>
            <AnswerCard result={status.result} onSelectCitation={(id) => setPanel({ kind: 'citation', id })} />
            <RetrievalDetails
              // A new `key` makes React build a fresh section for each answer, so
              // it starts open or closed for that answer instead of keeping
              // whatever the visitor did to the previous one.
              key={status.id}
              result={status.result}
              defaultOpen={status.result.generation !== 'ok'}
              onOpenPassage={(rank) => setPanel({ kind: 'passage', rank })}
              onOpenCitation={(id) => setPanel({ kind: 'citation', id })}
            />
            {panel && <SourcePanel result={status.result} target={panel} onClose={closePanel} />}
          </>
        )}
      </main>
    </div>
  )
}
