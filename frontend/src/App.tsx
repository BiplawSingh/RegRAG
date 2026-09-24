import { useState } from 'react'
import { ApiError, askQuestion } from './api'
import { AnswerCard } from './components/AnswerCard'
import { ErrorMessage } from './components/ErrorMessage'
import { LoadingState } from './components/LoadingState'
import { QuestionForm } from './components/QuestionForm'
import type { AskResponse } from './types'

// Every state the page can be in, one at a time. Each variant carries only the
// data that exists in that state, so there is no way to render an answer while
// loading, or an error alongside a result.
type Status =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: AskResponse }

export default function App() {
  const [question, setQuestion] = useState('')
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [activeCitation, setActiveCitation] = useState<number | null>(null)

  async function ask() {
    setStatus({ kind: 'loading' })
    setActiveCitation(null)
    try {
      const result = await askQuestion(question.trim())
      setStatus({ kind: 'done', result })
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.'
      setStatus({ kind: 'error', message })
    }
  }

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
        <QuestionForm value={question} onChange={setQuestion} onSubmit={ask} busy={status.kind === 'loading'} />

        {status.kind === 'loading' && <LoadingState />}
        {status.kind === 'error' && <ErrorMessage message={status.message} onRetry={ask} />}
        {status.kind === 'done' && (
          <AnswerCard
            result={status.result}
            activeCitation={activeCitation}
            // Clicking the open citation again closes it.
            onSelectCitation={(id) => setActiveCitation((current) => (current === id ? null : id))}
          />
        )}
      </main>
    </div>
  )
}
