import type { FormEvent, KeyboardEvent } from 'react'
import { MAX_QUESTION_CHARS, MIN_QUESTION_CHARS } from '../types'

type Props = {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  busy: boolean
}

// A "controlled" form: the text lives in the parent's state, not in the
// textarea. The parent owns it so other things (the example questions, later)
// can fill the box too.
export function QuestionForm({ value, onChange, onSubmit, busy }: Props) {
  const tooShort = value.trim().length < MIN_QUESTION_CHARS

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault() // stop the browser's default full-page form submission
    if (!tooShort && !busy) onSubmit()
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter asks; Shift+Enter adds a new line. isComposing guards typists using
    // an input method (e.g. Hindi, Japanese), where Enter confirms a character.
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      e.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-xl border border-slate-200 bg-white shadow-sm focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-100">
      <label htmlFor="question" className="sr-only">
        Your question
      </label>
      <textarea
        id="question"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        maxLength={MAX_QUESTION_CHARS}
        rows={2}
        placeholder="e.g. What are the KYC requirements for Payments Banks?"
        className="block w-full resize-none rounded-t-xl bg-transparent px-4 pt-3 text-base outline-none placeholder:text-slate-400"
      />
      <div className="flex items-center justify-between gap-3 px-4 pb-3">
        <span className="text-xs text-slate-400" aria-live="polite">
          {value.length}/{MAX_QUESTION_CHARS}
        </span>
        <button
          type="submit"
          disabled={busy || tooShort}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </div>
    </form>
  )
}
