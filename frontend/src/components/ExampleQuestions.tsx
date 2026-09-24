// Each example shows off a different part of the pipeline. All four were run
// against the live API before being chosen, so their behaviour is observed,
// not hoped for.
const EXAMPLES = [
  { question: 'Who can be an Internal Ombudsman?', shows: 'Quotes shared by sibling circulars' },
  { question: 'What are the KYC requirements for Payments Banks?', shows: 'The entity filter' },
  { question: 'When is a loan account treated as out of order?', shows: 'Exact-phrase keyword search' },
  { question: 'What time do banks close for lunch?', shows: 'Declining when the circulars are silent' },
]

type Props = {
  onPick: (question: string) => void
  disabled: boolean
  compact: boolean // after the first answer: one line of small chips, no captions
}

export function ExampleQuestions({ onPick, disabled, compact }: Props) {
  if (compact) {
    // On a phone the four chips would wrap into four rows and push the answer
    // down, so there they scroll sideways in one row; from `sm` up they wrap.
    return (
      <div className="-mx-4 flex items-center gap-2 overflow-x-auto px-4 text-sm sm:mx-0 sm:flex-wrap sm:overflow-visible sm:px-0">
        <span className="shrink-0 text-slate-500">Try:</span>
        {EXAMPLES.map((e) => (
          <button
            key={e.question}
            type="button"
            disabled={disabled}
            onClick={() => onPick(e.question)}
            className="shrink-0 whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1.5 text-slate-700 hover:border-indigo-300 hover:text-indigo-700 disabled:opacity-50 sm:py-1"
          >
            {e.question}
          </button>
        ))}
      </div>
    )
  }

  return (
    <div>
      <p className="mb-2 text-sm text-slate-500">Or try one of these:</p>
      <div className="grid gap-2 sm:grid-cols-2">
        {EXAMPLES.map((e) => (
          <button
            key={e.question}
            type="button"
            disabled={disabled}
            onClick={() => onPick(e.question)}
            className="rounded-lg border border-slate-200 bg-white p-3 text-left shadow-sm transition hover:border-indigo-300 hover:shadow disabled:opacity-50"
          >
            <span className="block text-sm font-medium text-slate-800">{e.question}</span>
            <span className="mt-0.5 block text-xs text-slate-500">Shows: {e.shows}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
