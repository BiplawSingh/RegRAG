type Props = {
  message: string
  onRetry: () => void
}

export function ErrorMessage({ message, onRetry }: Props) {
  return (
    <div role="alert" className="flex flex-col gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-800 sm:flex-row sm:items-center sm:justify-between">
      <p className="text-sm">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="shrink-0 self-start rounded-lg border border-rose-300 bg-white px-3 py-1.5 text-sm font-medium hover:bg-rose-100 sm:self-auto"
      >
        Try again
      </button>
    </div>
  )
}
