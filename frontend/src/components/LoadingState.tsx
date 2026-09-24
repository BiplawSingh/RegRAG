// Shown while a question is in flight. The grey bars hint at the shape of the
// answer to come; role="status" makes screen readers announce the text.
export function LoadingState() {
  return (
    <div role="status" className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm text-slate-500">
        Searching 9,278 passages, then writing an answer. This usually takes 3 to 6 seconds.
      </p>
      <div className="mt-4 space-y-2 animate-pulse" aria-hidden="true">
        <div className="h-3 w-full rounded bg-slate-200" />
        <div className="h-3 w-11/12 rounded bg-slate-200" />
        <div className="h-3 w-4/5 rounded bg-slate-200" />
      </div>
    </div>
  )
}
