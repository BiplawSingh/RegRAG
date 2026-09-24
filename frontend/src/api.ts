import type { AskResponse } from './types'

// Set at build time from VITE_API_URL (see .env.example). Anything prefixed
// VITE_ is baked into the public JavaScript bundle, so it must never be a secret.
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// Long enough for a host waking from sleep; short enough that a dead server
// ends in a message rather than a spinner that never stops.
const TIMEOUT_MS = 90_000

// An error whose message is written for the visitor, not for a developer.
export class ApiError extends Error {}

export async function askQuestion(question: string): Promise<AskResponse> {
  let res: Response
  try {
    res = await fetch(`${API_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
  } catch (err) {
    // fetch only throws when no HTTP response arrived at all.
    if (err instanceof DOMException && err.name === 'TimeoutError') {
      throw new ApiError('The server took too long to answer. Please try again.')
    }
    throw new ApiError(
      "Couldn't reach the server. The free host sleeps when idle and can take a minute to wake up, so try again shortly.",
    )
  }

  if (res.ok) {
    return (await res.json()) as AskResponse
  }

  // Error bodies are {"detail": ...}. Read defensively: a proxy in front of the
  // server can send an HTML error page instead.
  const body = await res.json().catch(() => null)
  const detail = typeof body?.detail === 'string' ? body.detail : null

  if (res.status === 422) {
    throw new ApiError('Questions need to be between 3 and 500 characters.')
  }
  throw new ApiError(detail ?? `The server returned an error (${res.status}). Please try again.`)
}
