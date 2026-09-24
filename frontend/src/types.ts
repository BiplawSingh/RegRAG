// The API contract, mirrored by hand from api/schemas.py.
// If a field changes there, it changes here in the same commit.

export const MIN_QUESTION_CHARS = 3
export const MAX_QUESTION_CHARS = 500

export type Circular = {
  number: string
  ref_code: string | null
  title: string | null
  issue_date: string | null
  department: string | null
  applies_to: string[]
  source_url: string | null
}

// Each ranking stage's view of one passage.
export type Scores = {
  dense_rank: number | null // position in the dense top 50; null if absent
  bm25_rank: number | null // position in the BM25 top 50; null if absent
  rrf: number // fused score: what built the candidate pool
  rerank: number | null // cross-encoder score: what ordered the final list
}

export type Source = {
  rank: number // 1-based; what a citation's `source` points at
  chunk_id: number
  text: string
  section: string | null
  circular: Circular
  scores: Scores
}

export type Highlight = {
  source: number // which Source the span is in
  start: number // character offsets into that Source's text
  end: number
}

export type CitationStatus = 'ok' | 'wrong_source' | 'not_found'

export type Citation = {
  id: number
  source: number // the Source the model attributed the quote to
  quote: string
  status: CitationStatus
  verified: boolean
  found_in: number[] // every Source that contains the quote
  highlight: Highlight | null
}

export type AskResponse = {
  question: string
  answer: string | null
  insufficient_context: boolean
  generation: 'ok' | 'rate_limited' | 'unavailable'
  notice: string | null
  citations: Citation[]
  sources: Source[]
  retrieval: {
    entities: string[]
    filter_applied: boolean
    reranked: boolean
    candidates: number
    k: number
  }
  timings_ms: { retrieval: number; generation: number | null }
  model: string | null
}
