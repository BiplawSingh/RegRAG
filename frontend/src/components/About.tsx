// Every figure here is copied from the evaluation in the repo, never estimated.
// Each one's source is noted beside it so it can be re-checked:
//   retrieval   python eval.py retrieval      data/eval/retrieval_k8_filtered*.json
//   answers     python judge.py report        data/eval/judged_*_k8*.jsonl
//   citations   data/eval/answers_e2e_k8.jsonl
//   entity      python retrieval_lab.py entity   data/eval/entity_filter_k8.json
// If a re-run changes a number, it changes here too.
const REPO = 'https://github.com/BiplawSingh/RegRAG'

const STATS = [
  {
    value: '60% → 78%',
    label: 'chunk recall',
    detail:
      'How often the passage that answers the question is among the 8 retrieved (42 → 54 of 69). Adding keyword search and a cross-encoder reranker did this, with the same embeddings.',
  },
  {
    value: '65% → 69%',
    label: 'answers fully correct',
    detail:
      'End to end, graded against a hand-written answer key by an LLM judge that passed 8 of 8 positive and 8 of 8 negative controls first (48 of 69 now).',
  },
  {
    value: '95%',
    label: 'correct given the right passage',
    detail:
      'Feeding the known-correct passage directly, bypassing retrieval: 63 of 66. Most of the remaining errors happen before generation, in retrieval.',
  },
  {
    value: '95%',
    label: 'citations verified',
    detail:
      '195 of 205 quotes appear word for word in the passage they cite. But 22% of those also appear in sibling circulars, so their attribution cannot be confirmed.',
  },
  {
    value: '35%',
    label: 'wrong-institution passages without the filter',
    detail:
      "Across 36 questions that name an institution, even after reranking (43% with dense search alone). Filtering inside the search removes them: better ranking can't tell copies apart that differ by one noun.",
  },
  {
    value: '3 of 12',
    label: 'plausible but unanswerable questions refused',
    detail:
      'The weakest result. Off-topic questions are declined reliably; questions that sound regulatory but that the circulars don\'t answer are usually answered anyway.',
  },
]

export function About() {
  return (
    <section id="about" aria-labelledby="about-heading" className="scroll-mt-6 border-t border-slate-200 pt-10">
      <h2 id="about-heading" className="text-xl font-semibold tracking-tight">
        About this project
      </h2>
      <div className="mt-3 space-y-3 text-slate-600">
        <p>
          RegRAG answers questions over 506 Reserve Bank of India circulars, split into 9,278 passages. The hard part is
          that RBI reissues the same rule once for each class of institution, so 74% of the corpus sits in families of
          near-identical documents. An answer that quotes the Small Finance Banks copy of a rule to someone asking about
          Payments Banks looks right and is wrong.
        </p>
        <p>
          Every number below comes from a 90-question answer key written by hand <em>before</em> looking at any model
          output, so the key could not be shaped to fit the system.
        </p>
      </div>

      <dl className="mt-6 grid gap-3 sm:grid-cols-2">
        {STATS.map((s) => (
          <div key={s.label} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <dt className="text-sm text-slate-500">{s.label}</dt>
            <dd>
              <span className="block text-2xl font-semibold tracking-tight text-slate-900">{s.value}</span>
              <span className="mt-1 block text-sm text-slate-600">{s.detail}</span>
            </dd>
          </div>
        ))}
      </dl>

      <div className="mt-6 space-y-2 text-sm text-slate-600">
        <p>
          <strong className="font-medium text-slate-800">How it works:</strong> an entity filter inside the search,
          dense (BGE) and BM25 retrieval fused with reciprocal-rank fusion, a cross-encoder reranker, generation on Groq
          with strict JSON output, and a check that every quote actually appears in the passage it cites. The web app
          is FastAPI and React with TypeScript.
        </p>
        <p className="flex flex-wrap gap-x-4 gap-y-1">
          <a href={REPO} target="_blank" rel="noreferrer" className="font-medium text-indigo-700 hover:underline">
            Source code on GitHub ↗
          </a>
          <a href={`${REPO}#measured-results`} target="_blank" rel="noreferrer" className="font-medium text-indigo-700 hover:underline">
            Full evaluation ↗
          </a>
          <a href={`${REPO}/blob/main/DECISIONS.md`} target="_blank" rel="noreferrer" className="font-medium text-indigo-700 hover:underline">
            Design decisions ↗
          </a>
        </p>
        <p className="text-xs text-slate-500">
          A research demo, not legal advice. For anything that matters, read the circular itself on rbi.org.in.
        </p>
      </div>
    </section>
  )
}
