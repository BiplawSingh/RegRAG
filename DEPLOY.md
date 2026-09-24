# Deploying the web demo

Two pieces, two free hosts:

| Piece | Host | What it serves |
|---|---|---|
| Frontend (`frontend/`) | **Vercel** | Static files: HTML, JS, CSS. No server code. |
| API (`api/` + the pipeline) | **Hugging Face Spaces**, Docker | FastAPI, both models, the corpus. Calls Groq. |

```
browser ──HTTPS──▶ Vercel (static React app)
   │
   └────HTTPS, CORS──▶ Hugging Face Space (FastAPI) ──▶ Groq API
```

The browser downloads the app from Vercel, then calls the Space directly. The
Groq key lives only in the Space's secrets. It is never in the frontend bundle,
the repo, or the browser.

---

## Why these hosts

The API needs about 1–1.5 GB of RAM: PyTorch, a 33M-parameter embedding model, a
22M-parameter cross-encoder, a BM25 index and 14 MB of vectors. That rules out
most free tiers before anything else is compared.

| Option | RAM | Idle behaviour | Cost | Verdict |
|---|---|---|---|---|
| **Hugging Face Spaces (Docker, CPU basic)** | 16 GB, 2 vCPU | sleeps after ~48 h without visitors; the next visit wakes it (a minute or two) | free | **chosen**: fits comfortably, runs any Dockerfile, and is where ML demos are expected to live |
| Render (free web service) | 512 MB | sleeps after 15 min idle; slow cold starts | free | too little memory for torch plus two models |
| Fly.io | configurable (2 GB is plenty) | can stay warm | paid: no free allowance for new accounts | good if a warm, always-fast demo is worth a few dollars a month |
| Qdrant Cloud (free cluster) | 1 GB, vectors only | suspended when unused for a while | free | **not needed**: see below |

**Why no Qdrant Cloud.** The deployed API searches `embeddings.npy` with a numpy
dot product. `retrieval_lab.py baseline` verifies that path returns exactly the
same results as Qdrant, question for question. The entity filter, the one thing
Qdrant was chosen for, is applied identically by `retrieval.entity_mask`. Adding
Qdrant Cloud would add a second service, a second secret, a network hop per
question, and a second thing that goes to sleep. At 9,278 vectors an exact search
takes milliseconds. Qdrant stays the store for local development and the eval.

**Why Vercel for the frontend.** It is a static site. Vercel builds it on every
push to `main`, serves it from a CDN, gives HTTPS, and costs nothing. Netlify or
Cloudflare Pages would do the same job.

Prices and free-tier limits change. Check the current terms before relying on
any figure above.

---

## Part 1: the API on Hugging Face Spaces

You need a Hugging Face account and a Groq API key (https://console.groq.com).

### 1. Create the Space

On https://huggingface.co/new-space:

- **Name:** `regrag-api`
- **SDK:** Docker → Blank
- **Hardware:** CPU basic (free)
- **Visibility: Public.** A private Space rejects requests without a Hugging Face
  login, so the browser could not call it. Everything the Space contains is
  either in the public GitHub repo already or is the RBI's own published text.

Its API address will be `https://<your-hf-username>-regrag-api.hf.space`.

### 2. Add the secret and variables

In the Space: **Settings → Variables and secrets.**

| Name | Kind | Value |
|---|---|---|
| `GROQ_API_KEY` | **Secret** | your Groq key |
| `ALLOWED_ORIGINS` | Variable | `http://localhost:5173` for now; the Vercel URL after Part 2 |
| `FORWARDED_HOPS` | Variable | `1`, then verify it in step 5 |
| `DEBUG_FORWARDING` | Variable | `1` until step 5 is done, then delete it |

Optional, all with defaults sized to the Groq free tier (see `.env.example`):
`RATE_IP_PER_MINUTE` (5), `RATE_IP_GENERATIONS_PER_DAY` (20),
`RATE_GENERATIONS_PER_MINUTE` (3), `RATE_GENERATIONS_PER_DAY` (60), `GROQ_MODEL`.

Use a **Secret**, not a Variable, for the key: variables are visible to anyone who
can see the Space.

### 3. Assemble and upload

From the repo root, with the corpus built (`data/chunks.jsonl` and
`data/embeddings.npy` exist):

```bash
python space/sync.py --target api          # copies pipeline + data into api_space/
python space/sync.py --target api --check  # confirms nothing has drifted

python -m huggingface_hub.cli.hf auth login   # paste a token with "write" access
python -m huggingface_hub.cli.hf upload <your-hf-username>/regrag-api api_space . \
    --repo-type space --exclude "**/__pycache__/**" \
    --commit-message "Deploy the RegRAG API"
```

`hf upload` handles the two large data files itself and avoids running
`git init` inside this repo. A git repo nested in another one confuses the
outer one.

### 4. Watch it build

The Space's **Logs** tab shows the Docker build (about 5–10 minutes the first
time, mostly PyTorch), then startup. It is ready when this returns
`"status":"ok"`:

```bash
curl https://<your-hf-username>-regrag-api.hf.space/health
```

`"status":"starting"` means the models are still loading. Wait a moment. The
container starts from a built image, so nothing is downloaded at this point.

### 5. Check the rate limiter sees real visitor addresses

Rate limits are per visitor IP, read from the `X-Forwarded-For` header that
Hugging Face's proxy adds. `FORWARDED_HOPS` says how many entries at the right of
that header were added by trusted proxies. It is worth checking once, not
guessing:

```bash
curl https://<your-hf-username>-regrag-api.hf.space/debug/forwarding \
     -H "X-Forwarded-For: 203.0.113.9"
```

- **`rate_limit_key` is your own public IP** (compare with https://ifconfig.me):
  correct. The fake `203.0.113.9` was ignored.
- **It is `203.0.113.9`:** the setting trusts too much, and anyone could dodge the
  limit with a fake header. Lower `FORWARDED_HOPS`.
- **It is an address that isn't yours:** that is a proxy's address, and every
  visitor would share one budget. Raise `FORWARDED_HOPS` by the number of
  entries between your IP and the right-hand end of `x_forwarded_for`.

Then **delete `DEBUG_FORWARDING`** so the endpoint switches off.

### 6. Ask a real question

```bash
curl -X POST https://<your-hf-username>-regrag-api.hf.space/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Who can be an Internal Ombudsman?"}'
```

The response should have `"generation": "ok"` and five or so citations.

---

## Part 2: the frontend on Vercel

### 1. Push the repo to GitHub

Vercel builds from GitHub. Before pushing, check the repo is the visibility you
expect and that no personal data or key is staged:

```bash
git diff origin/main --stat
git grep -nE "gsk_|hf_[A-Za-z0-9]{20}|@gmail" -- . ':!DEPLOY.md'   # expect nothing
git push
```

### 2. Import the project

On https://vercel.com/new, import the GitHub repo, then:

- **Root Directory:** `frontend` (the repo root is the Python project)
- **Framework Preset:** Vite (detected). Build command `npm run build`, output
  directory `dist`. Leave both as detected.
- **Environment Variables:** `VITE_API_URL` = `https://<your-hf-username>-regrag-api.hf.space`
  (no trailing slash)

Deploy. The site is at `https://<project>.vercel.app`.

`VITE_API_URL` is compiled into the JavaScript at build time. If you change it
later, redeploy for it to take effect. Anything prefixed `VITE_` ends up in the
public bundle, so never put a key in one.

### 3. Allow the frontend to call the API

Back in the Space: set `ALLOWED_ORIGINS` to the Vercel address, for example
`https://regrag.vercel.app`. That's the exact origin, with no trailing slash.
Keep `http://localhost:5173` after a comma if you still develop locally. The
Space restarts with the new value.

Only that address may call the API from a browser. Vercel's preview deployments
use other URLs, so they will show "Couldn't reach the server". That is CORS doing
its job, not a bug.

### 4. Try it

Open the Vercel URL and click an example question. If the Space was asleep, the
first question can take a minute while it wakes.

---

## All environment variables

| Where | Name | Required | Purpose |
|---|---|---|---|
| Space (secret) | `GROQ_API_KEY` | yes | generation. Without it the API serves retrieval only |
| Space | `ALLOWED_ORIGINS` | yes | comma-separated origins allowed to call the API from a browser |
| Space | `FORWARDED_HOPS` | yes | trusted proxies in front of the app. Verify with Part 1, step 5 |
| Space | `DEBUG_FORWARDING` | no | `1` enables `/debug/forwarding`. Remove after checking |
| Space | `RATE_*` | no | rate limits. Defaults fit the Groq free tier |
| Space | `GROQ_MODEL` | no | default `openai/gpt-oss-120b`. Must support strict JSON schema |
| Vercel | `VITE_API_URL` | yes | the Space's address, baked in at build time |

---

## Running it day to day

- **Updating the API.** Change code in the repo, then re-run step 3 of Part 1:
  `sync.py --target api`, then `hf upload`. The Space rebuilds.
- **Updating the frontend.** Push to `main`. Vercel rebuilds automatically.
- **The daily allowance.** Groq's free tier is about 200,000 tokens a day, roughly
  70 answers. The API stops generating at 60 a day (`RATE_GENERATIONS_PER_DAY`),
  keeping a margin. After that, visitors see the retrieved passages with a note
  explaining why, rather than an error. The count resets on a rolling 24 hours,
  and also when the Space restarts.
- **Sleeping.** After about 48 hours without visitors the Space sleeps. Visiting
  it wakes it. Before sending the link to someone, open it yourself first.
- **Logs.** The Space's Logs tab shows startup, errors and failed generations.
  The API does not log questions.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Couldn't reach the server" on the Vercel site | the Space is asleep or still building, or `ALLOWED_ORIGINS` doesn't exactly match the site's address. The browser console will say "CORS" for the latter |
| `/health` stuck on `"starting"` | model loading failed. Check the Logs tab for the error |
| Every answer is retrieval-only | `GROQ_API_KEY` missing or wrong (the note says "switched off"), or the day's allowance is used up (the note says so) |
| 429 for everyone at once | `FORWARDED_HOPS` too high or too low. Redo Part 1, step 5 |
