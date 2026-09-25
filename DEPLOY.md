# Deploying the web demo

Two pieces, two hosts:

| Piece | Host | What it serves |
|---|---|---|
| Frontend (`frontend/`) | **Vercel** | Static files: HTML, JS, CSS. No server code. |
| API (`api/` + the pipeline) | **Google Cloud Run** | FastAPI, both models, the corpus. Calls Groq. |

```
browser ──HTTPS──▶ Vercel (static React app)
   │
   └────HTTPS, CORS──▶ Cloud Run (FastAPI container) ──▶ Groq API
```

The browser downloads the app from Vercel, then calls Cloud Run directly. The
Groq key lives only in Google Secret Manager. It is never in the frontend
bundle, the repo, the image, or the browser.

---

## Why these hosts

The API needs about 1–1.5 GB of RAM: PyTorch, a 33M-parameter embedding model, a
22M-parameter cross-encoder, a BM25 index and 14 MB of vectors. That rules out
most free tiers before anything else is compared.

| Option | RAM | Idle behaviour | Cost | Verdict |
|---|---|---|---|---|
| **Google Cloud Run** | configurable; 2 GiB used | scales to zero when idle; the next request starts a container (models load from the image, ~10–30 s) | billed per second of use, with a monthly free allowance; needs a billing account | **chosen**: runs the existing Dockerfile unchanged, costs nothing while nobody is visiting |
| Hugging Face Spaces (Docker) | 16 GB, 2 vCPU | sleeps after ~48 h | Docker and Gradio Spaces now need a PRO subscription | was the first choice; still the least work if a subscription is fine |
| Render (free web service) | 512 MB | sleeps after 15 min idle | free | too little memory for torch plus two models |
| Fly.io | configurable | can stay warm | paid | good if an always-warm demo is worth a few dollars a month |
| Qdrant Cloud (free cluster) | 1 GB, vectors only | suspended when unused | free | **not needed**: see below |

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

## Part 1: the API on Cloud Run

You need a Google Cloud project with billing enabled (free-trial credits count),
a Groq API key (https://console.groq.com), and the gcloud CLI
(https://cloud.google.com/sdk/docs/install). The commands below are for Git Bash
or any POSIX shell.

### 1. Point gcloud at the project

```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud config set run/region asia-south1          # Mumbai; pick one near your visitors

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com
```

### 2. Store the Groq key in Secret Manager

```bash
read -rsp "Groq key: " KEY && printf %s "$KEY" \
    | gcloud secrets create groq-api-key --data-file=- && unset KEY

# Let Cloud Run's runtime identity (the default compute service account) read it.
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" \
    --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding groq-api-key \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
```

`read -s` keeps the key off the screen and out of shell history. To rotate it
later: `gcloud secrets versions add groq-api-key --data-file=-`, then redeploy.

### 3. Assemble and deploy

From the repo root, with the corpus built (`data/chunks.jsonl` and
`data/embeddings.npy` exist):

```bash
python space/sync.py --target api          # copies pipeline + data into api_space/
python space/sync.py --target api --check  # confirms nothing has drifted

gcloud run deploy regrag-api --source api_space \
    --memory 2Gi --cpu 1 --cpu-boost \
    --min-instances 0 --max-instances 1 \
    --allow-unauthenticated \
    --set-secrets GROQ_API_KEY=groq-api-key:latest \
    --set-env-vars ALLOWED_ORIGINS=http://localhost:5173,FORWARDED_HOPS=1,DEBUG_FORWARDING=1
```

What the flags are for:

- `--source api_space` uploads the folder; Cloud Build builds the Dockerfile in
  it (about 5–10 minutes the first time, mostly PyTorch) and stores the image in
  Artifact Registry. Accept when it offers to create a repository.
- `--max-instances 1` is required, not a tuning choice. The rate limits live in
  memory (DECISIONS.md #11), so a second instance would get a second copy of
  every budget, including the global Groq cap. It also caps the bill.
- `--min-instances 0` lets it scale to zero, so idle time costs nothing. The price
  is a cold start of roughly 10–30 seconds on the first question after a quiet
  spell. `--cpu-boost` shortens it.
- `--allow-unauthenticated` makes it public. Without it every browser request is
  rejected with 403.

`--set-env-vars` splits on commas. To give `ALLOWED_ORIGINS` several origins,
change the delimiter: `--update-env-vars "^;^ALLOWED_ORIGINS=https://a.app,http://localhost:5173"`.

The deploy prints the service URL, something like
`https://regrag-api-xxxxxxxxxx.asia-south1.run.app`. Retrieve it any time with:

```bash
gcloud run services describe regrag-api --format='value(status.url)'
```

It is ready when this returns `"status":"ok"`:

```bash
curl https://<service-url>/health
```

The container only accepts traffic once the models are loaded, so a slow first
response is the cold start, not an error.

### 4. Check the rate limiter sees real visitor addresses

Rate limits are per visitor IP, read from the `X-Forwarded-For` header that
Google's front end adds. `FORWARDED_HOPS` says how many entries at the right of
that header were added by trusted proxies. It is worth checking once, not
guessing:

```bash
curl https://<service-url>/debug/forwarding -H "X-Forwarded-For: 203.0.113.9"
```

- **`rate_limit_key` is your own public IP** (compare with https://ifconfig.me):
  correct. The fake `203.0.113.9` was ignored.
- **It is `203.0.113.9`:** the setting trusts too much, and anyone could dodge the
  limit with a fake header. Lower `FORWARDED_HOPS`.
- **It is an address that isn't yours:** that is a proxy's address, and every
  visitor would share one budget. Raise `FORWARDED_HOPS` by the number of
  entries between your IP and the right-hand end of `x_forwarded_for`.

Then switch the endpoint off:

```bash
gcloud run services update regrag-api --remove-env-vars DEBUG_FORWARDING
```

### 5. Ask a real question

```bash
curl -X POST https://<service-url>/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Who can be an Internal Ombudsman?"}'
```

The response should have `"generation": "ok"` and five or so citations.

### 6. Set a budget alert

In the Cloud console: **Billing → Budgets & alerts → Create budget**, scoped to
this project, with email alerts at 50%, 90% and 100%. A budget only alerts; it
does not stop spending. `--max-instances 1` is what bounds the cost.

If the project runs on free-trial credits, it stops when the credits run out or
expire, unless the billing account is upgraded to a paid one. After upgrading,
Cloud Run's monthly free allowance still applies, and a demo with occasional
visitors usually stays inside it.

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
- **Environment Variables:** `VITE_API_URL` = the Cloud Run service URL (no
  trailing slash)

Deploy. The site is at `https://<project>.vercel.app`.

`VITE_API_URL` is compiled into the JavaScript at build time. If you change it
later, redeploy for it to take effect. Anything prefixed `VITE_` ends up in the
public bundle, so never put a key in one.

### 3. Allow the frontend to call the API

Set `ALLOWED_ORIGINS` to the Vercel address, for example
`https://regrag.vercel.app`. That's the exact origin, with no trailing slash.
Keep `http://localhost:5173` if you still develop locally:

```bash
gcloud run services update regrag-api \
    --update-env-vars "^;^ALLOWED_ORIGINS=https://regrag.vercel.app,http://localhost:5173"
```

The update starts a new revision with the new value.

Only those addresses may call the API from a browser. Vercel's preview
deployments use other URLs, so they will show "Couldn't reach the server". That
is CORS doing its job, not a bug.

### 4. Try it

Open the Vercel URL and click an example question. If the service had scaled to
zero, the first question takes a little longer while a container starts.

---

## All environment variables

| Where | Name | Required | Purpose |
|---|---|---|---|
| Cloud Run (from Secret Manager) | `GROQ_API_KEY` | yes | generation. Without it the API serves retrieval only |
| Cloud Run | `ALLOWED_ORIGINS` | yes | comma-separated origins allowed to call the API from a browser |
| Cloud Run | `FORWARDED_HOPS` | yes | trusted proxies in front of the app. Verify with Part 1, step 4 |
| Cloud Run | `DEBUG_FORWARDING` | no | `1` enables `/debug/forwarding`. Remove after checking |
| Cloud Run | `RATE_*` | no | rate limits. Defaults fit the Groq free tier (see `.env.example`) |
| Cloud Run | `GROQ_MODEL` | no | default `openai/gpt-oss-120b`. Must support strict JSON schema |
| Vercel | `VITE_API_URL` | yes | the Cloud Run service URL, baked in at build time |

Cloud Run sets `PORT` itself; the Dockerfile reads it.

---

## Running it day to day

- **Updating the API.** Change code in the repo, then
  `python space/sync.py --target api` and `gcloud run deploy regrag-api --source api_space`.
  A redeploy without flags keeps the existing secret, variables and limits.
- **Updating the frontend.** Push to `main`. Vercel rebuilds automatically.
- **The daily allowance.** Groq's free tier is about 200,000 tokens a day, roughly
  70 answers. The API stops generating at 60 a day (`RATE_GENERATIONS_PER_DAY`),
  keeping a margin. After that, visitors see the retrieved passages with a note
  explaining why, rather than an error. The counts are in memory, so they reset
  whenever the container stops, which on Cloud Run happens after it has been
  idle for a while. If Groq's own limit is reached first, generation fails and
  the visitor gets the same retrieval-only answer.
- **Cold starts.** After an idle spell the first request waits for a container to
  start. Before sending the link to someone, open it yourself first.
- **Logs.** `gcloud run services logs read regrag-api --limit 50`, or the service's
  Logs tab in the console. They show startup, errors and failed generations. The
  API does not log questions.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Couldn't reach the server" on the Vercel site | the first request is still cold-starting, or `ALLOWED_ORIGINS` doesn't exactly match the site's address. The browser console will say "CORS" for the latter |
| 403 from the service URL | deployed without `--allow-unauthenticated`. Redeploy with it, or grant `allUsers` the Cloud Run Invoker role |
| Deploy fails with a permission error during the build | the build's service account lacks a role. The error names it; `roles/run.builder` on the default compute service account is the usual fix |
| Deploy fails reading the secret | step 2's `add-iam-policy-binding` was skipped |
| Container fails to start, "memory limit exceeded" in the logs | `--memory` below 2Gi |
| `/health` stuck on `"starting"` | model loading failed. Check the logs for the error |
| Every answer is retrieval-only | `GROQ_API_KEY` missing or wrong (the note says "switched off"), or the day's allowance is used up (the note says so) |
| 429 for everyone at once | `FORWARDED_HOPS` too high or too low. Redo Part 1, step 4 |
