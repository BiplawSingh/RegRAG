---
title: RegRAG API
emoji: 📜
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Question answering over 506 RBI circulars, with verified citations
---

# RegRAG API

The backend for the RegRAG web demo: FastAPI over a retrieval pipeline for 506
Reserve Bank of India circulars.

- `POST /ask` with `{"question": "..."}` returns the answer, each citation's
  verification result and highlight offsets, the retrieved passages, and every
  ranking stage's score for each passage.
- `GET /health` reports whether the models are loaded.
- `GET /docs` is the interactive API reference.

Source, evaluation and design notes: https://github.com/BiplawSingh/RegRAG

This directory is assembled by `python space/sync.py --target api`. The pipeline
modules and data are copies of the originals in that repo and are never edited
here.
