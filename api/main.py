"""
RegRAG's HTTP API.

    uvicorn api.main:app --port 8000        # from the repo root

    POST /ask     {"question": "..."}  ->  answer, citations, sources, retrieval info
    GET  /health  whether the models are loaded and generation is configured

The endpoint is a plain `def`, not `async def`. Everything it calls blocks --
torch inference, a synchronous Groq client -- and FastAPI runs a plain `def`
endpoint in a thread pool, so one slow request does not freeze the server. An
`async def` calling the same code would block the event loop for every visitor.
"""

import logging
import math
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.limits import RateLimiter, client_ip
from api.pipeline import Pipeline
from api.schemas import AskRequest, AskResponse, Health

log = logging.getLogger("regrag.api")


def create_app(pipeline=None, limiter=None, origins=None, forwarded_hops=None):
    """Build the app. Tests pass a fake pipeline and a small limiter; the server
    passes nothing and gets the real ones, configured from the environment."""
    pipeline = pipeline or Pipeline()
    limiter = limiter or RateLimiter.from_env()
    if origins is None:
        origins = [o.strip() for o in
                   os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
                   if o.strip()]
    if forwarded_hops is None:
        forwarded_hops = int(os.getenv("FORWARDED_HOPS", "0"))

    @asynccontextmanager
    async def lifespan(app):
        # Warm at startup so the first visitor does not pay for model loading.
        # Not fatal: a server that is up and reporting "starting" on /health is
        # easier to diagnose than one that died before it could say anything.
        try:
            pipeline.warm()
        except Exception:                                  # noqa: BLE001
            log.exception("warm-up failed; /ask will return 503 until restart")
        yield

    app = FastAPI(title="RegRAG API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/")
    def root():
        # The Hugging Face Space page shows this path. Point a curious visitor
        # somewhere useful instead of at a 404.
        return {"name": "RegRAG API", "ask": "POST /ask", "health": "/health",
                "docs": "/docs", "source": "https://github.com/BiplawSingh/RegRAG"}

    if os.getenv("DEBUG_FORWARDING") == "1":
        @app.get("/debug/forwarding")
        def debug_forwarding(request: Request):
            # Off unless switched on. How many proxies a host puts in front of
            # the app is not something to guess: set FORWARDED_HOPS wrong and
            # every visitor shares one rate-limit budget, or anyone can reset
            # theirs by sending a fake header. This shows the caller their OWN
            # request as the server sees it, so the setting can be checked once
            # after deploying and the endpoint switched off again.
            return {"x_forwarded_for": request.headers.get("x-forwarded-for"),
                    "peer": request.client.host if request.client else None,
                    "forwarded_hops": forwarded_hops,
                    "rate_limit_key": client_ip(request, forwarded_hops)}

    @app.get("/health", response_model=Health)
    def health():
        return Health(
            status="ok" if pipeline.ready else "starting",
            chunks=len(pipeline.chunks),
            models_loaded=pipeline.ready,
            generation_enabled=pipeline.generation_enabled(),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        ip = client_ip(request, forwarded_hops)
        wait = limiter.admit(ip)
        if wait:
            seconds = math.ceil(wait)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many questions from this address. "
                                   "Please wait a moment and try again.",
                         "retry_after": seconds},
                headers={"Retry-After": str(seconds)},
            )
        if not pipeline.ready:
            raise HTTPException(503, "The server is still loading its models. "
                                     "Try again in a minute.")

        allowed = pipeline.generation_enabled() and limiter.allow_generation(ip)
        try:
            return pipeline.ask(body.question, generation_allowed=allowed)
        except Exception:                                  # noqa: BLE001
            # Generation failures are absorbed inside pipeline.ask; reaching here
            # means retrieval itself failed, and there is nothing useful to show.
            log.exception("retrieval failed")
            raise HTTPException(503, "Retrieval failed on the server. "
                                     "Try again shortly.") from None

    return app


app = create_app()
