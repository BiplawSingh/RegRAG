"""
Rate limits for a public endpoint that spends a personal Groq allowance.

Two different things are being protected, so there are two kinds of limit:

  REQUESTS  every /ask costs CPU (embedding + a 50-pair cross-encoder pass), so
            one visitor gets a few per minute. Over it: HTTP 429.
  GENERATIONS  the Groq free tier is ~200,000 tokens a day, about 70 answers, and
            ~3 calls a minute. Over any of these: the request still succeeds with
            retrieval only, and says why. The retrieved passages and their scores
            are the half of the project a visitor can still learn from.

The global per-minute generation cap also keeps rag._throttle() from sleeping
inside a request: that pacing is right for a batch run, but a visitor would see
it as a minute-long hang.

In-memory and per-process, deliberately. The server runs one worker (see
DECISIONS.md), so there is exactly one copy of these counters; a Redis-backed
limiter would add a service to solve a problem this deployment does not have.
A restart resets them, which errs in the visitor's favour.
"""

import os
import threading
import time
from collections import defaultdict, deque

MINUTE = 60
DAY = 24 * 60 * 60


class Window:
    """At most `limit` events in any rolling `seconds`."""

    def __init__(self, limit, seconds):
        self.limit, self.seconds = limit, seconds
        self.events = deque()

    def _prune(self, now):
        while self.events and now - self.events[0] >= self.seconds:
            self.events.popleft()

    def has_room(self, now):
        self._prune(now)
        return len(self.events) < self.limit

    def record(self, now):
        self.events.append(now)

    def retry_after(self, now):
        self._prune(now)
        if len(self.events) < self.limit:
            return 0.0
        return self.seconds - (now - self.events[0])


class RateLimiter:
    def __init__(self, ip_per_minute=5, ip_gen_per_day=20,
                 gen_per_minute=3, gen_per_day=60, clock=time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self._ip_requests = defaultdict(lambda: Window(ip_per_minute, MINUTE))
        self._ip_generations = defaultdict(lambda: Window(ip_gen_per_day, DAY))
        self._gen_minute = Window(gen_per_minute, MINUTE)
        self._gen_day = Window(gen_per_day, DAY)

    @classmethod
    def from_env(cls):
        e = os.environ
        return cls(ip_per_minute=int(e.get("RATE_IP_PER_MINUTE", 5)),
                   ip_gen_per_day=int(e.get("RATE_IP_GENERATIONS_PER_DAY", 20)),
                   gen_per_minute=int(e.get("RATE_GENERATIONS_PER_MINUTE", 3)),
                   gen_per_day=int(e.get("RATE_GENERATIONS_PER_DAY", 60)))

    def admit(self, ip):
        """Count one request. Returns 0 if admitted, else seconds until it would be."""
        with self._lock:
            now = self.clock()
            w = self._ip_requests[ip]
            if not w.has_room(now):
                return w.retry_after(now)
            w.record(now)
            return 0.0

    def allow_generation(self, ip):
        """Count one generation if every generation budget has room; else False.

        Checked all-or-nothing: a request refused by the global cap must not
        still spend the visitor's own daily allowance.
        """
        with self._lock:
            now = self.clock()
            windows = (self._ip_generations[ip], self._gen_minute, self._gen_day)
            if not all(w.has_room(now) for w in windows):
                return False
            for w in windows:
                w.record(now)
            return True


def client_ip(request, forwarded_hops):
    """The visitor's address, as seen through `forwarded_hops` trusted proxies.

    Behind a proxy every request arrives from the proxy's address, so the real
    one has to come from X-Forwarded-For. Only the entries the trusted proxies
    appended can be believed -- a visitor can put anything at the front of that
    header -- so read from the right, `forwarded_hops` back. With 0 hops (local
    development, no proxy) the header is ignored entirely.
    """
    if forwarded_hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")
                 if p.strip()]
        if len(parts) >= forwarded_hops:
            return parts[-forwarded_hops]
    return request.client.host if request.client else "unknown"
