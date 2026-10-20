"""Load test for the gateway.

run against PROVIDER_MODE=fake to measure the gateway's own overhead (auth, rate limit, cache, logging)
without spending tokens. needs an api key with a rate limit high enough not to block the test:

  RATE_LIMIT_CAPACITY=100000 RATE_LIMIT_REFILL_PER_S=100000 PROVIDER_MODE=fake uvicorn app.main:app --port 8000 --workers 4
  GATEWAY_KEY=gw-... locust -f loadtest/locustfile.py --host http://localhost:8000 \
      --headless -u 100 -r 20 -t 60s --csv loadtest/results/run
"""

import os
import uuid

from locust import HttpUser, between, task

KEY = os.environ["GATEWAY_KEY"]


class GatewayUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.client.headers["Authorization"] = f"Bearer {KEY}"

    @task(6)
    def completion_unique(self):
        # a new prompt every time: always a cache miss, goes to the provider
        self.client.post("/v1/chat/completions", name="complete (miss)", json={
            "model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": f"hello {uuid.uuid4()}"}]})

    @task(3)
    def completion_cached(self):
        # same deterministic prompt: a cache hit after the first request
        self.client.post("/v1/chat/completions", name="complete (cache hit)", json={
            "model": "openai/gpt-oss-20b", "temperature": 0, "messages": [{"role": "user", "content": "what is 2+2?"}]})

    @task(1)
    def completion_stream(self):
        with self.client.post("/v1/chat/completions", name="stream", stream=True, catch_response=True, json={
                "model": "openai/gpt-oss-20b", "stream": True, "messages": [{"role": "user", "content": "count"}]}) as r:
            # read the whole stream so the timing covers the full answer, not just the headers
            body = b"".join(r.iter_content(None))
            if b"[DONE]" not in body:
                r.failure("stream did not finish")
