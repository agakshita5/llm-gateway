"""In-process stub provider, selected with PROVIDER_MODE=fake.

Used for load tests: it measures the gateway's own overhead (auth, rate limit, cache, logging)
without spending real tokens or hitting provider rate limits.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator

from .base import Provider
from app.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)

_WORDS = ["this", "is", "a", "fake", "streamed", "answer", "from", "the", "stub", "provider"]


class FakeProvider(Provider):
    def __init__(self, name: str, latency_ms: int = 200):
        self.name = name
        self._latency_s = latency_ms / 1000

    async def complete(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        # pretend the model takes this long
        await asyncio.sleep(self._latency_s)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=" ".join(_WORDS)), finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=len(_WORDS), total_tokens=10 + len(_WORDS)),
        )

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        # spread the latency across the words, like tokens arriving one by one
        per_word = self._latency_s / len(_WORDS)
        for i, word in enumerate(_WORDS):
            await asyncio.sleep(per_word)
            last = i == len(_WORDS) - 1
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=req.model,
                choices=[ChunkChoice(index=0, delta=Delta(role="assistant" if i == 0 else None, content=word + " "), finish_reason="stop" if last else None)],
                usage=Usage(prompt_tokens=10, completion_tokens=len(_WORDS), total_tokens=10 + len(_WORDS)) if last else None,
            )
