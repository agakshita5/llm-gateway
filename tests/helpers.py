"""Shared test doubles: scripted providers and response builders."""

import time
from collections.abc import AsyncIterator

from app.providers.base import Provider
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


def make_response(model: str = "m", text: str = "hi", tokens: tuple[int, int] = (3, 2)) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test", created=int(time.time()), model=model,
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop")],
        usage=Usage(prompt_tokens=tokens[0], completion_tokens=tokens[1], total_tokens=sum(tokens)),
    )


def make_chunks(model: str = "m", words=("Hel", "lo")) -> list[ChatCompletionChunk]:
    chunks = [
        ChatCompletionChunk(id="c", created=1, model=model, choices=[ChunkChoice(index=0, delta=Delta(content=w))])
        for w in words
    ]
    chunks.append(ChatCompletionChunk(id="c", created=1, model=model,
                                      choices=[ChunkChoice(index=0, delta=Delta(), finish_reason="stop")],
                                      usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
    return chunks


class ScriptedProvider(Provider):
    """Each call takes the next outcome from a script: an exception to raise, or a value to return.

    for stream(), a value is a list of chunks; an exception inside that list is raised mid-stream.
    """

    def __init__(self, name: str, outcomes: list):
        self.name = name
        self.outcomes = list(outcomes)
        self.calls: list[ChatCompletionRequest] = []

    def _next(self, req):
        self.calls.append(req)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def complete(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._next(req)

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        for item in self._next(req):
            if isinstance(item, BaseException):
                raise item
            yield item
