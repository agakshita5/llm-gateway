from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from app.schemas import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse

class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, req: ChatCompletionRequest) -> ChatCompletionResponse: ...

    @abstractmethod
    def stream(self, req: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]: ...

    async def aclose(self) -> None:
        """Release resources (e.g. the http client) on app shutdown. Not abstract: no-op by default."""


class ProviderError(Exception):
    """Base for every error an adapter raises. The router catches only these, never httpx/vendor errors."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class ProviderRateLimitError(ProviderError):
    """429: the provider says to slow down."""

    def __init__(self, provider: str, message: str, retry_after: float | None = None):
        super().__init__(provider, message, status_code=429)
        self.retry_after = retry_after    # seconds, from the Retry-After header if present


class ProviderUnavailableError(ProviderError):
    """5xx, timeout, or connection failure: the provider is down or too slow."""


class ProviderAuthError(ProviderError):
    """Our credentials for this provider were rejected (401/403). Another provider may still work."""


class ProviderRequestError(ProviderError):
    """Any other 4xx: the request itself was rejected (400, 404, 422, ...)."""

