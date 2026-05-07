from __future__ import annotations

import pytest

from app.services import llm_client


class _FakeAnthropicMessages:
    def create(self, **_kwargs):
        raise AssertionError("unexpected network call in unit test")


class _FakeAnthropic:
    last_api_key: str | None = None

    def __init__(self, *, api_key: str, timeout: float) -> None:
        _FakeAnthropic.last_api_key = api_key
        self.timeout = timeout
        self.messages = _FakeAnthropicMessages()


class _FakeOpenAICompletions:
    def create(self, **_kwargs):
        raise AssertionError("unexpected network call in unit test")


class _FakeOpenAIChat:
    def __init__(self) -> None:
        self.completions = _FakeOpenAICompletions()


class _FakeOpenAI:
    last_api_key: str | None = None
    last_base_url: str | None = None

    def __init__(self, *, api_key: str, base_url: str | None = None, timeout: float) -> None:
        _FakeOpenAI.last_api_key = api_key
        _FakeOpenAI.last_base_url = base_url
        self.base_url = base_url
        self.timeout = timeout
        self.chat = _FakeOpenAIChat()


def test_openai_llm_client_reads_secure_local_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "stored-openai-key" if env_var == "OPENAI_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)

    client = llm_client.OpenAILLMClient(model="gpt-5.4")
    client._ensure_client()

    assert _FakeOpenAI.last_api_key == "stored-openai-key"


def test_anthropic_llm_client_reads_secure_local_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "stored-anthropic-key" if env_var == "ANTHROPIC_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "Anthropic", _FakeAnthropic)

    client = llm_client.LLMClient(model="claude-opus-4")
    client._ensure_client()

    assert _FakeAnthropic.last_api_key == "stored-anthropic-key"


def test_minimax_llm_client_reads_api_key_and_base_url(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "minimax-test-key" if env_var == "MINIMAX_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)

    client = llm_client.OpenAILLMClient(
        model="MiniMax-M2.7",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        temperature=1.0,
    )
    client._ensure_client()

    assert _FakeOpenAI.last_api_key == "minimax-test-key"
    assert _FakeOpenAI.last_base_url == "https://api.minimax.io/v1"


def test_minimax_llm_client_temperature_is_set(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "minimax-test-key" if env_var == "MINIMAX_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)

    client = llm_client.OpenAILLMClient(
        model="MiniMax-M2.7",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        temperature=1.0,
    )
    assert client._temperature == 1.0


class _FakeResponseMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponseChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeResponseMessage(content)


class _FakeOpenAIResponse:
    def __init__(self, content: str = "ok") -> None:
        self.choices = [_FakeResponseChoice(content)]


class _OpenAIBadRequestError(Exception):
    def __init__(self, message: str = "bad request") -> None:
        super().__init__(message)
        self.status_code = 400


class _OpenAIRateLimitError(Exception):
    def __init__(self, message: str = "rate limited") -> None:
        super().__init__(message)
        self.status_code = 429


class _OpenAIStatus503Error(Exception):
    def __init__(self, message: str = "server unavailable") -> None:
        super().__init__(message)
        self.status_code = 503


def _build_openai_client_for_errors(
    monkeypatch,
    error_factory,
    *,
    api_key_env: str = "OPENAI_API_KEY",
):
    class _ErroringCompletions:
        call_count = 0

        def create(self, **_kwargs):
            _ErroringCompletions.call_count += 1
            raise error_factory()

    class _ErroringChat:
        def __init__(self) -> None:
            self.completions = _ErroringCompletions()

    class _ErroringOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None = None, timeout: float) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.chat = _ErroringChat()

    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "test-key" if env_var == api_key_env else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _ErroringOpenAI)
    return (
        llm_client.OpenAILLMClient(model="gpt-5.4", api_key_env=api_key_env),
        _ErroringCompletions,
    )


def test_openai_non_retriable_errors_fail_fast(monkeypatch) -> None:
    client, completions = _build_openai_client_for_errors(monkeypatch, _OpenAIBadRequestError)
    sleep_calls: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(
        RuntimeError,
        match=r"Openai API request failed \(HTTP 400: _OpenAIBadRequestError\)\. "
        r"Check model name, request parameters, and credentials\.",
    ) as exc_info:
        client.invoke("hello")

    assert isinstance(exc_info.value.__cause__, _OpenAIBadRequestError)
    assert completions.call_count == 1
    assert sleep_calls == []


def test_openai_retries_rate_limit_and_surfaces_actionable_message(monkeypatch) -> None:
    client, completions = _build_openai_client_for_errors(monkeypatch, _OpenAIRateLimitError)
    sleep_calls: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(
        RuntimeError,
        match=(
            "Openai API is rate-limited \\(HTTP 429\\) after multiple retries\\. "
            "Try again in a few seconds\\."
        ),
    ) as exc_info:
        client.invoke("hello")

    assert isinstance(exc_info.value.__cause__, _OpenAIRateLimitError)
    assert completions.call_count == 3
    assert sleep_calls == [1.0, 2.0]


def test_openai_retry_message_is_provider_aware(monkeypatch) -> None:
    client, completions = _build_openai_client_for_errors(
        monkeypatch,
        _OpenAIStatus503Error,
        api_key_env="MINIMAX_API_KEY",
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(
        RuntimeError,
        match=(
            "Minimax API is temporarily unavailable \\(HTTP 503\\) after multiple retries\\. "
            "Try again in a few seconds\\."
        ),
    ) as exc_info:
        client.invoke("hello")

    assert isinstance(exc_info.value.__cause__, _OpenAIStatus503Error)
    assert completions.call_count == 3
    assert sleep_calls == [1.0, 2.0]
