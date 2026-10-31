import json

import httpx
import pytest

from nanobot.providers.base import ProviderCallContext
from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def test_local_endpoint_sends_session_id():
    provider = OpenAICompatProvider(api_key="k", api_base="http://172.17.0.1:4000/v1")
    headers = provider._session_headers(ProviderCallContext(session_id="websocket:7e0d3939"))
    assert headers == {"x-litellm-session-id": "websocket:7e0d3939"}


def test_session_id_is_ascii_safe():
    provider = OpenAICompatProvider(api_key="k", api_base="http://localhost:4000/v1")
    value = provider._session_headers(ProviderCallContext(session_id="sdk:中文 chat"))["x-litellm-session-id"]
    assert value.isascii() and value.startswith("sdk:")


def test_no_session_no_header():
    provider = OpenAICompatProvider(api_key="k", api_base="http://localhost:4000/v1")
    assert provider._session_headers(ProviderCallContext()) is None


def test_hosted_endpoint_never_sees_session_id():
    provider = OpenAICompatProvider(api_key="k", api_base="https://api.openai.com/v1")
    assert provider._session_headers(ProviderCallContext(session_id="websocket:abc")) is None


@pytest.mark.parametrize("stream", [False, True])
async def test_session_id_reaches_the_wire(monkeypatch, stream):
    from openai import AsyncOpenAI

    from nanobot.providers import openai_compat_provider

    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        message = {"role": "assistant", "content": "ok"}
        if stream:
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                     "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
            return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                                         "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(openai_compat_provider, "AsyncOpenAI",
                            lambda **kwargs: AsyncOpenAI(**{**kwargs, "http_client": transport}))
        provider = OpenAICompatProvider(api_key="k", api_base="http://172.17.0.1:4000/v1", default_model="m")
        call = provider.chat_stream_with_retry if stream else provider.chat_with_retry
        result = await call(messages=[{"role": "user", "content": "hi"}],
                            provider_context=ProviderCallContext(session_id="websocket:abc"))
    assert result.content == "ok"
    assert requests[-1].headers["x-litellm-session-id"] == "websocket:abc"
