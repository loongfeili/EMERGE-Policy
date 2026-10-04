import asyncio
import json

import httpx

from Emerge.config.schema import Config
from Emerge.providers import azure_openai_provider as azure
from Emerge.providers.factory import create_provider


def test_azure_wire_contract_and_tool_roundtrip(monkeypatch):
    config = Config.model_validate({
        "agents": {"defaults": {"model": "gpt-6-astra", "provider": "azure_openai", "reasoningEffort": "low"}},
        "providers": {"azure_openai": {
            "apiKey": "test-placeholder", "apiBase": "https://example.test/api/modelhub/online/v2/crawl",
            "apiVersion": "2024-02-01", "maxTokensParameter": "max_tokens", "trustEnv": False,
            "extraHeaders": {"X-Test": "preserved"},
        }},
    })
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append((request, body))
        assert request.url.path == "/api/modelhub/online/v2/crawl/openai/deployments/gpt-6-astra/chat/completions"
        assert request.url.params["api-version"] == "2024-02-01"
        assert request.headers["api-key"] == "test-placeholder"
        assert request.headers["X-Test"] == "preserved"
        assert "authorization" not in request.headers
        assert request.headers["X-TT-LOGID"]
        assert body["model"] == "gpt-6-astra" and body["stream"] is False
        assert body["max_tokens"] == 500 and "max_completion_tokens" not in body
        assert body["reasoning_effort"] == "low" and "temperature" not in body
        assert all("timestamp" not in m and "run_id" not in m for m in body["messages"])
        if len(captured) == 1:
            assert body["messages"][0]["content"][1]["type"] == "image_url"
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_check", "type": "function",
                "function": {"name": "check", "arguments": '{"value":"OK"}'},
            }]}
            finish = "tool_calls"
        else:
            assert body["messages"][-1]["tool_call_id"] == "call_check"
            message, finish = {"role": "assistant", "content": "OK"}, "stop"
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}})

    original_client = httpx.AsyncClient

    def client_factory(**kwargs):
        assert kwargs["trust_env"] is False
        return original_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(azure.httpx, "AsyncClient", client_factory)

    async def run():
        provider = create_provider(config)
        messages = [{"role": "user", "timestamp": "local-only", "run_id": "local-only", "content": [
            {"type": "text", "text": "Check this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
        ]}]
        tools = [{"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}]
        reply = await provider.chat_with_retry(messages, tools=tools, max_tokens=500)
        assert reply.finish_reason == "tool_calls" and reply.tool_calls[0].arguments == {"value": "OK"}
        messages += [{"role": "assistant", "content": None, "tool_calls": [reply.tool_calls[0].to_openai_tool_call()]},
                     {"role": "tool", "tool_call_id": reply.tool_calls[0].id, "content": "OK"}]
        reply = await provider.chat_with_retry(messages, max_tokens=500)
        assert reply.content == "OK" and reply.usage["total_tokens"] == 14
        await provider.aclose()

    asyncio.run(run())
    assert captured[0][0].headers["X-TT-LOGID"] != captured[1][0].headers["X-TT-LOGID"]


def test_existing_azure_configuration_keeps_its_api_contract():
    provider = create_provider(Config.model_validate({
        "agents": {"defaults": {"provider": "azure_openai", "model": "gpt-5.2-chat"}},
        "providers": {"azure_openai": {"apiKey": "test-placeholder", "apiBase": "https://example.test"}},
    }))
    assert provider._build_chat_url("gpt-5.2-chat").endswith("?api-version=2024-10-21")
    body = provider._prepare_request_payload("gpt-5.2-chat", [{"role": "user", "content": "Hi"}])
    assert body["max_completion_tokens"] == 4096 and "max_tokens" not in body
    assert "temperature" not in body
    body = provider._prepare_request_payload("gpt-6-astra", [{"role": "user", "content": "Hi"}])
    assert "temperature" not in body
