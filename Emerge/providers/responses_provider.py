"""API-key provider for explicitly configured Responses API endpoints."""
from __future__ import annotations
import json
import os
from typing import Any
import httpx
from Emerge.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from Emerge.providers.openai_codex_provider import _convert_messages, _convert_tools

class ResponsesProvider(LLMProvider):
    def __init__(self, api_key: str, api_base: str, default_model: str):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = httpx.AsyncClient(trust_env=False, proxy=os.environ.get("EMERGE_RESPONSES_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or None, timeout=httpx.Timeout(180, connect=15))

    async def aclose(self):
        await self._client.aclose()

    def get_default_model(self):
        return self.default_model

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        instructions, items = _convert_messages(messages)
        body: dict[str, Any] = {'model': model or self.default_model, 'input': items,
                                'stream': True, 'store': False, 'max_output_tokens': max(1, max_tokens)}
        for item in items:
            if item.get('type') == 'function_call':
                item.pop('id', None)
        if instructions:
            body['instructions'] = instructions
        if reasoning_effort:
            body['reasoning'] = {'effort': reasoning_effort}
        if tools:
            body['tools'] = _convert_tools(tools)
            choice = tool_choice or 'auto'
            if isinstance(choice, dict) and choice.get('type') == 'function':
                choice = {'type': 'function', 'name': choice.get('function', {}).get('name', choice.get('name'))}
            body['tool_choice'] = choice
        try:
            async with self._client.stream('POST', self.api_base,
                    headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}, json=body) as response:
                if response.is_error:
                    detail = (await response.aread()).decode("utf-8", "replace")[:1200]
                    raise RuntimeError(f"HTTP {response.status_code}: {detail}")
                return await self._consume(response)
        except Exception as exc:
            return LLMResponse(content=f'Responses API error: {type(exc).__name__}: {exc}'.replace(self.api_key, '[redacted]'), finish_reason='error')

    async def _consume(self, response):
        from Emerge.providers.openai_codex_provider import _iter_sse
        from Emerge.providers.streaming import text_sink
        async for event in _iter_sse(response):
            kind = event.get('type')
            if kind == 'response.output_text.delta':
                sink = text_sink.get()
                if sink is not None:
                    await sink.send(event.get('delta', ''))
            elif kind in ('response.completed', 'response.incomplete'):
                return self._parse(event['response'])
            elif kind in ('error', 'response.failed'):
                raise RuntimeError(json.dumps(event.get('error') or event.get('response', {}).get('error') or {'type':kind}))
        raise RuntimeError('Responses stream ended without a terminal response')

    @staticmethod
    def _parse(response):
        texts, calls = [], []
        for item in response.get('output', []):
            if item.get('type') == 'message':
                texts += [p['text'] for p in item.get('content', []) if p.get('type') == 'output_text']
            elif item.get('type') == 'function_call':
                arguments = json.loads(item.get('arguments') or '{}')
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be an object')
                calls.append(ToolCallRequest(id=item['call_id'], name=item['name'], arguments=arguments))
        usage = response.get('usage') or {}
        return LLMResponse(content=''.join(texts) or None, tool_calls=calls,
            finish_reason='length' if response.get('status') == 'incomplete' else ('tool_calls' if calls else 'stop'),
            usage={'prompt_tokens':usage.get('input_tokens',0),'completion_tokens':usage.get('output_tokens',0),'total_tokens':usage.get('total_tokens',0)})
