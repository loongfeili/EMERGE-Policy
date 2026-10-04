import asyncio
import json
import pytest
import httpx
from Emerge.providers.responses_provider import ResponsesProvider
from Emerge.providers.openai_codex_provider import _convert_messages

class Stream:
    def __init__(self, events): self.events = events
    async def aiter_lines(self):
        for event in self.events:
            yield 'data: '+json.dumps(event)
            yield ''

def test_tool_result_ids_and_image_conversion():
    result=ResponsesProvider._parse({'status':'completed','output':[{'type':'function_call','call_id':'call_a','name':'look','arguments':'{"camera":"head"}'}]})
    _,items=_convert_messages([{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,YQ=='}}]}, {'role':'assistant','tool_calls':[result.tool_calls[0].to_openai_tool_call()]}, {'role':'tool','tool_call_id':'call_a','content':'visible'}])
    assert items[0]['content'][0]['type']=='input_image'
    assert items[1]['call_id']==items[2]['call_id']=='call_a'

def test_truncated_stream_never_counts_as_success():
    provider=object.__new__(ResponsesProvider)
    with pytest.raises(RuntimeError,match='without a terminal response'):
        asyncio.run(provider._consume(Stream([{'type':'response.output_text.delta','delta':'partial'}])))

def test_terminal_usage_and_tool_arguments():
    provider=object.__new__(ResponsesProvider)
    event={'type':'response.completed','response':{'status':'completed','usage':{'input_tokens':12,'output_tokens':5,'total_tokens':17},'output':[{'type':'function_call','call_id':'c','name':'move','arguments':'{"x":2}'}]}}
    result=asyncio.run(provider._consume(Stream([event])))
    assert result.tool_calls[0].arguments=={'x':2}
    assert result.usage['total_tokens']==17


@pytest.mark.parametrize('suffix', ['', '/', '/responses', '/responses/'])
def test_explicit_responses_provider_roundtrip_and_internal_transport(monkeypatch, suffix):
    from Emerge.config.schema import Config
    from Emerge.providers.factory import create_provider
    from Emerge.providers import responses_provider

    config = Config.model_validate({
        'agents': {'defaults': {'provider': 'responses', 'model': 'gpt-6-astra', 'reasoningEffort': 'high'}},
        'providers': {'responses': {'apiKey': 'test-placeholder',
            'apiBase': 'https://example.test/api/modelhub/online' + suffix,
            'reasoningSummary': 'auto', 'trustEnv': False, 'extraHeaders': {'X-Test': 'preserved'}}},
    })
    captured = []
    def handler(request):
        body = json.loads(request.content)
        captured.append((request, body))
        assert str(request.url) == 'https://example.test/api/modelhub/online/responses'
        assert request.headers['authorization'] == 'Bearer test-placeholder'
        assert request.headers['X-Test'] == 'preserved'
        assert request.headers['X-TT-LOGID']
        assert body['reasoning'] == {'effort': 'high', 'summary': 'auto'}
        assert body['stream'] is True and body['store'] is False
        assert body['max_output_tokens'] == 2048 and 'temperature' not in body
        assert body['model'] == 'gpt-6-astra'
        assert body['instructions'] == 'Use the tool result.'
        assert body['input'][0]['content'][1]['type'] == 'input_image'
        assert all('timestamp' not in item for item in body['input'])
        if len(captured) == 1:
            assert body['tools'][0]['name'] == 'read_probe'
            assert body['tool_choice'] == {'type': 'function', 'name': 'read_probe'}
            output = [{'type': 'function_call', 'call_id': 'probe_call', 'name': 'read_probe', 'arguments': '{}'}]
        else:
            call, result = body['input'][-2:]
            assert call['type'] == 'function_call' and 'id' not in call
            assert result == {'type': 'function_call_output', 'call_id': 'probe_call', 'output': 'unique-tool-value'}
            assert call['call_id'] == result['call_id']
            output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'unique-tool-value'}]}]
        event = {'type': 'response.completed', 'response': {'status': 'completed', 'output': output}}
        return httpx.Response(200, content='data: ' + json.dumps(event) + '\n\n', headers={'Content-Type': 'text/event-stream'})

    monkeypatch.setenv('EMERGE_RESPONSES_PROXY', 'http://unreachable.invalid:9000')
    monkeypatch.setenv('HTTPS_PROXY', 'http://unreachable.invalid:9000')
    original_client = httpx.AsyncClient
    def client_factory(**kwargs):
        assert kwargs['proxy'] is None and kwargs['trust_env'] is False
        return original_client(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(responses_provider.httpx, 'AsyncClient', client_factory)

    async def run():
        provider = create_provider(config)
        messages = [{'role': 'system', 'content': 'Use the tool result.'},
            {'role': 'user', 'timestamp': 'local-only', 'content': [
                {'type': 'text', 'text': 'Read the probe for this image.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,dGVzdA=='}}]}]
        tools = [{'type': 'function', 'function': {'name': 'read_probe', 'parameters': {'type': 'object'}}}]
        try:
            first = await provider.chat_with_retry(messages, tools=tools, max_tokens=2048,
                tool_choice={'type': 'function', 'function': {'name': 'read_probe'}})
            assert first.finish_reason == 'tool_calls' and len(first.tool_calls) == 1
            call = first.tool_calls[0]
            messages.extend([{'role': 'assistant', 'content': first.content, 'tool_calls': [call.to_openai_tool_call()]},
                             {'role': 'tool', 'tool_call_id': call.id, 'content': 'unique-tool-value'}])
            final = await provider.chat_with_retry(messages, tools=tools, max_tokens=2048)
            assert final.finish_reason == 'stop' and final.content == 'unique-tool-value'
        finally:
            await provider.aclose()
    asyncio.run(run())
    assert captured[0][0].headers['X-TT-LOGID'] != captured[1][0].headers['X-TT-LOGID']
