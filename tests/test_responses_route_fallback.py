"""An explicitly configured alternate route must not change model or replay output."""
import asyncio,json
import httpx
import pytest
from Emerge.providers.responses_provider import ResponsesProvider
from Emerge.providers.streaming import TextSink,text_sink


def completed():
    event={'type':'response.completed','response':{'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'ok'}]}]}}
    return httpx.Response(200,content='data: '+json.dumps(event)+'\n\n',headers={'content-type':'text/event-stream'})

async def provider_with(handler):
    provider=ResponsesProvider('test-key','https://primary.invalid/responses','same-model',['https://alternate.invalid/responses'])
    await provider.aclose()
    provider._client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def test_gateway_failure_uses_same_body_and_keeps_working_route():
    async def run():
        calls=[]
        def handler(request):
            calls.append(request)
            return httpx.Response(502) if request.url.host=='primary.invalid' else completed()
        provider=await provider_with(handler)
        try:
            assert (await provider.chat([{'role':'user','content':'hello'}])).content=='ok'
            assert (await provider.chat([{'role':'user','content':'next'}])).content=='ok'
        finally:await provider.aclose()
        assert [r.url.host for r in calls]==['primary.invalid','alternate.invalid','alternate.invalid']
        assert json.loads(calls[0].content)==json.loads(calls[1].content)
        assert all(json.loads(r.content)['model']=='same-model' for r in calls)
        assert all(r.headers['authorization']=='Bearer test-key' for r in calls)
    asyncio.run(run())


@pytest.mark.parametrize('status',[400,401,403,429])
def test_non_gateway_error_does_not_switch_routes(status):
    async def run():
        calls=[]
        def handler(request):
            calls.append(request);return httpx.Response(status,content='test-key')
        provider=await provider_with(handler)
        try:result=await provider.chat([{'role':'user','content':'hello'}])
        finally:await provider.aclose()
        assert result.finish_reason=='error' and len(calls)==1
        assert 'test-key' not in result.content
    asyncio.run(run())


def test_read_timeout_can_switch_before_any_output():
    async def run():
        def handler(request):
            if request.url.host=='primary.invalid':raise httpx.ReadTimeout('timeout')
            return completed()
        provider=await provider_with(handler)
        try:assert (await provider.chat([{'role':'user','content':'hello'}])).content=='ok'
        finally:await provider.aclose()
    asyncio.run(run())


def test_partial_delivered_stream_is_never_replayed():
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
            raise httpx.ReadError('broken after partial output')
    async def run():
        calls=[];output=[]
        def handler(request):
            calls.append(request);return httpx.Response(200,stream=BrokenStream())
        async def receive(text):output.append(text)
        provider=await provider_with(handler);token=text_sink.set(TextSink(receive))
        try:result=await provider.chat([{'role':'user','content':'hello'}])
        finally:text_sink.reset(token);await provider.aclose()
        assert result.finish_reason=='error' and len(calls)==1 and output==['partial']
    asyncio.run(run())


def test_factory_reads_explicit_fallback_routes():
    from Emerge.config.schema import Config
    from Emerge.providers.factory import create_provider
    async def run():
        config=Config.model_validate({'agents':{'defaults':{'provider':'custom','model':'same-model'}},'providers':{'custom':{'apiKey':'test-key','apiBase':'https://primary.invalid/responses','apiBaseFallbacks':['https://alternate.invalid/responses']}}})
        provider=create_provider(config)
        try:assert provider._endpoints==['https://primary.invalid/responses','https://alternate.invalid/responses']
        finally:await provider.aclose()
    asyncio.run(run())
