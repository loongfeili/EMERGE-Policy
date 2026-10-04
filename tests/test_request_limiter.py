import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
import time

import httpx
import pytest

from Emerge.providers.request_limiter import RequestLimiter, quota_wait, wait_for_active_time
from Emerge.providers.responses_provider import ResponsesProvider


def test_quota_wait_does_not_consume_compute_timeout_but_work_does():
    async def run():
        async def queued():
            with quota_wait():
                await asyncio.sleep(.09)
            await asyncio.sleep(.005)
            return 'done'
        assert await wait_for_active_time(queued(), .04) == 'done'
        with pytest.raises(asyncio.TimeoutError):
            await wait_for_active_time(asyncio.sleep(.1), .02)
        finished = asyncio.Event()
        async def cancelled():
            try:
                with quota_wait():
                    await asyncio.sleep(10)
            finally:
                finished.set()
        task = asyncio.create_task(wait_for_active_time(cancelled(), 1))
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(run())


def test_global_pacing_is_shared_across_clients_and_backoff_is_coalesced():
    path = Path(__file__).resolve().parents[1] / 'deployment/merlin/api-limiter.py'
    spec = importlib.util.spec_from_file_location('api_limiter_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pacer = module.Pacer(1200, cooldown=.08)
    def acquire(_):
        pacer.acquire()
        return time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        stamps = sorted(pool.map(acquire, range(8)))
    assert min(b-a for a,b in zip(stamps,stamps[1:])) >= .045
    assert pacer.status()['grants']==8
    pacer.limited()
    pacer.limited()
    assert pacer.status()['rpm']==900
    start=time.monotonic()
    pacer.acquire()
    assert time.monotonic()-start >= .07

    server = module.server('127.0.0.1',0,1200,'coordinator-only')
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    async def run():
        url='http://127.0.0.1:'+str(server.server_port)
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.post(url+'/acquire',json={})).status_code==403
        client=RequestLimiter(url,'coordinator-only')
        try:
            await client.acquire()
            async with httpx.AsyncClient(trust_env=False) as http:
                state=(await http.get(url+'/healthz')).json()
                assert state['grants']==1
        finally:
            await client.aclose()
    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_every_upstream_attempt_needs_a_permit_and_429_reports_global_backoff():
    async def run():
        provider=ResponsesProvider('upstream-secret','https://upstream.test','test',
            rate_limit_url='http://coordinator.test',rate_limit_token='coordinator-only',trust_env=False)
        events=[]
        def limiter(request):
            assert request.headers['authorization']=='Bearer coordinator-only'
            assert b'upstream-secret' not in request.content
            events.append(request.url.path)
            return httpx.Response(200,json={'granted':True})
        def upstream(request):
            events.append('upstream')
            return httpx.Response(429,json={'error':'qpm limit'})
        await provider._client.aclose()
        await provider._limiter._client.aclose()
        provider._client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        provider._limiter._client=httpx.AsyncClient(transport=httpx.MockTransport(limiter),
            headers={'Authorization':'Bearer coordinator-only'})
        try:
            result=await provider.chat([{'role':'user','content':'test'}])
            assert result.finish_reason=='error' and '429' in result.content
            assert events==['/acquire','upstream','/rate-limited']
            events.clear()
            await provider._limiter._client.aclose()
            provider._limiter._client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request:httpx.Response(503,json={'error':'unavailable'})))
            result=await provider.chat([{'role':'user','content':'test'}])
            assert result.finish_reason=='error' and not events
        finally:
            await provider.aclose()
    asyncio.run(run())
