"""Shared API pacing; quota queue time is separate from component compute time."""
import asyncio
import contextvars
from contextlib import contextmanager
import time

import httpx
from loguru import logger

_budgets = contextvars.ContextVar('api_quota_wait_budgets', default=())


class WaitBudget:
    def __init__(self):
        self.depth = 0
        self.started = 0.0
        self.completed = 0.0

    def elapsed(self):
        return self.completed + (time.monotonic() - self.started if self.depth else 0.0)


@contextmanager
def quota_wait():
    budgets = _budgets.get()
    for budget in budgets:
        if not budget.depth:
            budget.started = time.monotonic()
        budget.depth += 1
    try:
        yield
    finally:
        for budget in budgets:
            budget.depth -= 1
            if not budget.depth:
                budget.completed += time.monotonic() - budget.started


async def wait_for_active_time(awaitable, timeout):
    """Keep the original timeout, excluding only explicit API quota waits."""
    if timeout is None:
        return await awaitable
    budget = WaitBudget()
    token = _budgets.set((*_budgets.get(), budget))
    task = asyncio.ensure_future(awaitable)
    started = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - started - budget.elapsed())
            if remaining <= 0:
                raise asyncio.TimeoutError
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        _budgets.reset(token)


class RequestLimiter:
    def __init__(self, url, token):
        if not token:
            raise ValueError('A shared API limiter requires its own access token')
        self.url = url.rstrip('/')
        self._client = httpx.AsyncClient(trust_env=False, proxy=None,
            timeout=httpx.Timeout(300, connect=10), headers={'Authorization': 'Bearer ' + token})

    async def acquire(self):
        started = time.monotonic()
        with quota_wait():
            response = await self._client.post(self.url + '/acquire', json={})
            response.raise_for_status()
            if response.json().get('granted') is not True:
                raise RuntimeError('API quota coordinator did not grant permission')
        logger.info('API quota wait | elapsed={:.3f}s', time.monotonic() - started)

    async def rate_limited(self):
        response = await self._client.post(self.url + '/rate-limited', json={})
        response.raise_for_status()

    async def aclose(self):
        await self._client.aclose()
