# Copyright 2026 LiveKit, Inc.
import aiohttp
import pytest_asyncio

from .fake_munsit import FakeMunsitServer


@pytest_asyncio.fixture
async def fake_munsit():
    server = FakeMunsitServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest_asyncio.fixture
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session
