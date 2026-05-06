# Copyright 2026 LiveKit, Inc.
import aiohttp
import pytest_asyncio

# Auto-load .env from the repo root (or any parent) so the integration test
# can pick up MUNSIT_API_KEY without an explicit `export`. Silent no-op if
# python-dotenv isn't installed or no .env exists (e.g. in CI).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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
