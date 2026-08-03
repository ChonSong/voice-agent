"""Pytest configuration for async tests."""
import asyncio
import sys
from pathlib import Path

import pytest

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

# Configure asyncio mode
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def aiohttp_server(aiohttp_server):
    """Override aiohttp_server fixture to handle loop issues."""
    from app import app
    server = await aiohttp_server(app)
    return server
