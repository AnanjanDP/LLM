import os
import tempfile

# The tests must not depend on (or write to) a developer's real settings: set everything BEFORE the app is imported.
_DATA_DIR = tempfile.mkdtemp(prefix="rag-tests-")
os.environ.update({
    "ENVIRONMENT": "test",
    "SECRET_KEY": "test-secret-key-for-the-unit-tests-only-0123456789",
    "GROQ_API_KEY": "",
    "DEFAULT_LLM_PROVIDER": "mock",
    "CACHE_ENABLED": "false",
    "FAISS_INDEX_PATH": os.path.join(_DATA_DIR, "faiss.index"),
    "FAISS_DOCS_PATH": os.path.join(_DATA_DIR, "documents.json"),
})

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.main import app
from app.core.database import Base, get_db
from app.services.rag_service import rag_service

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
def empty_index():
    """Every test starts and ends with an empty vector store, so tests cannot affect each other."""
    rag_service.clear_index(everything=True)
    yield
    rag_service.clear_index(everything=True)


@pytest_asyncio.fixture(scope="function")
async def test_db():
    engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with TestSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(test_db):
    async def _override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def register_and_login(client, email: str, password: str = "correct-horse-battery") -> dict:
    """Create a user and return Authorization headers for them."""
    res = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert res.status_code == 201, res.text
    login = await client.post("/api/v1/auth/token", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest_asyncio.fixture
async def alice(client):
    return await register_and_login(client, "alice@example.com")


@pytest_asyncio.fixture
async def bob(client):
    return await register_and_login(client, "bob@example.com")
