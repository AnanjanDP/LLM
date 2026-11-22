import io
import pytest
from pydantic import ValidationError
from app.core.config import DEFAULT_SECRET_KEY, Settings
from app.services.rag_service import rag_service

SESSION = "33333333-cccc-4ccc-8ccc-333333333333"
OTHER_SESSION = "44444444-dddd-4ddd-8ddd-444444444444"


def upload(client, content=b"The aardvark migrates at night.", name="notes.txt", session=SESSION, headers=None, **params):
    return client.post(
        "/api/v1/upload-doc",
        params={"session_id": session, **params},
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        headers=headers or {},
    )


# ---------------------------------------------------------------- uploads
@pytest.mark.asyncio
async def test_upload_indexes_the_document_for_its_session_only(client):
    res = await upload(client)
    assert res.status_code == 201
    assert res.json()["document"]["chunk_count"] >= 1

    chat = lambda session: client.post("/api/v1/chat/", json={"query": "when does the aardvark migrate", "session_id": session, "provider": "mock"})
    assert [s["source_doc"] for s in (await chat(SESSION)).json()["sources"]] == ["notes.txt"]
    assert (await chat(OTHER_SESSION)).json()["sources"] == []


@pytest.mark.asyncio
async def test_upload_requires_a_session(client):
    res = await client.post("/api/v1/upload-doc", files={"file": ("a.txt", io.BytesIO(b"text"), "text/plain")})
    assert res.status_code == 422  # no anonymous writes into the shared knowledge base
    assert rag_service.documents == []


@pytest.mark.asyncio
async def test_upload_rejects_bad_files(client):
    assert (await upload(client, name="malware.exe")).status_code == 400
    assert (await upload(client, content=b"   \n  ")).status_code == 400
    assert (await upload(client, content=b"%PDF-1.4 definitely not a real pdf", name="x.pdf")).status_code == 400


@pytest.mark.asyncio
async def test_upload_size_limit_is_enforced(client, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    res = await upload(client, content=b"a " * (600 * 1024))  # ~1.2 MB
    assert res.status_code == 413
    assert rag_service.documents == []


@pytest.mark.asyncio
async def test_upload_sanitises_the_filename(client):
    res = await upload(client, name="../../etc/passwd.txt")
    assert res.status_code == 201
    assert res.json()["document"]["filename"] == "passwd.txt"
    res = await upload(client, name="we\x00ird\x1fname.md")
    assert res.json()["document"]["filename"] == "weirdname.md"


@pytest.mark.asyncio
async def test_upload_validates_chunk_options(client):
    assert (await upload(client, chunk_strategy="nonsense")).status_code == 422
    assert (await upload(client, chunk_size=5)).status_code == 422
    assert (await upload(client, chunk_size=50, chunk_overlap=400)).status_code == 201  # overlap is capped, not an error


@pytest.mark.asyncio
async def test_uploading_into_someone_elses_conversation_is_refused(client, alice, bob):
    await client.post("/api/v1/chat/", json={"query": "private chat", "session_id": SESSION, "provider": "mock"}, headers=alice)
    assert (await upload(client, session=SESSION, headers=bob)).status_code == 404
    assert (await upload(client, session=SESSION)).status_code == 404
    assert (await upload(client, session=SESSION, headers=alice)).status_code == 201


# ---------------------------------------------------------------- clear-documents
@pytest.mark.asyncio
async def test_clear_documents_needs_a_session_and_can_never_wipe_everything(client):
    await upload(client, session=SESSION)
    await upload(client, content=b"Another document about herons.", name="herons.txt", session=OTHER_SESSION)
    total = rag_service.index.ntotal

    assert (await client.delete("/api/v1/clear-documents")).status_code == 422  # the old "clear all" call

    res = await client.delete("/api/v1/clear-documents", params={"session_id": SESSION})
    assert res.status_code == 200 and res.json()["removed"] >= 1
    assert rag_service.index.ntotal < total
    assert {d["session_id"] for d in rag_service.documents} == {OTHER_SESSION}


@pytest.mark.asyncio
async def test_clear_documents_respects_conversation_ownership(client, alice, bob):
    await client.post("/api/v1/chat/", json={"query": "mine", "session_id": SESSION, "provider": "mock"}, headers=alice)
    await upload(client, session=SESSION, headers=alice)

    assert (await client.delete("/api/v1/clear-documents", params={"session_id": SESSION}, headers=bob)).status_code == 404
    assert (await client.delete("/api/v1/clear-documents", params={"session_id": SESSION})).status_code == 404
    assert len(rag_service.documents) >= 1

    assert (await client.delete("/api/v1/clear-documents", params={"session_id": SESSION}, headers=alice)).status_code == 200
    assert rag_service.documents == []


# ---------------------------------------------------------------- configuration
def test_production_refuses_the_public_default_secret():
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(ENVIRONMENT="production", SECRET_KEY=DEFAULT_SECRET_KEY)
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(ENVIRONMENT="production", SECRET_KEY="too-short")
    assert Settings(ENVIRONMENT="production", SECRET_KEY="x" * 40).is_production
    # Development keeps working with the example key.
    assert not Settings(ENVIRONMENT="development", SECRET_KEY=DEFAULT_SECRET_KEY).is_production


def test_production_refuses_wildcard_cors():
    with pytest.raises(ValidationError, match="CORS"):
        Settings(ENVIRONMENT="production", SECRET_KEY="x" * 40, CORS_ORIGINS=["*"])


@pytest.mark.asyncio
async def test_cors_only_allows_configured_origins(client):
    allowed = await client.options("/api/v1/chat/", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    foreign = await client.options("/api/v1/chat/", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in foreign.headers


@pytest.mark.asyncio
async def test_the_api_is_not_mounted_twice(client):
    assert (await client.post("/chat/", json={"query": "hello there", "provider": "mock"})).status_code == 404
    assert (await client.delete("/clear-documents")).status_code in (404, 405)
