import json
import pytest
from app.core.exceptions import LLMProviderError
from app.services.cache_service import cache_service
from app.services.llm_provider import GroqProvider, llm_factory
from app.services.rag_service import rag_service

SESSION_A = "11111111-aaaa-4aaa-8aaa-111111111111"
SESSION_B = "22222222-bbbb-4bbb-8bbb-222222222222"


def chat_payload(query="Hello, how does this RAG system work?", session=SESSION_A, **extra):
    return {"query": query, "session_id": session, "provider": "mock", "use_rag": True, **extra}


@pytest.mark.asyncio
async def test_chat_completion_endpoint(client):
    response = await client.post("/api/v1/chat/", json=chat_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == SESSION_A
    assert data["provider"] == "mock"
    assert "answer" in data and "sources" in data


@pytest.mark.asyncio
async def test_chat_validates_its_input(client):
    assert (await client.post("/api/v1/chat/", json=chat_payload(top_k=0))).status_code == 422
    assert (await client.post("/api/v1/chat/", json=chat_payload(top_k=500))).status_code == 422
    assert (await client.post("/api/v1/chat/", json=chat_payload(retrieval_mode="nonsense"))).status_code == 422
    assert (await client.post("/api/v1/chat/", json=chat_payload(session="../../etc/passwd"))).status_code in (400, 422)
    assert (await client.post("/api/v1/chat/", json={**chat_payload(), "provider": "not-a-provider"})).status_code == 422


@pytest.mark.asyncio
async def test_guest_can_list_only_the_conversations_they_name(client):
    await client.post("/api/v1/chat/", json=chat_payload(session=SESSION_A))
    await client.post("/api/v1/chat/", json=chat_payload(session=SESSION_B))

    nothing = await client.get("/api/v1/chat/conversations")
    assert nothing.status_code == 200
    assert nothing.json() == []  # a guest sees none of the other conversations in the database

    mine = await client.get("/api/v1/chat/conversations", params={"ids": SESSION_A})
    assert [c["id"] for c in mine.json()] == [SESSION_A]


@pytest.mark.asyncio
async def test_delete_and_missing_conversations(client):
    await client.post("/api/v1/chat/", json=chat_payload())
    assert (await client.delete(f"/api/v1/chat/conversations/{SESSION_A}")).status_code == 204
    assert (await client.get(f"/api/v1/chat/conversations/{SESSION_A}")).status_code == 404
    assert (await client.delete(f"/api/v1/chat/conversations/{SESSION_A}")).status_code == 404


@pytest.mark.asyncio
async def test_signed_in_conversations_are_private(client, alice, bob):
    await client.post("/api/v1/chat/", json=chat_payload(query="alice's secret question"), headers=alice)

    # Alice sees and can read it.
    assert [c["id"] for c in (await client.get("/api/v1/chat/conversations", headers=alice)).json()] == [SESSION_A]
    own = await client.get(f"/api/v1/chat/conversations/{SESSION_A}", headers=alice)
    assert own.status_code == 200
    assert "alice's secret question" in json.dumps(own.json())

    # Bob, a guest and an unauthenticated caller cannot read, list, continue or delete it.
    for headers in (bob, {}):
        assert (await client.get(f"/api/v1/chat/conversations/{SESSION_A}", headers=headers)).status_code == 404
        assert (await client.delete(f"/api/v1/chat/conversations/{SESSION_A}", headers=headers)).status_code == 404
        assert (await client.post("/api/v1/chat/", json=chat_payload(query="hijack"), headers=headers)).status_code == 404
        assert (await client.post("/api/v1/chat/stream", json=chat_payload(query="hijack"), headers=headers)).status_code == 404
    assert (await client.get("/api/v1/chat/conversations", headers=bob)).json() == []

    # Still intact after the attempts.
    still = await client.get(f"/api/v1/chat/conversations/{SESSION_A}", headers=alice)
    assert len(still.json()["messages"]) == 2


@pytest.mark.asyncio
async def test_session_documents_are_only_used_for_their_own_conversation(client):
    rag_service.add_document("The zeppelin maintenance manual says to oil the propeller weekly.", doc_name="manual",
                             session_id=SESSION_A)
    mine = await client.post("/api/v1/chat/", json=chat_payload(query="how often to oil the propeller", session=SESSION_A))
    theirs = await client.post("/api/v1/chat/", json=chat_payload(query="how often to oil the propeller", session=SESSION_B))
    assert [s["source_doc"] for s in mine.json()["sources"]] == ["manual"]
    assert theirs.json()["sources"] == []


@pytest.mark.asyncio
async def test_stream_endpoint_streams_tokens_and_saves_the_exchange(client):
    res = await client.post("/api/v1/chat/stream", json=chat_payload(query="Explain hybrid retrieval please"))
    assert res.status_code == 200
    events = [json.loads(line[6:]) for line in res.text.split("\n") if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "metadata" and kinds[-1] == "done"
    assert "token" in kinds
    assert events[0]["data"]["conversation_id"] == SESSION_A

    history = await client.get(f"/api/v1/chat/conversations/{SESSION_A}")
    assert [m["role"] for m in history.json()["messages"]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_unexpected_errors_do_not_leak_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal path C:/srv/app/.env and SQL: SELECT * FROM users")

    monkeypatch.setattr(rag_service, "retrieve", boom)
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        res = await c.post("/api/v1/chat/", json=chat_payload())
    assert res.status_code == 500
    assert "secret internal path" not in res.text and "SELECT" not in res.text


# ---------------------------------------------------------------- cache keys
def test_cache_key_covers_everything_that_changes_the_answer():
    base = dict(query="q", mode="hybrid", model="m", provider="groq", prompt="v1.0", top_k=4,
                rerank=False, use_rag=True, session="s", index_version=1)
    reference = cache_service._key(**base)
    for field, other in [("top_k", 8), ("rerank", True), ("prompt", "v1.1_strict"), ("mode", "vector"),
                         ("provider", "mock"), ("model", "x"), ("session", "t"), ("index_version", 2), ("use_rag", False)]:
        assert cache_service._key(**{**base, field: other}) != reference, field
    assert cache_service._key(**base) == reference


def test_adding_a_document_changes_the_index_version(client):
    before = rag_service.version
    rag_service.add_document("a new fact about zeppelins", doc_name="d", session_id=SESSION_A)
    assert rag_service.version == before + 1
    rag_service.clear_index(SESSION_A)
    assert rag_service.version == before + 2


# ---------------------------------------------------------------- llm provider failures are not disguised
class _FakeCompletions:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs["model"])
        raise self.error


class _FakeClient:
    def __init__(self, error):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(error)})()


def _groq_error(cls, status):
    import groq
    import httpx
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/x"))
    return cls("failure", response=response, body=None)


def _provider(error):
    provider = GroqProvider.__new__(GroqProvider)
    provider.client = _FakeClient(error)
    provider.default_model = "model-a"
    return provider


def test_a_rejected_api_key_raises_instead_of_returning_a_fake_answer():
    import groq
    provider = _provider(_groq_error(groq.AuthenticationError, 401))
    with pytest.raises(LLMProviderError, match="key was rejected"):
        provider.generate([{"role": "user", "content": "hi"}])
    assert len(provider.client.chat.completions.calls) == 1  # no pointless retries on other models


def test_a_missing_model_falls_back_to_the_next_model_and_then_fails_loudly():
    import groq
    provider = _provider(_groq_error(groq.NotFoundError, 404))
    with pytest.raises(LLMProviderError):
        provider.generate([{"role": "user", "content": "hi"}], model="model-a")
    assert len(provider.client.chat.completions.calls) >= 2  # tried the fallbacks too


def test_unknown_provider_is_an_error_and_missing_key_uses_the_mock():
    with pytest.raises(LLMProviderError):
        llm_factory.get_provider("openai-typo")
    assert llm_factory.get_provider("groq").name == "mock"  # no key configured in tests
