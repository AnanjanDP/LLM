import pytest

@pytest.mark.asyncio
async def test_chat_completion_endpoint(client):
    payload = {
        "query": "Hello, how does this RAG system work?",
        "session_id": "test-session-123",
        "provider": "mock",
        "use_rag": True
    }
    response = await client.post("/api/v1/chat/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "conversation_id" in data
    assert "sources" in data


@pytest.mark.asyncio
async def test_list_and_delete_conversations(client):
    # 1. Create message
    await client.post("/api/v1/chat/", json={"query": "Test query", "session_id": "session-to-del", "provider": "mock"})

    # 2. List Conversations
    list_res = await client.get("/api/v1/chat/conversations")
    assert list_res.status_code == 200
    convs = list_res.json()
    assert len(convs) > 0
    target_id = convs[0]["id"]

    # 3. Delete Conversation
    del_res = await client.delete(f"/api/v1/chat/conversations/{target_id}")
    assert del_res.status_code == 204
