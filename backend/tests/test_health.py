import pytest

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_check(client):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "database_connected" in data
    assert "vector_index_vectors" in data
