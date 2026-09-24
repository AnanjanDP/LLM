import pytest

@pytest.mark.asyncio
async def test_register_and_login(client):
    email = "testuser@example.com"
    password = "secretpassword123"

    # 1. Register User
    reg_res = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["email"] == email

    # 2. Login User
    login_res = await client.post("/api/v1/auth/token", json={"email": email, "password": password})
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 3. Get /me with Token
    headers = {"Authorization": f"Bearer {token}"}
    me_res = await client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == email


@pytest.mark.asyncio
async def test_duplicate_registration_fails(client):
    email = "duplicate@example.com"
    password = "secretpassword123"

    res1 = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert res1.status_code == 201

    res2 = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert res2.status_code == 400
