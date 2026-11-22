import pytest
from app.core.limiter import limiter
from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_register_and_login(client):
    email = "testuser@example.com"
    password = "secretpassword123"

    reg_res = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert reg_res.status_code == 201
    assert reg_res.json()["email"] == email

    login_res = await client.post("/api/v1/auth/token", json={"email": email, "password": password})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]

    me_res = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == email


@pytest.mark.asyncio
async def test_duplicate_registration_fails(client):
    payload = {"email": "duplicate@example.com", "password": "secretpassword123"}
    assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 201
    assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 400


@pytest.mark.asyncio
async def test_email_is_case_insensitive(client):
    password = "secretpassword123"
    assert (await client.post("/api/v1/auth/register", json={"email": "Mixed@Example.com", "password": password})).status_code == 201
    # The same mailbox in another spelling is a duplicate, and can log in.
    assert (await client.post("/api/v1/auth/register", json={"email": "mixed@example.com", "password": password})).status_code == 400
    assert (await client.post("/api/v1/auth/token", json={"email": "MIXED@EXAMPLE.COM", "password": password})).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("password", ["short", "x" * 73])
async def test_password_length_limits(client, password):
    res = await client.post("/api/v1/auth/register", json={"email": "len@example.com", "password": password})
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_wrong_password_and_unknown_user_look_the_same(client):
    await client.post("/api/v1/auth/register", json={"email": "real@example.com", "password": "secretpassword123"})
    wrong = await client.post("/api/v1/auth/token", json={"email": "real@example.com", "password": "not-the-password"})
    unknown = await client.post("/api/v1/auth/token", json={"email": "nobody@example.com", "password": "not-the-password"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["error"] == unknown.json()["error"]


@pytest.mark.asyncio
async def test_garbage_and_non_access_tokens_are_ignored(client):
    assert (await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"})).status_code == 401
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    # A token that is validly signed but is not an access token must not authenticate.
    refresh_like = create_access_token("some-user-id")
    import jwt
    from app.core.config import settings
    payload = jwt.decode(refresh_like, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    payload["type"] = "refresh"
    forged = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    assert (await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {forged}"})).status_code == 401


@pytest.mark.asyncio
async def test_login_is_rate_limited(client, monkeypatch):
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()
    codes = []
    for _ in range(12):
        res = await client.post("/api/v1/auth/token", json={"email": "x@example.com", "password": "whatever-password"})
        codes.append(res.status_code)
    assert codes[:10] == [401] * 10
    assert 429 in codes[10:]
    limiter.reset()
