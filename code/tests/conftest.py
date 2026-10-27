"""
Test fixtures.

Tests use an in-memory SQLite database. Automatic processing opens a fresh
session, patched below to the same test engine; Groq is stubbed so tests never
require a network connection or a real API key.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app import models as _models  # noqa: F401 -- ensure all models are registered
from app.core.db import Base, get_db
from app.core.enums import UserRole
from app.core.security import hash_password
from app.main import app
from app.repositories.user_repository import UserRepository


@pytest.fixture()
def engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session, engine, monkeypatch):
    # Background tasks must use the same isolated test database, but still a
    # separate session from the request dependency (as they do in production).
    from app.services import paper_processing
    from app.services.groq_service import GroqService

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(paper_processing, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(GroqService, "extract_topics", lambda self, text: ["Test Topic", "Machine Learning"])

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def make_user(db_session):
    """Factory fixture for creating users directly in the test database."""
    def _make(
        email: str = "user@example.com",
        password: str = "password123",
        full_name: str = "Test User",
        role: UserRole = UserRole.STUDENT,
        is_active: bool = True,
    ):
        repo = UserRepository(db_session)
        user = repo.create(
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
            role=role,
        )
        if not is_active:
            user.is_active = False
            repo.save(user)
        return user
    return _make


def auth_headers(client: TestClient, email: str, password: str) -> dict:
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """Use a temporary uploads directory for each test."""
    from app.core.config import settings

    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(d))
    return d


def make_pdf_bytes(text: str = "Hello ScholarLink") -> bytes:
    """Build a small valid PDF in memory for extraction tests."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data
