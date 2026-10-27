import io
import uuid

from app.core.enums import PaperStatus
from app.models.paper import Paper
from app.models.topic import PaperTopic, Topic
from tests.conftest import make_pdf_bytes


def test_upload_automatically_extracts_text_calls_groq_and_publishes(client, db_session, upload_dir):
    client.post(
        "/auth/register",
        json={
            "email": "automatic@university.edu",
            "password": "password123",
            "full_name": "Automatic User",
            "role": "student",
        },
    )
    login = client.post(
        "/auth/login",
        json={"email": "automatic@university.edu", "password": "password123"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]

    response = client.post(
        "/papers",
        data={"title": "Automatic processing test", "abstract": "Test abstract"},
        files={"file": ("paper.pdf", io.BytesIO(make_pdf_bytes("AUTOMATIC_TEXT_MARKER")), "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201, response.text
    paper_id = uuid.UUID(response.json()["id"])

    # The BackgroundTask completes before TestClient returns; refresh the
    # request-session identity map to observe its separate-session updates.
    db_session.expire_all()
    paper = db_session.query(Paper).filter(Paper.id == paper_id).one()
    assert paper.status == PaperStatus.PUBLISHED
    assert "AUTOMATIC_TEXT_MARKER" in paper.extracted_text

    topic_names = {
        row[0]
        for row in db_session.query(Topic.name)
        .join(PaperTopic, PaperTopic.topic_id == Topic.id)
        .filter(PaperTopic.paper_id == paper_id)
        .all()
    }
    assert "Test Topic" in topic_names
    assert "Machine Learning" in topic_names
