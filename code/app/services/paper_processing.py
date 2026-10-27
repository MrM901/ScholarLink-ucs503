"""In-process background worker for uploaded-paper processing.

FastAPI BackgroundTasks is sufficient for this project's current single-process
prototype. The function opens its own database session because a request-scoped
session must not be reused after the HTTP response has been produced.
"""
import logging
import threading
import uuid

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.enums import PaperStatus
from app.core.storage import UploadStorage
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.user_repository import UserRepository
from app.services.groq_service import GroqService
from app.services.paper_service import PaperService

logger = logging.getLogger(__name__)
_in_flight_lock = threading.Lock()
_in_flight_papers: set[str] = set()


def process_paper_in_background(paper_id: str) -> None:
    """Extract PDF text, call Groq, and save topics without a manual API call."""
    key = str(paper_id)
    try:
        parsed_id = uuid.UUID(key)
    except (ValueError, TypeError, AttributeError):
        logger.error("Cannot process paper: invalid paper UUID %r", paper_id)
        return

    # Avoid duplicate Groq calls for the same paper inside one application
    # process (for example, when the client resubmits while work is in flight).
    with _in_flight_lock:
        if key in _in_flight_papers:
            logger.info("Paper %s is already being processed; skipping duplicate task", key)
            return
        _in_flight_papers.add(key)

    db = None
    try:
        db = SessionLocal()
        paper_repository = PaperRepository(db)
        paper = paper_repository.get_by_id(parsed_id)
        if paper is None:
            logger.warning("Paper %s disappeared before processing started", key)
            return
        if paper.status != PaperStatus.PROCESSING:
            logger.info("Paper %s is already processed; skipping", key)
            return

        service = PaperService(
            paper_repository=paper_repository,
            user_repository=UserRepository(db),
            topic_repository=TopicRepository(db),
            storage=UploadStorage(settings.UPLOAD_DIR),
            groq_service=GroqService(),
        )
        service.extract_text(parsed_id)
        logger.info("Automatic PDF/Groq processing completed for paper %s", key)
    except Exception:
        if db is not None:
            db.rollback()
        # Keep the paper in PROCESSING so the existing manual endpoint can be
        # used for a retry. Do not fail an upload whose file/record was saved.
        logger.exception("Automatic PDF/Groq processing failed for paper %s", key)
    finally:
        if db is not None:
            db.close()
        with _in_flight_lock:
            _in_flight_papers.discard(key)
