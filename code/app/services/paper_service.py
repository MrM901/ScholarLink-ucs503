"""
Business logic for the paper upload pipeline. No direct DB access (that's
the repository's job) and no HTTP/FastAPI concerns (that's the route's job).
"""
import uuid

from app.core.enums import PaperStatus
from app.core.exceptions import PaperFileMissingError, PaperNotFoundError
from app.core.storage import UploadStorage
from app.models.paper import Paper
from app.models.user import User
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.user_repository import UserRepository
from app.services.groq_service import GroqService
from app.services.pdf_extraction import extract_pdf_text

DEFAULT_DEDUPE_WINDOW_SECONDS = 10


class PaperService:
    def __init__(
        self,
        paper_repository: PaperRepository,
        user_repository: UserRepository,
        storage: UploadStorage,
        topic_repository: TopicRepository,
        groq_service: GroqService,
    ):
        self.papers = paper_repository
        self.users = user_repository
        self.storage = storage
        self.topics = topic_repository
        self.groq = groq_service

    def create_paper(
        self,
        *,
        uploader: User,
        title: str,
        abstract: str,
        filename: str | None,
        file_bytes: bytes,
        co_author_ids: list[uuid.UUID] | None = None,
        dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
    ) -> Paper:
        # Protect against double-submit and avoid creating duplicate records/files.
        existing = self.papers.find_recent_by_uploader_and_title(
            uploader.id, title, dedupe_window_seconds
        )
        if existing is not None:
            return existing

        storage_key = self.storage.save(file_bytes, filename)
        paper = self.papers.create(
            title=title,
            abstract=abstract,
            file_storage_key=storage_key,
            uploaded_by_id=uploader.id,
        )

        # The uploader is always author #0.
        self.papers.add_author_link(paper.id, uploader.id, author_order=0)
        order = 1
        for co_author_id in co_author_ids or []:
            if co_author_id == uploader.id:
                continue
            co_author = self.users.get_by_id(co_author_id)
            if co_author is None:
                continue
            self.papers.add_author_link(paper.id, co_author.id, author_order=order)
            order += 1
        return self.papers.get_by_id(paper.id)

    def extract_text(self, paper_id: uuid.UUID) -> Paper:
        """Extract PDF text, persist Groq topics, and publish only on success.

        This method is idempotent for repeated calls: it updates extracted_text
        and uses repository get-or-create operations for topics and paper links.
        """
        paper = self.papers.get_by_id(paper_id)
        if paper is None:
            raise PaperNotFoundError(f"No paper with id {paper_id}")
        if not paper.file_storage_key or not self.storage.exists(paper.file_storage_key):
            raise PaperFileMissingError(f"No stored file for paper {paper_id}")

        file_bytes = self.storage.read(paper.file_storage_key)
        text = extract_pdf_text(file_bytes)
        if not text.strip():
            raise ValueError(
                "No selectable text was found in this PDF. Scanned PDFs need OCR before topic extraction."
            )

        # Save the extracted text before the network call so it remains useful
        # for diagnostics/retry even if the Groq API is temporarily unavailable.
        self.papers.update_extracted_text(paper.id, text)

        topics = self.groq.extract_topics(text)
        if not topics:
            raise RuntimeError("Groq did not return any usable research topics for this paper.")

        for topic_name in topics:
            normalized_name = " ".join(topic_name.lower().split())
            topic = self.topics.get_by_normalized_name(normalized_name)
            if topic is None:
                topic = self.topics.create(
                    name=topic_name,
                    normalized_name=normalized_name,
                )
            self.topics.add_paper_link(paper_id=paper.id, topic_id=topic.id)

        # Publish only after the model response and topic persistence succeeded.
        self.papers.set_status(paper.id, PaperStatus.PUBLISHED)
        return self.papers.get_by_id(paper.id)
