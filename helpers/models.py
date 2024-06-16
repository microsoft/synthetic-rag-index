from datetime import datetime, UTC
from pydantic import BaseModel, Field
from typing import Optional


class AbstractdDocumentModel(BaseModel):
    # Immutable fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    file_md5: str = Field(frozen=True)
    file_path: str = Field(frozen=True)
    # Editable fields
    format: str
    langs: Optional[set[str]]
    title: Optional[str]


class ExtractedDocumentModel(AbstractdDocumentModel):
    # Editable fields
    document_content: str


class ChunkedDocumentModel(AbstractdDocumentModel):
    # Editable fields
    chunk_content: str
    chunk_number: int


class SynthetisedDocumentModel(ChunkedDocumentModel):
    """
    Third, chunks are synthesised into a coherent text.
    """
    # Editable fields
    synthesis: str


class PagedDocumentModel(SynthetisedDocumentModel):
    # Editable fields
    page_content: str
    page_number: int


class FactModel(BaseModel):
    # Editable fields
    question: str = Field(
        description="Question or problem of the fact. Should be in a few sentences, and end with a question. The question should avoid any ambiguity. Jargon or technical terms can be used if necessary.",
    )
    answer: str = Field(
        description="Direct answer to the question. Be as precise as possible. Jargon or technical terms can be used if necessary. The answer must should be a fact, not an opinion or interpretation.",
    )
    context: str = Field(
        description="Context of the fact. Should be in a few sentences. Should add more information to the answer by providing insights or details. Insights can be related to and not limited: source, author, date, position, political context, technical context, etc.",
    )


class FactedLlmModel(BaseModel):
    # Editable fields
    facts: list[FactModel]


class FactedDocumentModel(FactedLlmModel, PagedDocumentModel):
    """
    Fourth, facts are synthetises from the synthesised text.
    """
    pass


class IndexedDocumentModel(BaseModel):
    """
    Last, the document is indexed to be searchable by the user.
    """
    # Immutable fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    id: str = Field(frozen=True)
    # Editable fields
    answer: str
    context: str
    document_synthesis: str
    file_path: str
    question: str
