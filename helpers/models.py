from datetime import datetime, UTC
from pydantic import BaseModel, Field
from typing import Optional


class RawTextModel(BaseModel):
    # Immutable fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    file_path: str = Field(frozen=True)
    # Editable fields
    content: str
    format: str
    langs: list[str]
    title: Optional[str]


class LlmFactModel(BaseModel):
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


class LlmDocumentModel(BaseModel):
    # Editable fields
    facts: list[LlmFactModel] = Field(
        description="List of individual facts extracted from the document. It is crutial to provide as many facts as possible.",
    )
    synthesis: str = Field(
        description="Synthesis of the document. Should be in a few sentences. Anyone should be able to understand it without reading the document. Generally, it should be a summary or draw the document objectives.",
    )


class StoreDocumentModel(LlmDocumentModel):
    # Immutable fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    file_path: str = Field(frozen=True)


class IndexedFactModel(BaseModel):
    # Immutable fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    id: str = Field(frozen=True)
    # Editable fields
    answer: str
    context: str
    document_synthesis: str
    file_path: str
    question: str
