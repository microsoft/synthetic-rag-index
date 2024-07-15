from functools import cache

from pydantic import BaseModel, Field, SecretStr

from persistence.ianalyze import IAnalyze


class DocumentIntelligenceModel(BaseModel, frozen=True):
    access_key: SecretStr
    endpoint: str
    extract_lang_threshold: float = Field(
        default=0.75,
        ge=0,
        le=1,
    )
    pdf_pages_max: int = 2000

    @cache
    def instance(self) -> IAnalyze:
        from persistence.document_intelligence import DocumentIntelligenceAnalyze

        return DocumentIntelligenceAnalyze(self)
