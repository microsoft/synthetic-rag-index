from pydantic import BaseModel, Field


class FeaturesModel(BaseModel):
    sanitize_pdf_version: bool = Field(
        default="1.4",
        description="PDF specification version to use when sanitizing PDFs.",
    )
    extract_lang_confidence_threshold: float = Field(
        default=0.75,
        description="The minimum confidence level required to note a language as detected.",
        ge=0,
        le=1,
    )
    fact_iterations: int = Field(
        default=10,
        description="The number of iterations to run the fact extraction process.",
        ge=1,
    )
    fact_score_threshold: float = Field(
        default=0.5,
        description="The minimum score a fact must have to be considered valid.",
        ge=0,
        le=1,
    )
    page_split_size: int = Field(
        default=int(100 / 75 * 500),  # 100 tokens ~= 75 words, ~500 words per page for a dense book
        description="The maximum number of characters to allow on a single page.",
        ge=0,
    )
    page_split_margin: int = Field(
        default=100,
        description="The margin in characters to use when splitting pages.",
        ge=0,
    )
    llm_retry_count: int = Field(
        default=3,
        description="The number of times to retry a failed LLM request. This includes initial request and validation.",
        ge=0,
    )
