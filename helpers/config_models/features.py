from pydantic import BaseModel, Field


class FeaturesModel(BaseModel):
    sanitize_pdf_version: bool = Field(
        default="1.5",  # v1.5 allows JPEG 2000 which can reduce file size by 50%
    )
    fact_iterations: int = Field(
        default=10,
        ge=1,
    )
    fact_score_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )
    page_split_size: int = Field(
        default=int(100 / 75 * 500),  # 100 tokens ~= 75 words, ~500 words per page for a dense book
        ge=0,
    )
