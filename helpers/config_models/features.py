from pydantic import BaseModel, Field


class FeaturesModel(BaseModel):
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
        default=int(
            100 / 75 * 500 * 3
        ),  # 100 tokens ~= 75 words, ~500 words per page for a dense English book, 3 pages
        ge=0,
    )
