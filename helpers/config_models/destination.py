from enum import Enum
from functools import cache
from persistence.iindex import IIndex
from pydantic import SecretStr, BaseModel, ValidationInfo, field_validator
from typing import Optional


class ModeEnum(Enum):
    AI_SEARCH = "ai_search"


class AiSearchModel(BaseModel, frozen=True):
    access_key: SecretStr
    endpoint: str
    index: str

    @cache
    def instance(self) -> IIndex:
        from persistence.ai_search import AISearchIndex

        return AISearchIndex(self)


class DestinationModel(BaseModel):
    ai_search: AiSearchModel
    mode: ModeEnum

    @field_validator("ai_search")
    def _validate_sqlite(
        cls,
        ai_search: Optional[AiSearchModel],
        info: ValidationInfo,
    ) -> Optional[AiSearchModel]:
        if not ai_search and info.data.get("mode", None) == ModeEnum.AI_SEARCH:
            raise ValueError("AI Search config required")
        return ai_search

    def instance(self) -> IIndex:
        if self.mode == ModeEnum.AI_SEARCH:
            assert self.ai_search
            return self.ai_search.instance()
