from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from enum import Enum
from helpers.http import azure_transport
from pydantic import SecretStr, BaseModel, ValidationInfo, field_validator
from typing import Optional



class ModeEnum(Enum):
    AI_SEARCH = "ai_search"


class AiSearchModel(BaseModel, frozen=True):
    _client: Optional[SearchClient] = None
    access_key: SecretStr
    endpoint: str
    index: str

    async def instance(self) -> SearchClient:
        if not self._client:
            self._client = SearchClient(
                # Deployment
                endpoint=self.endpoint,
                index_name=self.index,
                # Performance
                transport=await azure_transport(),
                # Authentication
                credential=AzureKeyCredential(
                    self.access_key.get_secret_value()
                ),
            )
        return self._client


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
