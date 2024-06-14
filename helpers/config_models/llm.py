from enum import Enum
from openai import AsyncAzureOpenAI
from pydantic import SecretStr, BaseModel
from typing import Optional


class TypeEnum(Enum):
    FAST = "fast"
    SLOW = "slow"


class ConfigModel(BaseModel, frozen=True):
    _client: Optional[AsyncAzureOpenAI] = None
    api_key: SecretStr
    context: int
    deployment: str
    endpoint: str
    model: str
    type: TypeEnum

    async def instance(self) -> tuple[AsyncAzureOpenAI, "ConfigModel"]:
        if not self._client:
            self._client = AsyncAzureOpenAI(
                # Deployment
                api_version="2023-12-01-preview",
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                # Reliability
                max_retries=30,  # We are patient, this is a background job :)
                timeout=180,  # 3 minutes
                # Authentication
                api_key=self.api_key.get_secret_value(),
            )
        return self._client, self


class FastModel(ConfigModel):
    type: TypeEnum = TypeEnum.FAST


class SlowModel(ConfigModel):
    type: TypeEnum = TypeEnum.SLOW


class LlmModel(BaseModel):
    fast: FastModel
    slow: SlowModel
