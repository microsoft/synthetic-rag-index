from enum import Enum
from functools import cache
from persistence.illm import ILlm
from pydantic import Field, SecretStr, BaseModel


class TypeEnum(Enum):
    FAST = "fast"
    SLOW = "slow"


class BackendModel(BaseModel, frozen=True):
    api_key: SecretStr
    context: int
    deployment: str
    endpoint: str
    model: str
    type: TypeEnum
    validation_retry_max: int = Field(
        default=3,
        ge=0,
    )
    page_split_margin: int = Field(
        default=100,
        ge=0,
    )

    @cache
    def instance(self) -> ILlm:
        from persistence.azure_openai import AzureOpenaiLlm

        return AzureOpenaiLlm(self)


class FastModel(BackendModel):
    type: TypeEnum = TypeEnum.FAST


class SlowModel(BackendModel):
    type: TypeEnum = TypeEnum.SLOW


class LlmModel(BaseModel):
    fast: FastModel
    slow: SlowModel

    def instance(self, is_fast: bool) -> ILlm:
        if is_fast:
            return self.fast.instance()
        return self.slow.instance()
