from enum import Enum
from functools import cache
from typing import Optional, Union
from persistence.illm import ILlm
from pydantic import Field, SecretStr, BaseModel, field_validator, ValidationInfo


class ModeEnum(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


class AbstractPlatformModel(BaseModel, frozen=True):
    context: int
    model: str
    streaming: bool
    validation_retry_max: int = Field(
        default=3,
        ge=0,
    )
    page_split_margin: int = Field(
        default=100,
        ge=0,
    )


class AzureOpenaiPlatformModel(AbstractPlatformModel, frozen=True):
    api_key: Optional[SecretStr] = None
    deployment: str
    endpoint: str

    @cache
    def instance(self) -> ILlm:
        from persistence.openai import AzureOpenaiLlm

        return AzureOpenaiLlm(self)


class OpenaiPlatformModel(AbstractPlatformModel, frozen=True):
    api_key: SecretStr
    endpoint: str

    @cache
    def instance(self) -> ILlm:
        from persistence.openai import OpenaiLlm

        return OpenaiLlm(self)


class SelectedPlatformModel(BaseModel):
    azure_openai: Optional[AzureOpenaiPlatformModel] = None
    mode: ModeEnum
    openai: Optional[OpenaiPlatformModel] = None

    @field_validator("azure_openai")
    def _validate_azure_openai(
        cls,
        azure_openai: Optional[AzureOpenaiPlatformModel],
        info: ValidationInfo,
    ) -> Optional[AzureOpenaiPlatformModel]:
        if not azure_openai and info.data.get("mode", None) == ModeEnum.AZURE_OPENAI:
            raise ValueError("Azure OpenAI config required")
        return azure_openai

    @field_validator("openai")
    def _validate_openai(
        cls,
        openai: Optional[OpenaiPlatformModel],
        info: ValidationInfo,
    ) -> Optional[OpenaiPlatformModel]:
        if not openai and info.data.get("mode", None) == ModeEnum.OPENAI:
            raise ValueError("OpenAI config required")
        return openai

    def selected(self) -> Union[AzureOpenaiPlatformModel, OpenaiPlatformModel]:
        platform = (
            self.azure_openai if self.mode == ModeEnum.AZURE_OPENAI else self.openai
        )
        assert platform
        return platform


class LlmModel(BaseModel):
    fast: SelectedPlatformModel
    slow: SelectedPlatformModel

    def selected(
        self, is_fast: bool
    ) -> Union[AzureOpenaiPlatformModel, OpenaiPlatformModel]:
        platform = self.fast if is_fast else self.slow
        return platform.selected()
