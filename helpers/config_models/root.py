from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from helpers.config_models.destination import DestinationModel
from helpers.config_models.document_intelligence import DocumentIntelligenceModel
from helpers.config_models.features import FeaturesModel
from helpers.config_models.llm import LlmModel
from helpers.config_models.monitoring import MonitoringModel


class RootModel(BaseSettings):
    # Pydantic settings
    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_prefix="",
    )

    # Immutable fields
    version: str = Field(default="0.0.0-unknown", frozen=True)
    # Editable fields
    destination: DestinationModel
    document_intelligence: DocumentIntelligenceModel
    features: FeaturesModel = FeaturesModel()  # Object is fully defined by default
    llm: LlmModel
    monitoring: MonitoringModel = (
        MonitoringModel()
    )  # Object is fully defined by default
