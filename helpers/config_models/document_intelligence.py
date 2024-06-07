from pydantic import SecretStr, BaseModel


class DocumentIntelligenceModel(BaseModel, frozen=True):
    access_key: SecretStr
    endpoint: str
