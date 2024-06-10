from pydantic import SecretStr, BaseModel


class DocumentIntelligenceModel(BaseModel):
    access_key: SecretStr
    endpoint: str
