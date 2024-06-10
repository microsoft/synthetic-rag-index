from pydantic import SecretStr, BaseModel


class AiSearchModel(BaseModel):
    access_key: SecretStr
    endpoint: str
    index: str
