from pydantic import SecretStr, BaseModel


class AiSearchModel(BaseModel, frozen=True):
    access_key: SecretStr
    endpoint: str
    index: str
