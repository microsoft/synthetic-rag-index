from pydantic import SecretStr, BaseModel


class ConfigModel(BaseModel):
    api_key: SecretStr
    context: int
    deployment: str
    endpoint: str
    model: str


class LlmModel(BaseModel):
    fast: ConfigModel
    slow: ConfigModel
