from pydantic import SecretStr, BaseModel


class LlmModel(BaseModel):
    api_key: SecretStr
    context: int
    deployment: str
    endpoint: str
    model: str
