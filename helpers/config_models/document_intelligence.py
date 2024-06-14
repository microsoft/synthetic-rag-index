from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from helpers.http import azure_transport
from pydantic import SecretStr, BaseModel
from typing import Optional


class DocumentIntelligenceModel(BaseModel, frozen=True):
    _client: Optional[DocumentIntelligenceClient] = None
    access_key: SecretStr
    endpoint: str

    async def instance(self) -> DocumentIntelligenceClient:
        if not self._client:
            self._client = DocumentIntelligenceClient(
                # Deployment
                endpoint=self.endpoint,
                # Performance
                polling_interval=5,  # 5 seconds
                transport=await azure_transport(),
                # Authentication
                credential=AzureKeyCredential(
                    self.access_key.get_secret_value()
                ),
            )
        return self._client
