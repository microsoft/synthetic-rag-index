from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.search.documents.aio import SearchClient
from pydantic import TypeAdapter

from helpers.config_models.destination import AiSearchModel
from helpers.http import azure_transport
from helpers.logging import logger
from helpers.models import IndexedDocumentModel
from persistence.iindex import IIndex


class AISearchIndex(IIndex):
    _client: Optional[SearchClient] = None
    _config: AiSearchModel

    def __init__(self, config: AiSearchModel):
        self._config = config

    async def index(self, documents: list[IndexedDocumentModel]) -> bool:
        logger.info(f"Indexing {len(documents)} documents to AI Search")
        try:
            async with await self._use_client() as client:
                document_dicts = TypeAdapter(list[IndexedDocumentModel]).dump_python(
                    documents, mode="json"
                )
                await client.merge_or_upload_documents(
                    document_dicts
                )  # Will overwrite existing documents
        except HttpResponseError as e:
            logger.error(f"Error requesting AI Search: {e}")
        except ServiceRequestError as e:
            logger.error(f"Error connecting to AI Search: {e}")

    async def _use_client(self) -> SearchClient:
        if not self._client:
            self._client = SearchClient(
                # Deployment
                endpoint=self._config.endpoint,
                index_name=self._config.index,
                # Performance
                transport=await azure_transport(),
                # Authentication
                credential=AzureKeyCredential(
                    self._config.access_key.get_secret_value()
                ),
            )
        return self._client
