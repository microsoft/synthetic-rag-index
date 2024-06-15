from abc import ABC, abstractmethod
from helpers.models import IndexedDocumentModel


class IIndex(ABC):
    @abstractmethod
    async def index(self, documents: list[IndexedDocumentModel]) -> bool:
        """
        Index a list of documents.

        Returns True if the operation was successful, False otherwise.
        """
        pass
