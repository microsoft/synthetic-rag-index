from abc import ABC, abstractmethod
from typing import IO, Generator, Optional


class IAnalyze(ABC):
    @abstractmethod
    async def analyze(
        self,
        document: IO[bytes],
        file_name: str,
    ) -> tuple[str, Optional[str], list[str]]:
        """
        Analyze a document and return the content and title.

        The content is the main text of the document. The title is the main title of the document. If no title is found, it will be None.

        Returns a tuple of content, title and langs.
        """
        pass

    @abstractmethod
    def chunck(self, pages_count: int) -> Generator[tuple[list[int], int, int], None, None]:
        """
        Split the pages into chunks of the maximum size allowed by the service.

        Yields a tuple of pages, start and chunck count.
        """
        pass

    @abstractmethod
    def compatible_formats(self) -> set[str]:
        pass
