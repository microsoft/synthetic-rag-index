from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


class ILlm(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        res_type: type[T],
        validation_callback: Callable[
            [Optional[str]], tuple[bool, Optional[str], Optional[T]]
        ],
        max_tokens: Optional[int] = None,
        temperature: float = 0,
        validate_json: bool = False,
        _previous_result: Optional[str] = None,
        _retries_remaining: Optional[int] = None,
        _validation_error: Optional[str] = None,
    ) -> Optional[T]:
        """
        Generate a completion from a prompt using OpenAI.

        The completion is generated using the LLM model.
        """
        pass

    @abstractmethod
    def count_tokens(self, content: str) -> int:
        """
        Returns the number of tokens in the content, using the model's encoding.
        """
        pass

    @abstractmethod
    def chunck(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Split a text into chunks of text with a maximum number of tokens and characters.

        The function returns a list of text chunks.
        """
        pass
