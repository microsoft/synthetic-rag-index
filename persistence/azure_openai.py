from helpers.config_models.llm import BackendModel as LlmBackendModel
from helpers.logging import logger
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from persistence.illm import ILlm
from typing import Optional, TypeVar, Callable
import math
import tiktoken


T = TypeVar("T")


class AzureOpenaiLlm(ILlm):
    _client: Optional[AsyncAzureOpenAI] = None
    _config: LlmBackendModel

    def __init__(self, config: LlmBackendModel):
        self._config = config

    async def generate(
        self,
        prompt: str,
        res_object: type[T],
        validation_callback: Callable[[Optional[str]], tuple[bool, Optional[str], Optional[T]]],
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        _previous_result: Optional[str] = None,
        _retries_remaining: Optional[int] = None,
        _validation_error: Optional[str] = None,
    ) -> Optional[T]:
        logger.info("LLM completion generation")

        # Initialize retries
        if _retries_remaining is None:
            _retries_remaining = self._config.validation_retry_max

        # Initialize prompts
        messages = [
            ChatCompletionSystemMessageParam(
                content=prompt,
                role="system",
            ),
        ]
        if _validation_error:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"""
                    A validation error occurred during the previous attempt.

                    # Previous result
                    {_previous_result or "N/A"}

                    # Error details
                    {_validation_error}
                    """,
                )
            )

        # Generate
        client = self._use_client()
        res = await client.chat.completions.create(
            max_tokens=max_tokens,
            messages=messages,
            model=self._config.model,
            temperature=temperature,
        )
        res_content = res.choices[0].message.content  # type: ignore

        # Validate
        is_valid, validation_error, res_object = validation_callback(res_content)
        if not is_valid:  # Retry if validation failed
            if _retries_remaining == 0:
                logger.error(f"LLM validation error: {validation_error}")
                return None
            logger.warning(f"LLM validation error, retrying ({_retries_remaining} retries left)")
            return await self.generate(
                max_tokens=max_tokens,
                prompt=prompt,
                res_object=res_object,
                temperature=temperature,
                validation_callback=validation_callback,
                _previous_result=res_content,
                _retries_remaining=_retries_remaining - 1,
                _validation_error=validation_error,
            )

        # Return after validation or if failed too many times
        return res_object

    def count_tokens(self, content: str) -> int:
        model = self._config.model
        try:
            encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            encoding_name = tiktoken.encoding_name_for_model("gpt-3.5")
            logger.debug(f"Unknown model {model}, using {encoding_name} encoding")
        return len(tiktoken.get_encoding(encoding_name).encode(content))

    def chunck(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        contents = []
        max_chars = int(1048576 * 0.9)  # REST API has a limit of 1MB, with a 10% margin
        if not max_tokens:  # For simplicity, we count tokens with a 20% marginμ
            max_tokens=int(self._config.context * 0.8)

        token_count = self.count_tokens(content=text)

        if token_count < max_tokens:  # If the document is small enough, we don't split it
            contents.append(text)
            return contents

        ckuncks_count = math.ceil(token_count / max_tokens)
        chunck_size = math.ceil(len(text) / ckuncks_count)
        if chunck_size > max_chars:  # Test if chunk size is too big for REST API
            ckuncks_count = math.ceil(token_count / max_chars)
            chunck_size = math.ceil(len(text) / ckuncks_count)
        for i in range(ckuncks_count):  # Iterate over desired chunks count
            start = max(i * chunck_size - self._config.page_split_margin, 0)  # First chunk with margin
            end = min(
                (i + 1) * chunck_size + self._config.page_split_margin, len(text)
            )  # Last chunk with margin
            contents.append(text[start:end])
        return contents

    def _use_client(self) -> AsyncAzureOpenAI:
        if not self._client:
            self._client = AsyncAzureOpenAI(
                # Deployment
                api_version="2023-12-01-preview",
                azure_deployment=self._config.deployment,
                azure_endpoint=self._config.endpoint,
                # Reliability
                max_retries=30,  # We are patient, this is a background job :)
                timeout=180,  # 3 minutes
                # Authentication
                api_key=self._config.api_key.get_secret_value(),
            )
        return self._client
