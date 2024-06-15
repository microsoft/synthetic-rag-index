from abc import abstractmethod
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from helpers.config_models.llm import AzureOpenaiPlatformModel, OpenaiPlatformModel
from helpers.logging import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from persistence.illm import ILlm
from typing import Optional, TypeVar, Callable, Union
import tiktoken


T = TypeVar("T")


class AbstractOpenaiLlm(ILlm):
    _client: Optional[AsyncAzureOpenAI] = None
    _config: Union[AzureOpenaiPlatformModel, OpenaiPlatformModel]

    def __init__(self, config: Union[AzureOpenaiPlatformModel, OpenaiPlatformModel]):
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

        if self.count_tokens(text) < max_tokens and len(text) < max_chars:  # If the text is small enough
            contents.append(text)
            return contents

        # Split the text by Markdown headings
        h1_title = ""
        h2_title = ""
        h3_title = ""
        h4_title = ""
        headings: dict[dict[dict[dict[str, str], str], str], str] = {}
        for line in text.splitlines():
            if line.startswith("# "):
                h1_title = line[2:]
                h2_title = ""
                h3_title = ""
                h4_title = ""
                continue
            if line.startswith("## "):
                h2_title = line[3:]
                h3_title = ""
                h4_title = ""
                continue
            if line.startswith("### "):
                h3_title = line[4:]
                h4_title = ""
                continue
            if line.startswith("#### "):
                h4_title = line[5:]
                continue
            if not line:
                continue
            if h1_title not in headings:
                headings[h1_title] = {}
            if h2_title not in headings[h1_title]:
                headings[h1_title][h2_title] = {}
            if h3_title not in headings[h1_title][h2_title]:
                headings[h1_title][h2_title][h3_title] = {}
            if h4_title not in headings[h1_title][h2_title][h3_title]:
                headings[h1_title][h2_title][h3_title][h4_title] = ""
            headings[h1_title][h2_title][h3_title][h4_title] += line + "\n"

        # Split document into the biggest chunks possible
        current_chunk = ""
        for h1_head, h1_next in headings.items():
            if h1_head:
                current_chunk += f"# {h1_head}\n"
            for h2_head, h2_next in h1_next.items():
                if h2_head:
                    current_chunk += f"## {h2_head}\n"
                for h3_head, h3_next in h2_next.items():
                    if h3_head:
                        current_chunk += f"### {h3_head}\n"
                    for h4_head, h4_content in h3_next.items():
                        if self.count_tokens(current_chunk) >= max_tokens or len(current_chunk) >= max_chars:  # If the chunk is too big
                            to_remove = 0
                            previous_lines = current_chunk.splitlines()
                            previous_lines.reverse()
                            for previous_line in previous_lines:
                                if not previous_line.startswith("#"):
                                    break
                                to_remove += 1
                            contents.append("\n".join(current_chunk.splitlines()[:-(to_remove + 1)]).strip())  # Remove the last heading
                            # Re-apply the last heading to the next chunk
                            current_chunk = ""
                            if h1_head:
                                current_chunk = f"# {h1_head}\n"
                            if h2_head:
                                current_chunk += f"## {h2_head}\n"
                            if h3_head:
                                current_chunk += f"### {h3_head}\n"
                        if h4_content:
                            if h4_head:
                                current_chunk += f"#### {h4_head}\n"
                            current_chunk += h4_content + "\n"
        # Add the last chunk
        if current_chunk:
            contents.append(current_chunk.strip())

        # Return the chunks
        return contents

    @abstractmethod
    def _use_client(self) -> AsyncOpenAI:
        pass


class AzureOpenaiLlm(AbstractOpenaiLlm):
    def __init__(self, config: AzureOpenaiPlatformModel):
        super().__init__(config)

    def _use_client(self) -> AsyncAzureOpenAI:
        if not self._client:
            api_key = self._config.api_key.get_secret_value() if self._config.api_key else None
            token_func = (
                get_bearer_token_provider(
                    ManagedIdentityCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                if not self._config.api_key
                else None
            )
            self._client = AsyncAzureOpenAI(
                # Deployment
                api_version="2023-12-01-preview",
                azure_deployment=self._config.deployment,
                azure_endpoint=self._config.endpoint,
                # Reliability
                max_retries=30,  # We are patient, this is a background job :)
                timeout=180,  # 3 minutes
                # Authentication
                api_key=api_key,
                azure_ad_token_provider=token_func,
            )
        return self._client


class OpenaiLlm(AbstractOpenaiLlm):
    def __init__(self, config: OpenaiPlatformModel):
        super().__init__(config)

    def _use_client(self) -> AsyncOpenAI:
        if not self._client:
            self._client = AsyncOpenAI(
                # Reliability
                max_retries=30,  # We are patient, this is a background job :)
                timeout=180,  # 3 minutes
                # Authentication
                api_key=self._config.api_key.get_secret_value(),
            )
        return self._client
