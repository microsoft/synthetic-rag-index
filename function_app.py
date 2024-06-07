# First imports, to make sure the following logs are first
from helpers.config import CONFIG
from helpers.logging import logger, APP_NAME, trace


logger.info(f"{APP_NAME} v{CONFIG.version}")


# General imports
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    ContentFormat,
    ParagraphRole,
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.storage.blob.aio import BlobClient, ContainerClient
from azurefunctions.extensions.bindings.blob import BlobClient as BlobClientTrigger
from helpers.http import azure_transport
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from os import getenv
from pydantic import TypeAdapter, ValidationError
from typing import Optional
import asyncio
import azure.functions as func
import hashlib
import json
import math
import nltk
import tiktoken
from helpers.models import (
    IndexedFactModel,
    LlmDocumentModel,
    RawTextModel,
    StoreDocumentModel,
)


# Azure Functions
app = func.FunctionApp()

# Storage Account
CONTAINER_NAME = "trainings"
EXTRACTED_FOLDER = "extracted"
FACT_FOLDER = "fact"
FILTERED = "filtered"
RAW_FOLDER = "raw"

# Clients
_container_client: Optional[ContainerClient] = None
_doc_client: Optional[DocumentIntelligenceClient] = None
_openai_client: Optional[AsyncAzureOpenAI] = None
_search_client: Optional[SearchClient] = None


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{RAW_FOLDER}/{{name}}",
)
async def raw_to_extracted(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing raw blob ({blob_name})")
        downloader = await blob_client.download_blob()
        content = await downloader.readall()  # TODO: Use stream (files can be large)
    # Analyze document
    logger.info(f"Analyzing document ({blob_name})")
    doc_client = await _use_doc_client()
    doc_poller = await doc_client.begin_analyze_document(
        analyze_request=content,  # type: ignore
        content_type="application/octet-stream",
        model_id="prebuilt-layout",
        output_content_format=ContentFormat.MARKDOWN,
    )
    doc_result: AnalyzeResult = await doc_poller.result()
    # Build cracked model
    title_paragraph = next(
        (
            paragraph
            for paragraph in doc_result.paragraphs or []
            if paragraph.role == ParagraphRole.TITLE
        ),
        None,
    )
    raw_text_model = RawTextModel(
        content=doc_result.content,
        file_path=blob_name,
        format="markdown",
        langs=[lang.locale for lang in doc_result.languages or []],
        title=title_paragraph.content if title_paragraph else None,
    )
    # Store
    out_path = _replace_root_path(
        _replace_extension(blob_name, ".json"), EXTRACTED_FOLDER
    )
    out_client = await _use_blob_async_client(out_path)
    await out_client.upload_blob(
        data=raw_text_model.model_dump_json(),
        overwrite=True,
    )


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{EXTRACTED_FOLDER}/{{name}}",
)
async def extracted_to_filtered(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing extracted blob ({blob_name})")
        downloader = await blob_client.download_blob()
        content = await downloader.readall()
    # Deserialize
    raw_text_model = RawTextModel.model_validate_json(content)
    # Free up memory
    del content
    # Remove repetitions
    text = raw_text_model.content
    if _is_repetition_removal(
        text=text,
        threshold_ratio=2.0,  # We are less strict than the paper because this is all normally internal data and we are not training a model
    ):
        logger.info(f"Repetition detected, skipping ({blob_name})")
        return
    # Clean
    text = _clean_page(text)
    if not text:
        logger.info(f"Page skipped ({blob_name})")
        return
    # Store
    out_path = _replace_root_path(_replace_extension(blob_name, ".json"), FILTERED)
    out_client = await _use_blob_async_client(out_path)
    await out_client.upload_blob(
        data=raw_text_model.model_dump_json(),
        overwrite=True,
    )


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{FILTERED}/{{name}}",
)
async def filtered_to_fact(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing repetition-filtered blob ({blob_name})")
        downloader = await blob_client.download_blob()
        content = await downloader.readall()
    # Deserialize
    raw_text_model = RawTextModel.model_validate_json(content)
    # Prepare chunks for LLM
    contents = _split_text(
        text=raw_text_model.content,
        max_tokens=int(
            CONFIG.llm.context * 0.8
        ),  # For simplicity, we count tokens with a 20% margin
        max_chars=int(1048576 * 0.9),  # REST API has a limit of 1MB, with a 10% margin
    )
    logger.info(f"Splited to {len(contents)} parts ({blob_name})")
    # LLM does its magic
    await asyncio.gather(
        *[
            _llm_generate_synthetis(
                blob_name=_replace_extension(blob_name, f"-{i}.json"),
                content=content,
                format=raw_text_model.format,
                langs=raw_text_model.langs,
                openai_client=await _use_openai_client(),
                title=raw_text_model.title,
            )
            for i, content in enumerate(contents)
        ]
    )
    logger.info(f"Synthesises are generated and stored ({blob_name})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{FACT_FOLDER}/{{name}}",
)
async def fact_to_index(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing fact blob ({blob_name})")
        downloader = await blob_client.download_blob()
        content = await downloader.readall()
    # Deserialize
    store_model = StoreDocumentModel.model_validate_json(content)
    # Free up memory
    del content
    # Build indexed model
    indexed_models = [
        IndexedFactModel(
            answer=fact.answer,
            context=fact.context,
            document_synthesis=store_model.synthesis,
            file_path=store_model.file_path,
            id=_hash_text(f"{store_model.file_path}-{i}"),  # Reproducible ID
            question=fact.question,
        )
        for i, fact in enumerate(store_model.facts)
    ]
    # Index
    logger.info(f"Indexing {len(indexed_models)} documents to AI Search ({blob_name})")
    indexed_dicts = TypeAdapter(list[IndexedFactModel]).dump_python(
        indexed_models, mode="json"
    )
    search_client = await _use_search_client()
    await search_client.merge_or_upload_documents(indexed_dicts)


def _split_text(text: str, max_tokens: int, max_chars: int) -> list[str]:
    """
    Split a text into chunks of text with a maximum number of tokens and characters.

    The function returns a list of text chunks.
    """
    contents = []
    first_margin = 100
    last_margin = 100
    token_count = _count_tokens(content=text, model=CONFIG.llm.model)

    if token_count < max_tokens:  # For simplicity, we count tokens with a 10% margin
        contents.append(text)

    else:  # We split the document in chunks
        ckuncks_count = math.ceil(token_count / max_tokens)
        chunck_size = math.ceil(len(text) / ckuncks_count)
        if chunck_size > max_chars:  # Test if chunk size is too big for REST API
            ckuncks_count = math.ceil(token_count / max_chars)
            chunck_size = math.ceil(len(text) / ckuncks_count)
        for i in range(ckuncks_count):  # Iterate over desired chunks count
            start = max(i * chunck_size - first_margin, 0)  # First chunk with margin
            end = min(
                (i + 1) * chunck_size + last_margin, len(text)
            )  # Last chunk with margin
            contents.append(text[start:end])

    return contents


async def _llm_generate_synthetis(
    blob_name: str,
    content: str,
    format: str,
    langs: list[str],
    openai_client: AsyncAzureOpenAI,
    title: Optional[str],
    _previous_result: Optional[str] = None,
    _retries_remaining: int = 3,
    _validation_error: Optional[str] = None,
) -> None:
    """
    Generate a synthesis from a content using OpenAI.

    The synthesis is generated using the LLM model. Then, the synthesis is stored in the FACT folder.
    """
    logger.info(
        f"Generating synthesis ({blob_name}, {_retries_remaining} retries left)"
    )
    openai_client = await _use_openai_client()
    messages = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=f"""
            Assistant is an expert data analyst with 20 years of experience.

            # Objective
            Analyze the document and extract its details.

            # Rules
            - Answers in English, even if the document is in another language
            - Be exhaustive and complete
            - Only use the information provided in the document

            # Result format as a JSON schema
            {json.dumps(LlmDocumentModel.model_json_schema())}

            # Document metadata
            - Format: {format}
            - Lang: {", ".join(langs)}
            - Title: {title if title else "N/A"}

            # Document content
            {content}
            """,
        ),
    ]
    if _validation_error:
        messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content=f"""
                A validation error occurred during the previous attempt.

                # Previous result
                {_previous_result}

                # Error details
                {_validation_error}
                """,
            )
        )
    llm_res = await openai_client.chat.completions.create(
        messages=messages,
        model=CONFIG.llm.model,
        response_format={"type": "json_object"},
    )
    llm_content: str = llm_res.choices[0].message.content  # type: ignore
    # Parse LLM response
    try:
        llm_model = LlmDocumentModel.model_validate_json(llm_content)
    except ValidationError as e:
        if _retries_remaining == 0:
            raise e
        logger.warning(f"LLM validation error ({blob_name})")
        return await _llm_generate_synthetis(
            blob_name=blob_name,
            content=content,
            format=format,
            langs=langs,
            openai_client=openai_client,
            title=title,
            _previous_result=llm_content,
            _retries_remaining=_retries_remaining - 1,
            _validation_error=str(e),
        )
    # Build store model
    store_model = StoreDocumentModel(
        facts=llm_model.facts,
        file_path=blob_name,
        synthesis=llm_model.synthesis,
    )
    # Store
    out_path = _replace_root_path(blob_name, FACT_FOLDER)
    out_client = await _use_blob_async_client(out_path)
    await out_client.upload_blob(
        data=store_model.model_dump_json(),
        overwrite=True,
    )


def _clean_page(
    text: str,
    max_word_length: int = 1000,
    min_words_per_line: int = 5,
) -> Optional[str]:
    """
    Cleans text, return nothing if it should be skipped.

    Cleaning removes lines with no end marks or with too few words. After line filtering, pages are filtered out if they have too few sentences based on a simple count of end marks.

    This functions implement "Clean Crawled Corpus" from Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (https://arxiv.org/abs/1910.10683).

    Return the cleaned text.
    """
    ellipsis = "..."
    end_marks = (".", "?", "!", '"')
    policy_substrings = [
        "cookie policy",
        "privacy policy",
        "terms of use",
        "use cookies",
        "use of cookies",
        "uses cookies",
    ]

    lines = text.splitlines()  # Split by lines
    valid_lines = []

    def _line_has_too_long_word(line):
        """
        Check if a line contains a word that is too long.
        """
        for word in line.split():  # Split by whitespace
            if len(word) > max_word_length:  # Check if word is too long
                return True
        return False

    for line in lines:
        line = line.strip()
        if _line_has_too_long_word(line):  # Skip lines with too long words
            continue
        if not line.endswith(end_marks) or line.endswith(
            ellipsis
        ):  # Skip lines without end marks
            continue
        if len(line.split()) < min_words_per_line:  # Skip lines with too few words
            continue
        line_lower = line.lower()
        if "lorem ipsum" in line_lower:  # Skip entire page if it contains lorem ipsum
            logger.info("Lorem ipsum detected, skipping page")
            return
        if any(p in line_lower for p in policy_substrings):  # Skip policy lines
            continue
        valid_lines.append(line)

    if not valid_lines:  # Skip empty pages
        logger.info("Empty page, skipping")
        return

    logger.info(f"Page cleaned, {len(lines)/len(valid_lines):.2f}x reduction")
    return "\n".join(valid_lines).strip()


def _is_repetition_removal(
    text: str,
    threshold_ratio: float = 1.0,
) -> bool:
    """
    Check if there is repeated content in the input text. Excessive repetition is often linked with uninformative content and can be used to determine whether it is low-quality text.

    Threshold ratio is relative to the recommended values in the paper. The default value is 1.0, which corresponds to the recommended values.

    This function implements "Repetition Removal" from Scaling Language Models: Methods, Analysis & Insights from Training Gopher (https://arxiv.org/abs/2112.11446).

    Return True if the text is considered to have excessive repetition, False otherwise.
    """
    duplicate_line_character_faction = (
        0.2 * threshold_ratio
    )  # Duplicate line character fraction
    duplicate_line_fraction = 0.3 * threshold_ratio  # Duplicate line fraction

    dup_line = 0
    dup_line_chars = 0
    line_count = 0
    visit_lines = {}

    # Check for repeated lines
    for line in text.split("\n"):
        line_hash = _hash_text(line)
        if line_hash in visit_lines:
            dup_line += 1
            dup_line_chars += len(line)
        visit_lines[line_hash] = True
        line_count += 1

    if (
        float(dup_line) / line_count > duplicate_line_fraction
    ):  # Excessive repeated lines
        return True

    if (
        float(dup_line_chars) / len(text) > duplicate_line_character_faction
    ):  # Excessive repeated characters
        return True

    top_ngram_character_fractions = [
        (2, 0.2 * threshold_ratio),  # Top 2-gram character fraction
        (3, 0.18 * threshold_ratio),  # Top 3-gram character fraction
        (4, 0.16 * threshold_ratio),  # Top 4-gram character fraction
    ]
    for ngram, threshold in top_ngram_character_fractions:
        bgs = nltk.ngrams(text.split(), ngram)
        fdist = nltk.FreqDist(bgs)
        for word_list, repeat in fdist.items():
            char_count = sum([len(word) for word in word_list])
            if char_count * (repeat - 1) / len(text) > threshold:
                return True

    duplicate_ngram_character_fractions = [
        (5, 0.15 * threshold_ratio),  # Duplicate 5-gram character fraction
        (6, 0.14 * threshold_ratio),  # Duplicate 6-gram character fraction
        (7, 0.13 * threshold_ratio),  # Duplicate 7-gram character fraction
        (8, 0.12 * threshold_ratio),  # Duplicate 8-gram character fraction
        (9, 0.11 * threshold_ratio),  # Duplicate 9-gram character fraction
        (10, 0.10 * threshold_ratio),  # Duplicate 10-gram character fraction
    ]
    for ngram, threshold in duplicate_ngram_character_fractions:
        fdist = {}
        word_list = text.split()
        mark = [0] * len(word_list)
        for i in range(len(word_list) - ngram + 1):
            bag = tuple(word_list[i : i + ngram])
            if bag in fdist:
                for j in range(i, i + ngram):
                    mark[j] = len(word_list[j])
                fdist[bag] += 1
            else:
                fdist[bag] = 1

        if sum(mark) / float(len(text)) > threshold:
            return True

    return False


def _hash_text(text: str) -> str:
    """
    Hash a text using SHA-256.
    """
    return hashlib.sha256(
        string=text.encode(),
        usedforsecurity=False,
    ).hexdigest()


def _count_tokens(content: str, model: str) -> int:
    """
    Returns the number of tokens in the content, using the model's encoding.

    If the model is unknown to TikToken, it uses the GPT-3.5 encoding.
    """
    try:
        encoding_name = tiktoken.encoding_name_for_model(model)
    except KeyError:
        encoding_name = tiktoken.encoding_name_for_model("gpt-3.5")
        logger.warning(f"Unknown model {model}, using {encoding_name} encoding")
    return len(tiktoken.get_encoding(encoding_name).encode(content))


def _replace_root_path(file_path: str, new_root: str) -> str:
    """
    Replace the root path of a file path.

    For example, if the file path is "raw/2022-01-01/file.txt" and the new root is "fact", the new file path will be "fact/2022-01-01/file.txt".
    """
    return new_root + "/" + "".join(file_path.split("/")[1:])


def _replace_extension(file_path: str, new_extension: str) -> str:
    """
    Replace the extension of a file path.

    For example, if the file path is "file.txt" and the new extension is "json", the new file path will be "file.json".
    """
    return "".join(file_path.split(".")[:-1]) + new_extension


async def _use_blob_async_client(
    name: str,
    snapshot: Optional[str] = None,
) -> BlobClient:
    """
    Create a BlobClient client capable of async I/O.
    """
    BlobClient.from_blob_url
    container_client = await _use_container_async_client()
    return container_client.get_blob_client(
        blob=name,
        snapshot=snapshot,  # type: ignore
    )


async def _use_container_async_client() -> ContainerClient:
    """
    Create a ContainerClient client capable of async I/O.

    The client is created using the AzureWebJobsStorage env var. The client is cached for future use.
    """
    global _container_client
    if not isinstance(_container_client, ContainerClient):
        connection_string: str = getenv("AzureWebJobsStorage")  # type: ignore
        _container_client = ContainerClient.from_connection_string(
            conn_str=connection_string,
            container_name=CONTAINER_NAME,
        )
    return _container_client


async def _use_doc_client() -> DocumentIntelligenceClient:
    """
    Create a DocumentIntelligenceClient client capable of async I/O.

    The client is cached for future use.
    """
    global _doc_client
    if not isinstance(_doc_client, DocumentIntelligenceClient):
        _doc_client = DocumentIntelligenceClient(
            # Deployment
            endpoint=CONFIG.document_intelligence.endpoint,
            # Performance
            transport=await azure_transport(),
            # Authentication
            credential=AzureKeyCredential(
                CONFIG.document_intelligence.access_key.get_secret_value()
            ),
        )
    return _doc_client


async def _use_search_client() -> SearchClient:
    """
    Create a SearchClient client capable of async I/O.

    The client is cached for future use.
    """
    global _search_client
    if not isinstance(_search_client, SearchClient):
        _search_client = SearchClient(
            # Deployment
            endpoint=CONFIG.ai_search.endpoint,
            index_name=CONFIG.ai_search.index,
            # Performance
            transport=await azure_transport(),
            # Authentication
            credential=AzureKeyCredential(
                CONFIG.ai_search.access_key.get_secret_value()
            ),
        )
    return _search_client


async def _use_openai_client() -> AsyncAzureOpenAI:
    """
    Create a OpenAI client capable of async I/O.

    The client is cached for future use.
    """
    global _openai_client
    if not isinstance(_openai_client, AsyncAzureOpenAI):
        _openai_client = AsyncAzureOpenAI(
            # Deployment
            api_version="2023-12-01-preview",
            azure_deployment=CONFIG.llm.deployment,
            azure_endpoint=CONFIG.llm.endpoint,
            # Reliability
            max_retries=30,  # We are patient, this is a background job :)
            timeout=180,  # 3 minutes
            # Authentication
            api_key=CONFIG.llm.api_key.get_secret_value(),
        )
    return _openai_client
