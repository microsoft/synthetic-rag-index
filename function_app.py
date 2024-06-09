# First imports, to make sure the following logs are first
from helpers.config import CONFIG
from helpers.logging import logger, APP_NAME, trace


logger.info(f"{APP_NAME} v{CONFIG.version}")


# General imports
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    ContentFormat,
    DocumentAnalysisFeature,
    ParagraphRole,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.search.documents.aio import SearchClient
from azure.storage.blob import BlobProperties
from azure.storage.blob.aio import BlobClient, ContainerClient
from azurefunctions.extensions.bindings.blob import BlobClient as BlobClientTrigger
from helpers.http import azure_transport
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from os import getenv
from pydantic import TypeAdapter, ValidationError, BaseModel
from typing import Optional, Type, TypeVar
import asyncio
import azure.functions as func
import hashlib
import json
import math
import nltk
import tiktoken
from helpers.models import (
    ChunkedDocumentModel,
    ExtractedDocumentModel,
    FactedDocumentModel,
    FactedLlmModel,
    FactModel,
    IndexedDocumentModel,
    PagedDocumentModel,
    SynthetisedDocumentModel,
)
import re
import pikepdf
from io import BytesIO
from base64 import b64encode


# Azure Functions
app = func.FunctionApp()

# Storage Account
CHUNCK_FOLDER = "2-chunck"
CONTAINER_NAME = "trainings"
CRITIC_FOLDER = "6-critic"
EXTRACT_FOLDER = "1-extract"
FACT_FOLDER = "5-fact"
PAGE_FOLDER = "4-page"
RAW_FOLDER = "raw"
SANITIZE_FOLDER = "0-sanitize"
SYNTHESIS_FOLDER = "3-synthesis"

# Clients
_container_client: Optional[ContainerClient] = None
_doc_client: Optional[DocumentIntelligenceClient] = None
_openai_client: Optional[AsyncAzureOpenAI] = None
_search_client: Optional[SearchClient] = None

# Custom types
Model = TypeVar("Model", bound=BaseModel)


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{RAW_FOLDER}/{{name}}",
)
async def raw_to_sanitize(input: BlobClientTrigger) -> None:
    """
    First, raw documents are sanitized to remove any sensitive information.

    For PDF, QPDF (https://github.com/qpdf/qpdf) is used (from pikepdf) to save the document in a safe format.
    """
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing raw blob ({blob_name})")
        downloader = await blob_client.download_blob()
        in_bytes = BytesIO()
        await downloader.readinto(in_bytes)
    if _detect_extension(blob_name) == ".pdf":  # Sanitize PDF
        with pikepdf.open(in_bytes) as pdf:
            target_version = "1.4"
            logger.info(f"Sanitizing PDF from v{pdf.pdf_version} to v{target_version} ({blob_name})")
            out_stream = BytesIO()
            pdf.save(
                deterministic_id=True,  # Deterministic document ID for caching
                filename_or_stream=out_stream,
                linearize=True,  # Allows compliant readers to begin displaying a PDF file before it is fully downloaded
                min_version=target_version,  # Note, if a second PDF is created with a higher version, hash will be different and cache won't work
            )
            # Store
            out_path = _replace_root_path(blob_name, SANITIZE_FOLDER)
            out_client = await _use_blob_async_client(out_path)
            await out_client.upload_blob(
                data=out_stream.getbuffer(),
                overwrite=True,  # For the first upload, overwrite, next steps will validate MD5 for cache
            )
    else:  # Store as is
        logger.info(f"Storing raw blob as is ({blob_name})")
        out_path = _replace_root_path(blob_name, SANITIZE_FOLDER)
        out_client = await _use_blob_async_client(out_path)
        await out_client.upload_blob(
            data=in_bytes.getbuffer(),
            overwrite=True,  # For the first upload, overwrite, next steps will validate MD5 for cache
        )


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{SANITIZE_FOLDER}/{{name}}",
)
async def sanitize_to_extract(input: BlobClientTrigger) -> None:
    """
    First, document content is extracted from its binary form.
    """
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        blob_properties: BlobProperties = await blob_client.get_blob_properties()
        blob_md5 = b64encode(blob_properties.content_settings.content_md5).hex()  # See: https://github.com/Azure/azure-sdk-for-python/issues/13104#issuecomment-678033167
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
        features=[
            DocumentAnalysisFeature.BARCODES,
            DocumentAnalysisFeature.FORMULAS,
            DocumentAnalysisFeature.LANGUAGES,
            # DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,  # TODO: Enable this in the config?
        ]  # See: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-add-on-capabilities?view=doc-intel-4.0.0&tabs=rest-api
    )
    doc_result: AnalyzeResult = await doc_poller.result()
    # Build cracked model
    title_paragraph = next(
        (
            paragraph
            for paragraph in doc_result.paragraphs or []
            if paragraph.role == ParagraphRole.TITLE
        ),  # First, title
        next(
            (
                paragraph
                for paragraph in doc_result.paragraphs or []
                if paragraph.role == ParagraphRole.SECTION_HEADING
            ),  # Second, section heading
            None,  # Third, nothing
        ),
    )
    raw_text_model = ExtractedDocumentModel(
        document_content=doc_result.content,
        file_md5=blob_md5,
        file_path=blob_name,
        format="markdown",
        langs={lang.locale for lang in doc_result.languages or [] if lang.confidence > 0.5},
        title=title_paragraph.content if title_paragraph else None,
    )
    # Store
    out_path = f"{EXTRACT_FOLDER}/{blob_md5}.json"
    out_client = await _use_blob_async_client(out_path)
    try:
        await out_client.upload_blob(data=raw_text_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Document already exists, skipping ({blob_name})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{EXTRACT_FOLDER}/{{name}}",
)
async def extract_to_chunck(input: BlobClientTrigger) -> None:
    """
    Second, document content is chunked into smaller parts to make it understandable by the configured LLM.
    """
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing extracted blob ({blob_name})")
        downloader = await blob_client.download_blob()
        chunck = await downloader.readall()
        # Deserialize
        extracted_model = ExtractedDocumentModel.model_validate_json(chunck)
        # Free up memory
        del chunck
    # Prepare chunks for LLM
    chuncks = _split_text(
        text=extracted_model.document_content,
        max_tokens=int(
            CONFIG.llm.context * 0.8
        ),  # For simplicity, we count tokens with a 20% margin
    )
    logger.info(f"Splited to {len(chuncks)} chuncks ({blob_name})")
    # Store
    for i, chunck in enumerate(chuncks):  # TODO: Make this async
        out_model = ChunkedDocumentModel(
            chunk_content=chunck,
            chunk_number=i,
            file_md5=extracted_model.file_md5,
            file_path=extracted_model.file_path,
            format=extracted_model.format,
            langs=extracted_model.langs,
            title=extracted_model.title,
        )
        out_path = _replace_root_path(
            _replace_extension(blob_name, f"-{i}.json"), CHUNCK_FOLDER
        )
        out_client = await _use_blob_async_client(out_path)
        try:
            await out_client.upload_blob(data=out_model.model_dump_json())
        except ResourceExistsError:
            logger.info(f"Chunck already exists, skipping ({blob_name})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{CHUNCK_FOLDER}/{{name}}",
)
async def chunck_to_synthesis(input: BlobClientTrigger) -> None:
    """
    Third, chunks are synthesised into a coherent text.
    """
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing chuncked blob ({blob_name})")
        downloader = await blob_client.download_blob()
        content = await downloader.readall()
        # Deserialize
        chuncked_model = ChunkedDocumentModel.model_validate_json(content)
        # Free up memory
        del content
    # LLM does its magic
    synthesis_str = await _llm_generate_completion(
        max_tokens=500,  # 500 tokens ~= 375 words
        prompt=f"""
        Assistant is an expert data analyst with 20 years of experience.

        # Objective
        Synthesise the document. Content come from a chunked document created with an OCR tool, it may contain errors, repetitions, or missing parts, do your best to understand it.

        # Rules
        - Answer only with the synthesis, nothing else
        - Answers in English, even if the document is in another language
        - Be exhaustive and complete
        - Outline the main points but not the details
        - Should be in a single paragraph
        - Use only the information provided in the document

        # Document metadata
        - Format: {chuncked_model.format}
        - Lang: {", ".join(chuncked_model.langs)}
        - Title: {chuncked_model.title if chuncked_model.title else "N/A"}

        # Response example
        [synthesis]

        ## Example 1
        Content: Regulatory context. Scientific publications are unequivocal about the urgent challenges posed by climate change and the need for a transition to a climate-neutral economy. The International Energy Agency (IEA) asserts, in its Net Zero Emissions (NZE) scenario, that achieving carbon neutrality by 2050 and limiting warming to 1.5℃ by the end of the century requires an immediate end to all new fossil fuel exploration projects.
        Synthesis: This document addresses the urgent challenges posed by climate change and the need for a transition to a climate-neutral economy. Drafted by the International Energy Agency (IEA), the "Net Zero Emissions" (NZE) program aims to achieve carbon neutrality and limit global warming.

        ## Example 2
        Content: Life insurance fees: In order to increase the transparency of fees on these contracts, Gan Vie undertakes to update the information below on an annual basis. Last update September 01, 2023. Introductory remarks: contract management fees correspond to fees deducted directly by the insurer from the assets in Units of Account or in Euros. Additional fees may be charged depending on the management method chosen.
        Synthesis: Gan Vie undertakes to update information on life insurance fees annually. Fees are billed directly by the insurer, and additional fees may apply.

        # Document content
        {chuncked_model.chunk_content}
        """,  # TODO: Add at least 5 examples for different contexts
    )
    # Build model
    synthesis_model = SynthetisedDocumentModel(
        chunk_content=chuncked_model.chunk_content,
        chunk_number=chuncked_model.chunk_number,
        file_md5=chuncked_model.file_md5,
        file_path=chuncked_model.file_path,
        format=chuncked_model.format,
        langs=chuncked_model.langs,
        synthesis=synthesis_str,
        title=chuncked_model.title,
    )
    # Store
    out_path = _replace_root_path(
        _replace_extension(blob_name, ".json"), SYNTHESIS_FOLDER
    )
    out_client = await _use_blob_async_client(out_path)
    try:
        await out_client.upload_blob(data=synthesis_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Synthesis already exists, skipping ({blob_name})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{SYNTHESIS_FOLDER}/{{name}}",
)
async def synthesis_to_page(input: BlobClientTrigger) -> None:
    """
    Fourth, synthesises are chunked into pages.

    Pages are cleaned and filtered for repetitions (indicating low-quality content).
    """
    # Read
    async with await _use_blob_async_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing synthesis blob ({blob_name})")
        downloader = await blob_client.download_blob()
        page = await downloader.readall()
        # Deserialize
        synthesis_model = SynthetisedDocumentModel.model_validate_json(page)
        # Free up memory
        del page
    # Prepare chunks for LLM
    pages = _split_text(
        max_tokens=int(100 / 75 * 500),  # 100 tokens ~= 75 words, ~500 words per page for a dense book
        text=synthesis_model.chunk_content,
    )
    logger.info(f"Splited to {len(pages)} pages ({blob_name})")
    # Store
    for i, page in enumerate(pages):  # TODO: Make this async
        # First, clean
        page = _clean_page(page)
        if not page:
            logger.info(f"Page skipped ({blob_name})")
            return
        # Second, filter-out pages with excessive repetition
        if _is_repetition_removal(
            text=page,
            threshold_ratio=1.5,  # We are less strict than the paper because this is all normally internal data and we are not training a model
        ):
            logger.info(f"Repetition detected, skipping ({blob_name})")
            return
        out_model = PagedDocumentModel(
            chunk_content=synthesis_model.chunk_content,
            chunk_number=synthesis_model.chunk_number,
            file_md5=synthesis_model.file_md5,
            file_path=synthesis_model.file_path,
            format=synthesis_model.format,
            langs=synthesis_model.langs,
            page_content=page,
            page_number=i,
            synthesis=synthesis_model.synthesis,
            title=synthesis_model.title,
        )
        out_path = _replace_root_path(
            _replace_extension(blob_name, f"-{i}.json"), PAGE_FOLDER
        )
        out_client = await _use_blob_async_client(out_path)
        try:
            await out_client.upload_blob(data=out_model.model_dump_json())
        except ResourceExistsError:
            logger.info(f"Page already exists, skipping ({blob_name})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{PAGE_FOLDER}/{{name}}",
)
async def page_to_fact(input: BlobClientTrigger) -> None:
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
        paged_model = PagedDocumentModel.model_validate_json(content)
        # Free up memory
        del content
    # LLM does its magic
    facted_llm_model = await _llm_generate_model(
        model=FactedLlmModel,
        prompt=f"""
        Assistant is an expert data analyst with 20 years of experience.

        # Objective
        Create question/answer pairs for a document. Content come from a paged document created with an OCR tool, it may contain errors, repetitions, or missing parts, do your best to understand it.

        # Rules
        - Answers in English, even if the document is in another language
        - Be exhaustive and complete
        - Only use the information provided in the document

        # Result format as a JSON schema
        {json.dumps(FactedLlmModel.model_json_schema())}

        # Document metadata
        - Format: {format}
        - Lang: {", ".join(paged_model.langs)}
        - Title: {paged_model.title or "N/A"}

        # Document synthesis
        {paged_model.synthesis}

        # Document content
        {paged_model.page_content}
        """,  # TODO: Add at least 5 examples for different contexts
    )
    if not facted_llm_model.facts:
        logger.info(f"No facts detected, skipping")
        return
    # Build model
    facted_document_model = FactedDocumentModel(
        chunk_content=paged_model.chunk_content,
        chunk_number=paged_model.chunk_number,
        facts=facted_llm_model.facts,
        file_md5=paged_model.file_md5,
        file_path=paged_model.file_path,
        format=paged_model.format,
        langs=paged_model.langs,
        page_content=paged_model.page_content,
        page_number=paged_model.page_number,
        synthesis=paged_model.synthesis,
        title=paged_model.title,
    )
    # Store
    out_path = _replace_root_path(
        _replace_extension(blob_name, ".json"), FACT_FOLDER
    )
    out_client = await _use_blob_async_client(out_path)
    try:
        await out_client.upload_blob(data=facted_document_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Fact already exists, skipping ({blob_name})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{FACT_FOLDER}/{{name}}",
)
async def fact_to_critic(input: BlobClientTrigger) -> None:
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
        facted_model = FactedDocumentModel.model_validate_json(content)
        # Free up memory
        del content
    # Filter facts
    facts = await asyncio.gather(
        *[
            _critic_fact_filter(
                fact=fact,
                model=facted_model,
            )
            for fact in facted_model.facts
        ]
    )
    facted_model.facts = [fact for fact in facts if fact]
    if not facted_model.facts:
        logger.info(f"No facts left, skipping")
        return
    # Store
    out_path = _replace_root_path(
        _replace_extension(blob_name, ".json"), CRITIC_FOLDER
    )
    out_client = await _use_blob_async_client(out_path)
    try:
        await out_client.upload_blob(data=facted_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Critic already exists, skipping ({blob_name})")


async def _critic_fact_filter(
    fact: FactModel,
    model: FactedDocumentModel,
) -> Optional[FactedDocumentModel]:
    score_str = await _llm_generate_completion(
        max_tokens=10,  # We only need a float score
        prompt=f"""
        Assistant is an expert data analyst with 20 years of experience.

        # Objective
        Evaluate the quality of a fact. The fact is a question/answer pair created from a paged document.

        # Rules
        - Answer only with the score, nothing else
        - High scores indicate that the fact is likely to be correct and relevant
        - Low scores indicate that the fact is likely to be incorrect or irrelevant
        - Only use the information provided in the document
        - The score should be between 0.0 and 1.0
        - The score should reflect the quality of the fact based on the document synthesis, page content, and context

        # Document metadata
        - Format: {format}
        - Lang: {", ".join(model.langs)}
        - Title: {model.title or "N/A"}

        # Document synthesis
        {model.synthesis}

        # Page content
        {model.page_content}

        # Response example
        [score]

        ## Example 1
        Question: What is the capital of France?
        Answer: Paris
        Context: Paris, as the capital of France, is the political, economic, and cultural center of the country.
        Score: 1.0

        ## Example 2
        Question: What is the ISIN code for the stock?
        Answer: US0378331005
        Context: The ISIN code for the stock is FR0000120172.
        Score: 0.0

        ## Example 3
        Question: In which year was the company founded?
        Answer: 1939
        Context: The company, by its founder, was established during World War II to provide essential services to the population. Its exact founding date is unknown.
        Score: 0.6

        ## Example 4
        Question: What is the main product of the company?
        Answer: A software suite
        Context: The company is known for its software suite called "Office", which includes applications such as a text editor, a spreadsheet, and a presentation program.
        Score: 0.8


        # Fact
        Question: {fact.question}
        Answer: {fact.answer}
        Context: {fact.context}
        """,   # TODO: Add at least 5 examples for different contexts
    )
    try:
        score = float(score_str)  # LLM should return a float
    except ValueError:
        score = float(re.search(r"\d+\.\d+", score_str).group())  # As a fallback, we try to extract the score from the string
    if score < 0.5:
        logger.info(f"Low score detected ({score:.2f}), skipping")
        logger.info(f"Question: {fact.question}")
        logger.info(f"Answer: {fact.answer}")
        logger.info(f"Context: {fact.context}")
        logger.info(f"Score: {score:.2f}")
        return
    return fact


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{CRITIC_FOLDER}/{{name}}",
)
async def critic_to_index(input: BlobClientTrigger) -> None:
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
        facted_model = FactedDocumentModel.model_validate_json(content)
        # Free up memory
        del content
    # Build indexed model
    indexed_models = [
        IndexedDocumentModel(
            answer=fact.answer,
            context=fact.context,
            document_synthesis=facted_model.synthesis,
            file_path=facted_model.file_path,
            id=_hash_text(f"{facted_model.file_path}-{facted_model.chunk_number + facted_model.page_number + i}"),  # Reproducible ID over the same raw document
            question=fact.question,
        )
        for i, fact in enumerate(facted_model.facts)
    ]
    # Index
    logger.info(f"Indexing {len(indexed_models)} documents to AI Search ({blob_name})")
    indexed_dicts = TypeAdapter(list[IndexedDocumentModel]).dump_python(
        indexed_models, mode="json"
    )
    search_client = await _use_search_client()
    await search_client.merge_or_upload_documents(indexed_dicts)  # Will overwrite existing documents


def _split_text(text: str, max_tokens: int) -> list[str]:
    """
    Split a text into chunks of text with a maximum number of tokens and characters.

    The function returns a list of text chunks.
    """
    contents = []
    first_margin = 100
    last_margin = 100
    max_chars = int(1048576 * 0.9)  # REST API has a limit of 1MB, with a 10% margin
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


async def _llm_generate_completion(
    max_tokens: int,
    prompt: str,
) -> str:
    """
    Generate a completion from a prompt using OpenAI.

    The completion is generated using the LLM model.
    """
    logger.info("LLM completion generation")
    openai_client = await _use_openai_client()
    llm_res = await openai_client.chat.completions.create(
        max_tokens=max_tokens,
        model=CONFIG.llm.model,
        messages=[
            ChatCompletionSystemMessageParam(
                content=prompt,
                role="system",
            ),
        ],
    )
    return llm_res.choices[0].message.content  # type: ignore


async def _llm_generate_model(
    model: Type[Model],
    prompt: str,
    _previous_result: Optional[str] = None,
    _retries_remaining: int = 3,
    _validation_error: Optional[str] = None,
) -> Model:
    """
    Generate a synthesis from a content using OpenAI.

    The synthesis is generated using the LLM model. Then, the synthesis is stored in the FACT folder.
    """
    logger.info(
        f"LLM model generation ({_retries_remaining} retries left)"
    )
    openai_client = await _use_openai_client()
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
        model_res = model.model_validate_json(llm_content)
    except ValidationError as e:
        if _retries_remaining == 0:
            raise e
        logger.warning(f"LLM validation error")
        return await _llm_generate_model(
            model=model,
            prompt=prompt,
            _previous_result=llm_content,
            _retries_remaining=_retries_remaining - 1,
            _validation_error=str(e),
        )
    return model_res


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
    for line in text.splitlines():
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
    Hash a text using MD5.

    MD5 is as of today the fastest hash function available. It has collision vulnerabilities, but it is not a concern in this context.

    Return the MD5 hash of the text.
    """
    return hashlib.md5(
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


def _detect_extension(file_path: str) -> str:
    """
    Detect the extension of a file path.

    For example, if the file path is "file.txt", the extension will be ".txt".

    Return the extension in lowercase.
    """
    return "." + file_path.lower().split(".")[-1]


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
