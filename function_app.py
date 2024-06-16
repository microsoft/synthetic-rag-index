# First imports, to make sure the following logs are first
from helpers.config import CONFIG
from helpers.logging import logger, APP_NAME


logger.info(f"{APP_NAME} v{CONFIG.version}")


# General imports
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobProperties
from azure.storage.blob.aio import BlobClient, ContainerClient
from azurefunctions.extensions.bindings.blob import BlobClient as BlobClientTrigger
from os import getenv
from pydantic import ValidationError
from typing import IO, Optional, TypeVar
import asyncio
import azure.functions as func
import json
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
from base64 import b64encode
from unidecode import unidecode
import pymupdf
import re
from helpers.file import detect_extension, replace_root_path, replace_extension, hash_text, sanitize_text, has_excessive_repetition
from tempfile import NamedTemporaryFile
from os import remove


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

# Custom types
T = TypeVar("T")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{RAW_FOLDER}/{{name}}",
)
async def raw_to_sanitize(input: BlobClientTrigger) -> None:
    """
    First, raw documents are sanitized to remove any sensitive information.
    """
    async def _upload(local_file: IO, remote_path: str) -> None:
        remote_path = unidecode(replace_root_path(remote_path, SANITIZE_FOLDER), replace_str="")  # Decode possible non ASCII characters
        out_client = await _use_blob_client(remote_path)
        await out_client.upload_blob(
            data=local_file,
            overwrite=True,  # For the first upload, overwrite, next steps will validate MD5 for cache
        )

    # Read
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        in_remote_path = blob_client.blob_name
        logger.info(f"Processing raw blob ({in_remote_path})")

        with NamedTemporaryFile() as in_local_path:  # Temp file for source
            # Download and save
            downloader = await blob_client.download_blob()
            await downloader.readinto(in_local_path)
            in_local_path.seek(0)  # Reset file pointer

            if detect_extension(in_remote_path) in {".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz", ".svg", ".txt"}:  # Sanitize with PyMuPDF
                logger.info(f"Sanitizing ({in_remote_path})")
                doc_client = CONFIG.document_intelligence.instance()
                # Open
                in_pdf = pymupdf.open(in_local_path)
                if not in_pdf.is_pdf:  # Convert to PDF
                    in_pdf = pymupdf.open("pdf", in_pdf.convert_to_pdf())
                # Sanitize
                in_pdf.scrub(
                    hidden_text=False,  # Keep hidden text (it may contain OCR)
                    remove_links=False,  # Keep links (they are extracted later)
                    reset_fields=False,  # Keep form fields (they are extracted later)
                )

                for pages_numbers, i, chuncks_count in doc_client.chunck(in_pdf.page_count):  # Iterate over chuncks
                    out_remote_path = replace_extension(in_remote_path, f"-{i}.pdf")
                    out_local_path = NamedTemporaryFile(delete=False)  # Temp file for destination
                    out_local_path.close()  # Close to allow pymupdf to open it later
                    logger.info(f"Saving PDF file {i + 1}/{chuncks_count} ({out_remote_path})")

                    # Create new PDF and insert selected pages
                    out_pdf = pymupdf.Document()
                    for pages_number in pages_numbers:
                        out_pdf.insert_pdf(in_pdf, from_page=pages_number, to_page=pages_number)

                    # Save and optimize
                    out_pdf.save(
                        out_local_path.name,
                        appearance=True,  # Annotate buttons, form fields, and links
                        clean=True,  # Clean and sanitize content streams
                        compression_effort=100,  # Maximum compression
                        deflate_fonts=True,  # Compress uncompressed fonts
                        deflate_images=True,  # Compress uncompressed images
                        deflate=True,  # Compress uncompressed streams
                        garbage=4,  # Remove unused objects
                        linear=True,  # Linearize the PDF for fast web view
                        preserve_metadata=True,  # Preserve original metadata
                    )

                    # Upload
                    with open(out_local_path.name, "rb") as out_local_path:
                        await _upload(
                            local_file=out_local_path,
                            remote_path=out_remote_path,
                        )
                    remove(out_local_path.name)  # Clean-up

            else:  # Store as is
                logger.info(f"Saving raw blob as is ({in_remote_path})")
                await _upload(
                    local_file=in_local_path,
                    remote_path=in_remote_path,
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
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        in_remote_path = blob_client.blob_name
        blob_properties: BlobProperties = await blob_client.get_blob_properties()
        blob_md5 = b64encode(blob_properties.content_settings.content_md5).hex()  # See: https://github.com/Azure/azure-sdk-for-python/issues/13104#issuecomment-678033167
        logger.info(f"Processing raw blob ({in_remote_path})")

        with NamedTemporaryFile() as in_local_path:  # Temp file for source
            # Download and save
            downloader = await blob_client.download_blob()
            await downloader.readinto(in_local_path)
            in_local_path.seek(0)  # Reset file pointer
            doc_client = CONFIG.document_intelligence.instance()

            if detect_extension(in_remote_path) in doc_client.compatible_formats():  # Analyze document
                logger.info(f"Extracting content with Document Intelligence ({in_remote_path})")
                content, title, langs = await doc_client.analyze(
                    document=in_local_path.file,
                    file_name=in_remote_path,
                )
                format = "markdown"
                # Clean content
                content = sanitize_text(content)

            else:  # Store as is
                logger.info(f"Extracting binary content ({in_remote_path})")
                with open(in_local_path.name, "rb") as in_local_path:
                    content = in_local_path.read().decode("utf-8")
                    format = "raw"
                    langs = None
                    title = None

        # Build model
        if not content:
            logger.warning(f"Content skipped ({in_remote_path})")
            return
        raw_text_model = ExtractedDocumentModel(
            document_content=content,
            file_md5=blob_md5,
            file_path=in_remote_path,
            format=format,
            langs=langs,
            title=title,
        )

        # Store
        out_path = f"{EXTRACT_FOLDER}/{blob_md5}.json"
        out_client = await _use_blob_client(out_path)
        try:
            await out_client.upload_blob(data=raw_text_model.model_dump_json())
        except ResourceExistsError:
            logger.info(f"Document already exists, skipping ({out_path})")


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
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing extracted blob ({blob_name})")
        downloader = await blob_client.download_blob()
        # Deserialize
        extracted_model = ExtractedDocumentModel.model_validate_json(await downloader.readall())

    # Prepare chunks for LLM
    llm_client = CONFIG.llm.selected(
        is_fast=False,  # We will use the slow model next step
    ).instance()
    chuncks = llm_client.chunck(text=extracted_model.document_content)
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
        out_path = replace_root_path(
            replace_extension(blob_name, f"-{i}.json"), CHUNCK_FOLDER
        )
        logger.info(f"Saving chunck {i + 1}/{len(chuncks)} ({out_path})")
        out_client = await _use_blob_client(out_path)
        try:
            await out_client.upload_blob(data=out_model.model_dump_json())
        except ResourceExistsError:
            logger.info(f"Chunck already exists, skipping ({out_path})")


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
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing chuncked blob ({blob_name})")
        downloader = await blob_client.download_blob()
        # Deserialize
        chuncked_model = ChunkedDocumentModel.model_validate_json(await downloader.readall())

    # LLM does its magic
    def _validate(req: Optional[str]) -> tuple[bool, Optional[str], Optional[str]]:
        if not req:
            return False, "Empty response", None
        req = req.strip()
        if len(req) < 10:  # Arbitrary minimum length
            return False, "Response too short", None
        return True, None, req
    llm_client = CONFIG.llm.selected(
        is_fast=False,  # We want high quality summaries because they are used to avoid hallucinations in the next steps
    ).instance()
    synthesis_str = await llm_client.generate(
        max_tokens=500,  # 500 tokens ~= 375 words
        res_object=str,
        validation_callback=_validate,
        prompt=f"""
        Assistant is an expert data analyst with 20 years of experience.

        # Objective
        Synthesise the document. Content come from a chunked document created with an OCR tool, it may contain errors, repetitions, or missing parts, do your best to understand it.

        # Rules
        - Answer only with the synthesis, nothing else
        - Answers in English, even if the document is in another language
        - Be concise
        - Outline the main points but not the details
        - Use only the information provided in the document

        # Document metadata
        - Format: {chuncked_model.format}
        - Lang: {", ".join(chuncked_model.langs) if chuncked_model.langs else "N/A"}
        - Title: {chuncked_model.title if chuncked_model.title else "N/A"}

        # Response format
        [synthesis, single paragraph]

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
    out_path = replace_root_path(
        replace_extension(blob_name, ".json"), SYNTHESIS_FOLDER
    )
    out_client = await _use_blob_client(out_path)
    try:
        await out_client.upload_blob(data=synthesis_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Synthesis already exists, skipping ({out_path})")


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
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing synthesis blob ({blob_name})")
        downloader = await blob_client.download_blob()
        # Deserialize
        synthesis_model = SynthetisedDocumentModel.model_validate_json(await downloader.readall())

    # Prepare chunks for LLM
    llm_client = CONFIG.llm.selected(
        is_fast=True,  # We will use the fast model
    ).instance()
    pages = llm_client.chunck(
        max_tokens=CONFIG.features.page_split_size,
        text=synthesis_model.chunk_content,
    )
    logger.info(f"Splited to {len(pages)} pages ({blob_name})")

    # Store
    for i, page in enumerate(pages):  # TODO: Make this async
        # Filter-out pages with excessive repetition
        if has_excessive_repetition(
            text=page,
            threshold_ratio=1.5,  # We are less strict than the paper because this is all normally internal data and we are not training a model
        ):
            logger.warning(f"Repetition detected, skipping ({blob_name})")
            continue
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
        out_path = replace_root_path(
            replace_extension(blob_name, f"-{i}.json"), PAGE_FOLDER
        )
        logger.info(f"Saving page {i + 1}/{len(pages)} ({out_path})")
        out_client = await _use_blob_client(out_path)
        try:
            await out_client.upload_blob(data=out_model.model_dump_json())
        except ResourceExistsError:
            logger.info(f"Page already exists, skipping ({out_path})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{PAGE_FOLDER}/{{name}}",
)
async def page_to_fact(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing repetition-filtered blob ({blob_name})")
        downloader = await blob_client.download_blob()
        # Deserialize
        paged_model = PagedDocumentModel.model_validate_json(await downloader.readall())

    # LLM does its magic
    llm_client = CONFIG.llm.selected(
        is_fast=True,  # We will use the fast model
    ).instance()
    facts: list[FactModel] = []
    for _ in range(CONFIG.features.fact_iterations):  # We will generate facts 10 times
        def _validate(req: Optional[str]) -> tuple[bool, Optional[str], Optional[FactedLlmModel]]:
            try:
                return True, None, FactedLlmModel.model_validate_json(req)
            except ValidationError as e:
                return False, str(e), None
        facted_llm_model = await llm_client.generate(
            res_object=FactedLlmModel,
            temperature=1,  # We want creative answers
            validate_json=True,
            validation_callback=_validate,
            prompt=f"""
            Assistant is an expert data analyst with 20 years of experience.

            # Objective
            Create new question/answer pairs for a document. Content come from a paged document created with an OCR tool, it may contain errors, repetitions, or missing parts, do your best to understand it.

            # Rules
            - Answers in English, even if the document is in another language
            - Be concise
            - New facts must be on different points than the ones already generated
            - Only use the information provided in the document

            # Document metadata
            - Format: {format}
            - Lang: {", ".join(paged_model.langs) if paged_model.langs else "N/A"}
            - Title: {paged_model.title or "N/A"}

            # Document synthesis
            {paged_model.synthesis}

            # Document content
            {paged_model.page_content}

            # Facts already generated
            {FactedLlmModel(facts=facts).model_dump_json() if facts else "N/A"}

            # Response format (JSON schema)
            {json.dumps(FactedLlmModel.model_json_schema())}

            ## Example 1
            Synthesis: This document addresses the demographic challenges faced by the country. The population is aging, and the birth rate is declining. The government has implemented policies to address these issues.
            Content: The mayor of the Parisian district of Montmartre has announced a new initiative to address the demographic issues. This is a first step for the capital.
            Response: {FactedLlmModel(facts=[FactModel(question="What is the capital of France?", answer="Paris", context="Paris, as the capital of France, is the political, economic, and cultural center of the country.")]).model_dump_json()}
            """,  # TODO: Add at least 5 examples for different contexts
        )
        if not facted_llm_model:
            continue
        facts += facted_llm_model.facts
    if not facts:
        logger.info(f"No facts detected, skipping")
        return
    logger.info(f"Generated {len(facts)} facts ({blob_name})")

    # Build model
    facted_document_model = FactedDocumentModel(
        chunk_content=paged_model.chunk_content,
        chunk_number=paged_model.chunk_number,
        facts=facts,
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
    out_path = replace_root_path(
        replace_extension(blob_name, ".json"), FACT_FOLDER
    )
    out_client = await _use_blob_client(out_path)
    try:
        await out_client.upload_blob(data=facted_document_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Fact already exists, skipping ({out_path})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{FACT_FOLDER}/{{name}}",
)
async def fact_to_critic(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing fact blob ({blob_name})")
        downloader = await blob_client.download_blob()
        # Deserialize
        facted_model = FactedDocumentModel.model_validate_json(await downloader.readall())

    # Score facts
    initial_fact_count = len(facted_model.facts)
    def _validate(req: Optional[str]) -> tuple[bool, Optional[str], Optional[float]]:
        if not req:
            return False, "Empty response", None
        req = req.strip()
        try:
            return True, None, float(req)
        except ValueError:
            group = re.search(r"\d+\.\d+", req)
            if group:
                return True, None, float(group.group())
            return False, "Score not detected", None
    llm_client = CONFIG.llm.selected(
        is_fast=False,  # We want high quality to avoid using human validation which is even more costly and slower
    ).instance()
    fact_scores = await asyncio.gather(
        *[
            llm_client.generate(
                max_tokens=10,  # We only need a float score
                res_object=float,
                validation_callback=_validate,
                prompt=f"""
                Assistant is an expert data analyst with 20 years of experience.

                # Objective
                Evaluate the quality of a fact. The fact is a question/answer pair created from a paged document.

                # Rules
                - Answer only with the score, nothing else
                - High scores indicate that the fact is likely to be correct and relevant
                - Low scores indicate that the fact is likely to be incorrect or irrelevant
                - Only use the information provided in the document
                - The score should reflect the quality of the fact based on the document synthesis, page content, and context

                # Document metadata
                - Format: {format}
                - Lang: {", ".join(facted_model.langs) if facted_model.langs else "N/A"}
                - Title: {facted_model.title or "N/A"}

                # Document synthesis
                {facted_model.synthesis}

                # Page content
                {facted_model.page_content}

                # Response format
                [score, a float between 0.0 and 1.0]

                ## Example 1
                Question: What is the capital of France?
                Answer: Paris
                Context: Paris, as the capital of France, is the political, economic, and cultural center of the country.
                Assistant: 1.0

                ## Example 2
                Question: What is the ISIN code for the stock?
                Answer: US0378331005
                Context: The ISIN code for the stock is FR0000120172.
                Assistant: 0.0

                ## Example 3
                Question: In which year was the company founded?
                Answer: 1939
                Context: The company, by its founder, was established during World War II to provide essential services to the population. Its exact founding date is unknown.
                Assistant: 0.6

                ## Example 4
                Question: What is the main product of the company?
                Answer: A software suite
                Context: The company is known for its software suite called "Office", which includes applications such as a text editor, a spreadsheet, and a presentation program.
                Assistant: 0.8

                # Fact
                Question: {fact.question}
                Answer: {fact.answer}
                Context: {fact.context}
                """,   # TODO: Add at least 5 examples for different contexts
            )
            for fact in facted_model.facts
        ]
    )

    # Filter facts
    kept_facts = []
    for i, fact_score in enumerate(fact_scores):
        if fact_score >= CONFIG.features.fact_score_threshold:  # Discard low quality facts
            kept_facts.append(facted_model.facts[i])
    facted_model.facts = kept_facts
    if not facted_model.facts:
        logger.info(f"No facts left, skipping")
        return
    logger.info(f"Filtered to {len(facted_model.facts)}/{initial_fact_count} facts ({blob_name})")

    # Store
    out_path = replace_root_path(
        replace_extension(blob_name, ".json"), CRITIC_FOLDER
    )
    out_client = await _use_blob_client(out_path)
    try:
        await out_client.upload_blob(data=facted_model.model_dump_json())
    except ResourceExistsError:
        logger.info(f"Critic already exists, skipping ({out_path})")


@app.blob_trigger(
    arg_name="input",
    connection="AzureWebJobsStorage",
    data_type=func.DataType.BINARY,
    path=f"{CONTAINER_NAME}/{CRITIC_FOLDER}/{{name}}",
)
async def critic_to_index(input: BlobClientTrigger) -> None:
    # Read
    async with await _use_blob_client(
        name=input.blob_name,  # type: ignore
        snapshot=input.snapshot,  # type: ignore
    ) as blob_client:
        blob_name = blob_client.blob_name
        logger.info(f"Processing fact blob ({blob_name})")
        downloader = await blob_client.download_blob()
        # Deserialize
        facted_model = FactedDocumentModel.model_validate_json(await downloader.readall())

    # Build indexed model
    indexed_models = [
        IndexedDocumentModel(
            answer=fact.answer,
            context=fact.context,
            document_synthesis=facted_model.synthesis,
            file_path=facted_model.file_path,
            id=hash_text(f"{facted_model.file_md5}-{facted_model.chunk_number + facted_model.page_number + i}"),  # Reproducible ID over the same raw document
            question=fact.question,
        )
        for i, fact in enumerate(facted_model.facts)
    ]

    # Index
    destination_client = CONFIG.destination.instance()
    await destination_client.index(indexed_models)


async def _use_blob_client(
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
