from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from helpers.config_models.document_intelligence import DocumentIntelligenceModel
from helpers.file import detect_extension
from helpers.http import azure_transport
from helpers.logging import logger
from persistence.ianalyze import IAnalyze
from typing import Any, Generator, Optional
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    ContentFormat,
    DocumentAnalysisFeature,
    ParagraphRole,
)
import math
import html


class DocumentIntelligenceAnalyze(IAnalyze):
    _client: Optional[DocumentIntelligenceClient] = None
    _config: DocumentIntelligenceModel

    def __init__(self, config: DocumentIntelligenceModel):
        self._config = config

    async def analyze(
        self,
        document: bytes,
        file_name: str,
    ) -> tuple[str, Optional[str], list[str]]:
        logger.info(f"Analyzing document ({file_name})")

        # Detect features
        features: list[DocumentAnalysisFeature] = []
        if detect_extension(file_name) in [".pdf", ".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".heif", ".heic"]:
            features.append(DocumentAnalysisFeature.BARCODES)
            features.append(DocumentAnalysisFeature.FORMULAS)
            features.append(DocumentAnalysisFeature.LANGUAGES)
        logger.info(f"Features enabled: {features}")

        # Analyze
        async with await self._use_client() as client:
            poller = await client.begin_analyze_document(
                analyze_request=document,
                content_type="application/octet-stream",
                features=features,  # See: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-add-on-capabilities?view=doc-intel-4.0.0&tabs=rest-api
                model_id="prebuilt-layout",
                output_content_format=ContentFormat.MARKDOWN,
            )
            res: AnalyzeResult = await poller.result()
            title_paragraph = next(
                (
                    paragraph
                    for paragraph in res.paragraphs or []
                    if paragraph.role == ParagraphRole.TITLE
                ),  # First, title
                next(
                    (
                        paragraph
                        for paragraph in res.paragraphs or []
                        if paragraph.role == ParagraphRole.SECTION_HEADING
                    ),  # Second, section heading
                    None,  # Third, nothing
                ),
            )
            content = html.unescape(res.content)
            title = html.unescape(title_paragraph.content) if title_paragraph else None
            langs = {lang.locale for lang in res.languages or [] if lang.confidence >= self._config.extract_lang_threshold}

        # Return title, content and langs
        return content, title, langs

    def chunck(self, pages: list[Any]) -> Generator[tuple[list[int], int, int], None, None]:
        first_pages_count = 2
        last_pages_count = 2
        files_count = math.ceil(len(pages) / (self._config.pdf_pages_max - first_pages_count - last_pages_count))  # Limit pages because this is the hard limit in Document Intelligence
        for i in range(files_count):  # Iterate over desired chunks
            pages_numbers = []
            pages_numbers += list(range(0, first_pages_count))  # First pages are always kept
            content_from = max(i * self._config.pdf_pages_max - first_pages_count, first_pages_count)
            content_to = min((i + 1) * self._config.pdf_pages_max, len(pages)) - last_pages_count
            pages_numbers += list(range(content_from, content_to))  # Add middle pages
            pages_numbers += list(range(len(pages) - last_pages_count, len(pages)))  # Last pages are always kept
            # Yield pages numbers, current file index and total files count
            yield pages_numbers, i, files_count

    async def _use_client(self) -> DocumentIntelligenceClient:
        if not self._client:
            self._client = DocumentIntelligenceClient(
                # Deployment
                endpoint=self._config.endpoint,
                # Performance
                polling_interval=5,  # 5 seconds
                transport=await azure_transport(),
                # Authentication
                credential=AzureKeyCredential(
                    self._config.access_key.get_secret_value()
                ),
            )
        return self._client
