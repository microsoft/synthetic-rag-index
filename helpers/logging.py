from azure.monitor.opentelemetry import configure_azure_monitor
from helpers.config import CONFIG
from logging import getLogger, basicConfig
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from os import getenv, environ


APP_NAME = "synthetic-rag-index"

# Logger
basicConfig(level=CONFIG.monitoring.logging.sys_level.value)
logger = getLogger(APP_NAME)
logger.setLevel(CONFIG.monitoring.logging.app_level.value)

# OpenTelemetry
configure_azure_monitor(
    connection_string=getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
)  # Configure Azure Application Insights exporter
AioHttpClientInstrumentor().instrument()  # Instrument aiohttp
HTTPXClientInstrumentor().instrument()  # Instrument httpx
environ["TRACELOOP_TRACE_CONTENT"] = str(
    True
)  # Instrumentation logs prompts, completions, and embeddings to span attributes, set to False to lower monitoring costs or to avoid logging PII
OpenAIInstrumentor().instrument()  # Instrument OpenAI
