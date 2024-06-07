from azure.monitor.opentelemetry import configure_azure_monitor
from helpers.config import CONFIG
from logging import getLogger, basicConfig
from opentelemetry import trace
from os import getenv


APP_NAME = "synthetic-rag-index"

# Logger
basicConfig(level=CONFIG.monitoring.logging.sys_level.value)
logger = getLogger(APP_NAME)
logger.setLevel(CONFIG.monitoring.logging.app_level.value)

# OpenTelemetry
configure_azure_monitor(
    connection_string=getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
)  # Configure Azure Application Insights exporter
tracer = trace.get_tracer(
    instrumenting_module_name=f"com.github.clemlesne.{APP_NAME}",
)  # Create a tracer that will be used in the app
