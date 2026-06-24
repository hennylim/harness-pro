"""
Centralised structlog configuration.
Call ``configure_logging()`` once at process startup.
"""
import logging
import sys
import uuid
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    run_id: Optional[str] = None,
) -> str:
    """Configure structlog + stdlib logging. Returns the active run_id."""
    run_id = run_id or str(uuid.uuid4())[:8]

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
        handler = RichHandler(console=_console, rich_tracebacks=True, show_path=False)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))

    # Bind run_id globally so it appears in every log line
    structlog.contextvars.bind_contextvars(run_id=run_id)

    return run_id
