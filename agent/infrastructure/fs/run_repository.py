"""
JSON-based run repository.

Stores each :class:`RunSummary` as a single JSON file inside a designated
``runs/`` directory.  Suitable for local use and simple CI pipelines.

For high-throughput production use, swap with a database-backed
implementation that satisfies :class:`IRunRepository`.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import structlog

from agent.domain.entities import RunSummary
from agent.domain.interfaces import IRunRepository

log = structlog.get_logger(__name__)


class JsonRunRepository(IRunRepository):
    """
    Persist run summaries as ``<runs_dir>/<run_id>.json``.

    Parameters
    ----------
    runs_dir:
        Directory where run JSON files are stored.
        Created automatically if it does not exist.
    """

    def __init__(self, runs_dir: str = "./runs") -> None:
        self._dir = os.path.abspath(runs_dir)
        os.makedirs(self._dir, exist_ok=True)

    def save(self, summary: RunSummary) -> None:
        path = os.path.join(self._dir, f"{summary.run_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(summary.model_dump_json(indent=2))
            log.debug("run_repo.saved", run_id=summary.run_id, path=path)
        except OSError as exc:
            log.warning("run_repo.save_failed", run_id=summary.run_id, error=str(exc))

    def load(self, run_id: str) -> Optional[RunSummary]:
        path = os.path.join(self._dir, f"{run_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return RunSummary(**data)
        except (OSError, ValueError) as exc:
            log.warning("run_repo.load_failed", run_id=run_id, error=str(exc))
            return None
