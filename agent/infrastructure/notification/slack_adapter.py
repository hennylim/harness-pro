"""
Notification adapters.

``SlackWebhookAdapter`` posts to a Slack incoming-webhook URL.
``NullNotificationAdapter`` is a no-op used when notifications are disabled.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen
from urllib.error import URLError

import structlog

from agent.domain.entities import RunStatus, RunSummary, Task
from agent.domain.interfaces import INotificationAdapter

log = structlog.get_logger(__name__)

_STATUS_EMOJI = {
    RunStatus.COMPLETED: "✅",
    RunStatus.FAILED: "❌",
    RunStatus.ABORTED: "🛑",
    RunStatus.RUNNING: "⏳",
}


class SlackWebhookAdapter(INotificationAdapter):
    """
    Posts messages to Slack via an incoming-webhook URL.

    Parameters
    ----------
    webhook_url:
        The Slack incoming-webhook URL (from App configuration).
    timeout_seconds:
        HTTP request timeout.
    """

    def __init__(self, webhook_url: str, timeout_seconds: int = 10) -> None:
        self._url = webhook_url
        self._timeout = timeout_seconds

    def notify_run_complete(self, summary: RunSummary) -> None:
        emoji = _STATUS_EMOJI.get(summary.status, "❓")
        duration = (
            f"{summary.duration_seconds:.1f}s"
            if summary.duration_seconds is not None
            else "n/a"
        )
        text = (
            f"{emoji} *Harness run {summary.run_id}* — {summary.status.value.upper()}\n"
            f"Tasks: ✅ {summary.completed_tasks}  ❌ {summary.failed_tasks}  "
            f"⏭ {summary.skipped_tasks}  |  Duration: {duration}"
        )
        self._post({"text": text})

    def notify_task_failed(self, task: Task, reason: str) -> None:
        short_reason = reason[:300] + ("…" if len(reason) > 300 else "")
        text = (
            f"❌ *Task failed* — `{task.file_path}`\n"
            f"Retries: {task.retry_count}\n"
            f"```{short_reason}```"
        )
        self._post({"text": text})

    def _post(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        req = Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    log.warning("slack.post.non200", status=resp.status)
        except URLError as exc:
            log.warning("slack.post.error", error=str(exc))


class NullNotificationAdapter(INotificationAdapter):
    """No-op adapter used when notifications are disabled."""

    def notify_run_complete(self, summary: RunSummary) -> None:
        pass

    def notify_task_failed(self, task: Task, reason: str) -> None:
        pass
