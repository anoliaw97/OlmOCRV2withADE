from __future__ import annotations

from backend.runtime import WorkflowRuntime


_RUNTIME = WorkflowRuntime()


def get_runtime() -> WorkflowRuntime:
    return _RUNTIME


def close_runtime() -> None:
    _RUNTIME.close()
