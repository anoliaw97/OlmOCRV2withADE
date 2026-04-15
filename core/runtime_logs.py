from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime


VALID_LOG_KINDS = ("status", "debug", "error", "reasoning")


@dataclass(slots=True)
class RuntimeLogItem:
    time: str
    kind: str
    message: str


class RuntimeLogs:
    def __init__(self, max_items_per_kind: int = 1000) -> None:
        self._store = {
            "status": deque(maxlen=max_items_per_kind),
            "debug": deque(maxlen=max_items_per_kind),
            "error": deque(maxlen=max_items_per_kind),
            "reasoning": deque(maxlen=max_items_per_kind),
        }

    def add(self, kind: str, message: str) -> None:
        safe_kind = kind.strip().lower()
        if safe_kind not in self._store:
            safe_kind = "debug"
        item = RuntimeLogItem(
            time=datetime.now().strftime("%H:%M:%S"),
            kind=safe_kind,
            message=str(message),
        )
        self._store[safe_kind].append(item)

    def list(self, kind: str, limit: int = 200) -> list[RuntimeLogItem]:
        safe_limit = max(1, min(int(limit), 1500))
        safe_kind = kind.strip().lower()
        if safe_kind not in self._store:
            safe_kind = "status"
        items = list(self._store[safe_kind])
        return items[-safe_limit:]

    def clear(self, kind: str = "all") -> None:
        safe_kind = kind.strip().lower()
        if safe_kind == "all":
            for key in self._store:
                self._store[key].clear()
            return
        if safe_kind in self._store:
            self._store[safe_kind].clear()
